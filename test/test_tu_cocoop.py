import os
import torch
import open_clip
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tuberlin_dataset import TUBerlinDataset

# ─── 설정 ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "/media/hdd/hahyeon/open_clip/model_checkpoint"
checkpoint_path = os.path.join(save_dir, "cocoop_final.pth")

# ─── 모델 & 전처리 로드 ────────────────────────────────────────────
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B/32-quickgelu", pretrained="openai"
)
clip_model = clip_model.to(device)
clip_model.eval()

# ─── 테스트 데이터로더 ─────────────────────────────────────────────
test_dataset = TUBerlinDataset(split='zeroshot', transform=preprocess)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False,
    num_workers=4, pin_memory=True
)

# ─── CoCoOpPromptLearner (train 코드와 동일) ───────────────────────
class CoCoOpPromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init="a photo of a"):
        super().__init__()
        self.n_ctx = n_ctx
        self.embed_dim = clip_model.token_embedding.weight.shape[1]
        self.num_classes = len(classnames)

        # 1) Learnable base context
        if ctx_init:
            tokens = open_clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                emb = clip_model.token_embedding(tokens).squeeze(0)
            if emb.shape[0] >= n_ctx:
                init_ctx = emb[:n_ctx]
            else:
                pad = emb.new_zeros(n_ctx - emb.shape[0], emb.shape[1])
                init_ctx = torch.cat([emb, pad], dim=0)
            self.ctx = nn.Parameter(init_ctx)
        else:
            self.ctx = nn.Parameter(torch.randn(n_ctx, self.embed_dim))

        # 2) Meta‑network
        self.meta_net = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_ctx * self.embed_dim)
        )

        # 3) 클래스별 토큰 임베딩 (패딩 후 하나의 버퍼로)
        class_embs, lengths = [], []
        for cname in classnames:
            toks = open_clip.tokenize(cname).to(device)
            with torch.no_grad():
                emb = clip_model.token_embedding(toks).squeeze(0)
            class_embs.append(emb); lengths.append(emb.shape[0])
        L_max = max(lengths)
        padded = []
        for emb in class_embs:
            if emb.shape[0] < L_max:
                pad = emb.new_zeros(L_max - emb.shape[0], emb.shape[1])
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)
        # (C, L_max, D)
        self.register_buffer("class_embeddings", torch.stack(padded, dim=0))

    def forward(self, image_features):
        """
        image_features: (B, D)
        returns prompts: (B, C, 1+n_ctx+L_max, D)
        """
        B, D = image_features.shape
        C, L = self.class_embeddings.shape[:2]

        # dynamic context
        bias = self.meta_net(image_features).view(B, self.n_ctx, D)
        ctx = self.ctx.unsqueeze(0) + bias             # (B, n_ctx, D)

        # SOS token
        sos = clip_model.token_embedding.weight[0].view(1,1,D).expand(B,C,1,D)
        # expand ctx & class
        ctx = ctx.unsqueeze(1).expand(-1, C, -1, -1)    # (B, C, n_ctx, D)
        cls = self.class_embeddings.unsqueeze(0).expand(B, -1, -1, -1)  # (B, C, L, D)

        # concat → (B, C, 1+n_ctx+L, D)
        prompts = torch.cat([sos, ctx, cls], dim=2)
        return prompts

# ─── 텍스트 임베딩 인코딩 (train 코드와 동일, 이어짐) ─────────────────────
def encode_text_embeddings(clip_model, prompt_embeddings, chunk_size=64):
    context_length = clip_model.context_length
    prompt_embeddings = prompt_embeddings[:, :context_length, :]
    pos_embed = clip_model.positional_embedding[:context_length]

    outs = []
    for i in range(0, prompt_embeddings.shape[0], chunk_size):
        chunk = prompt_embeddings[i:i+chunk_size] + pos_embed.unsqueeze(0)
        x = clip_model.transformer(chunk)
        x = clip_model.ln_final(x)
        x = x[:, 0, :] @ clip_model.text_projection
        outs.append(x)
    return torch.cat(outs, dim=0)

# ─── 프롬프트 학습기 로드 ───────────────────────────────────────────────
prompt_learner = CoCoOpPromptLearner(
    clip_model, test_dataset.classes, n_ctx=4, ctx_init="a photo of a"
).to(device)

def remove_orig_mod_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    return new_state_dict

# 체크포인트 로드
# ↓ 전체 state_dict 대신 필요한 것만 뽑기
loaded = torch.load(checkpoint_path, map_location=device)
ckpt = remove_orig_mod_prefix(loaded["prompt_learner"])

# 현재 모델의 state_dict 받아오기
model_dict = prompt_learner.state_dict()

# 필요한 key만 남김 (ctx, meta_net)
filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_dict and v.size() == model_dict[k].size()}

# 현재 모델 dict 업데이트
model_dict.update(filtered_ckpt)

# 반영
prompt_learner.load_state_dict(model_dict)
prompt_learner.eval()

# ─── 테스트 루프 ──────────────────────────────────────────────────────
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Test"):
        images = images.to(device)
        labels = labels.to(device)

        # 1. 이미지 피처 인코딩
        img_feats = clip_model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        # 2. 동적 프롬프트 생성
        prompts = prompt_learner(img_feats)
        B, C, L, D = prompts.shape

        # 3. 텍스트 임베딩
        flat = prompts.view(B*C, L, D)
        txt_feats = encode_text_embeddings(clip_model, flat)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats.view(B, C, -1)

        # 4. 유사도 계산 및 예측
        logits = (img_feats.unsqueeze(1) * txt_feats).sum(-1)  # (B, C)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += B

# ─── 최종 정확도 출력 ────────────────────────────────────────────────
acc = 100 * correct / total
print(f"▶ Test Accuracy: {acc:.2f}%")