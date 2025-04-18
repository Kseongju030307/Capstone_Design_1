import os
import torch
import open_clip
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tuberlin_dataset import TUBerlinDataset  # TU-Berlin 데이터셋 클래스

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/media/hdd/hahyeon/open_clip/model_checkpoint/tu_clip_photo_coop_final.pth"

#########################################
# PromptLearner (CoOp 방식) 정의
#########################################
class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init="a photo of a"):
        """
        Args:
            clip_model: OpenCLIP 모델 (텍스트 임베딩 레이어 접근용)
            classnames: 클래스 이름 리스트 (예: test_dataset.classes)
            n_ctx: 학습할 컨텍스트 토큰의 수
            ctx_init: 초기 컨텍스트 텍스트
        """
        super().__init__()
        self.classnames = classnames
        self.n_ctx = n_ctx
        embed_dim = clip_model.token_embedding.weight.shape[1]

        # 초기 컨텍스트 토큰 임베딩
        if ctx_init:
            ctx_tokens = open_clip.tokenize(ctx_init).to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                ctx_embeddings = clip_model.token_embedding(ctx_tokens).squeeze(0)  # (L, embed_dim)
            if ctx_embeddings.shape[0] >= n_ctx:
                init_ctx = ctx_embeddings[:n_ctx].clone()
            else:
                pad = torch.zeros(n_ctx - ctx_embeddings.shape[0], embed_dim).to(ctx_embeddings.device)
                init_ctx = torch.cat([ctx_embeddings, pad], dim=0)
            self.ctx = nn.Parameter(init_ctx)
        else:
            self.ctx = nn.Parameter(torch.randn(n_ctx, embed_dim).to(clip_model.token_embedding.weight.device))

        # 각 클래스별 텍스트(클래스 이름) 토큰화
        self.class_tokenized = []
        for cname in self.classnames:
            tokenized = open_clip.tokenize(cname).to(clip_model.token_embedding.weight.device)
            self.class_tokenized.append(tokenized.squeeze(0))  # (L,)

    def forward(self, clip_model):
        """
        각 클래스에 대해 [SOS] 토큰 + learned context + 클래스 이름 토큰으로 프롬프트 생성.
        반환 텐서의 shape: (num_classes, prompt_length, embed_dim)
        """
        prompts = []
        sos_token = clip_model.token_embedding.weight[0].unsqueeze(0)  # (1, embed_dim)
        for tokenized_class in self.class_tokenized:
            with torch.no_grad():
                class_embeddings = clip_model.token_embedding(tokenized_class)  # (L, embed_dim)
            prompt = torch.cat([sos_token, self.ctx, class_embeddings], dim=0)  # (1+n_ctx+L, embed_dim)
            prompts.append(prompt)
        prompts = torch.stack(prompts, dim=0)  # (num_classes, prompt_length, embed_dim)
        return prompts

#########################################
# 텍스트 임베딩 생성 함수
#########################################
def encode_text_embeddings(clip_model, prompt_embeddings):
    """
    Args:
        prompt_embeddings: (num_classes, prompt_length, embed_dim)
    Returns:
        text_features: (num_classes, feature_dim)
    """
    # CLIP의 context_length (예: 77)보다 긴 경우 자름
    context_length = clip_model.context_length
    if prompt_embeddings.shape[1] > context_length:
        prompt_embeddings = prompt_embeddings[:, :context_length, :]
    # positional embedding 추가
    pos_embed = clip_model.positional_embedding[:context_length]
    x = prompt_embeddings + pos_embed
    # Transformer 처리 (입력 shape: (num_classes, context_length, embed_dim))
    x = clip_model.transformer(x)
    x = clip_model.ln_final(x)
    # 보통 첫 토큰(SOS)의 출력을 사용하여 텍스트 프로젝션 적용
    x = x[:, 0, :] @ clip_model.text_projection
    return x

def main():
    # 모델 및 전처리 로드
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B/32-quickgelu", pretrained="openai")
    clip_model = clip_model.to(device)
    
    # 테스트 데이터셋 로드 (split 이름은 'test' 또는 'val'로 조정)
    test_dataset = TUBerlinDataset(split='zeroshot', transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 전체 클래스 정보
    all_classes = test_dataset.classes

    # 학습 시 사용한 PromptLearner 생성
    prompt_learner = PromptLearner(clip_model, all_classes, n_ctx=4, ctx_init="a photo of a").to(device)
    
    # 체크포인트에서 모델 불러오기
    checkpoint = torch.load(checkpoint_path, map_location=device)
    clip_model.load_state_dict(checkpoint["clip_model_state_dict"])
    prompt_learner.load_state_dict(checkpoint["prompt_learner_state_dict"])
    
    clip_model.eval()
    prompt_learner.eval()
    
    # 학습된 프롬프트를 통해 텍스트 피처 생성 (한 번만 수행)
    with torch.no_grad():
        learned_prompts = prompt_learner(clip_model)  # (num_classes, prompt_length, embed_dim)
        text_features = encode_text_embeddings(clip_model, learned_prompts)  # (num_classes, feature_dim)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 테스트 루프
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            # 이미지 피처 추출
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # 이미지-텍스트 유사도 계산
            similarity = image_features @ text_features.T  # (batch_size, num_classes)
            predicted = similarity.argmax(dim=1)
            
            # 정답 인덱스 처리 (labels가 문자열이면 인덱스로 변환)
            if isinstance(labels[0], str):
                true_labels = torch.tensor([all_classes.index(lbl) for lbl in labels]).to(device)
            else:
                true_labels = labels.to(device)
                
            correct += (predicted == true_labels).sum().item()
            total += images.size(0)
    
    accuracy = 100 * correct / total
    print(f"[✓] Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
