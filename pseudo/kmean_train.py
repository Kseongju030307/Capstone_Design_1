import os
import random
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import open_clip
from custom_dataset import CustomDataset

# -------------------------------------
# 설정
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
save_dir = "/media/hdd/hahyeon/open_clip/pseudo/model_checkpoint"
os.makedirs(save_dir, exist_ok=True)

relabel_interval = 10
num_epochs = 40
batch_size_emb = 128
batch_size_train = 64

# -------------------------------------
# 1) 초기 모델 및 데이터 로드
# -------------------------------------
base_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B/32-quickgelu", pretrained="openai"
)
base_model = base_model.to(device).eval()
orig_train = CustomDataset(split='train', transform=preprocess)

class_names = orig_train.categories
prompts = [f"a sketch of a {c}" for c in class_names]
text_tokens = open_clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_feats = base_model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

# -------------------------------------
# Pseudo-Label 생성 함수 (개선된 버전)
# -------------------------------------
def generate_pseudo_labels(model, dataset, text_feats, n_clusters, top_k=3):
    model.eval()
    all_feats = []
    loader_emb = DataLoader(dataset, batch_size=batch_size_emb, shuffle=False, num_workers=4)
    for imgs, _ in tqdm(loader_emb, desc="Extracting Features"):
        feats = model.encode_image(imgs.to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().detach().numpy())
    all_feats = np.concatenate(all_feats, axis=0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(all_feats)
    centers = torch.from_numpy(kmeans.cluster_centers_).to(device)
    centers = centers / centers.norm(dim=-1, keepdim=True)

    sims = centers @ text_feats.T
    cluster2class = {}
    for i in range(n_clusters):
        sim_row = sims[i]
        topk_scores, topk_indices = sim_row.topk(top_k)
        avg_scores = torch.zeros(len(class_names), device=device)
        counts = torch.zeros(len(class_names), device=device)
        for j in range(top_k):
            cls_idx = topk_indices[j].item()
            avg_scores[cls_idx] += topk_scores[j].item()
            counts[cls_idx] += 1
        avg_scores = avg_scores / (counts + 1e-6)
        best_class = int(avg_scores.argmax().item())
        cluster2class[i] = best_class

    pseudo = [cluster2class[cid] for cid in kmeans.labels_]
    return pseudo

# -------------------------------------
# 초기 Pseudo-Label 생성
# -------------------------------------
cluster_multiplier = 5
n_clusters = len(class_names) * cluster_multiplier
pseudo_labels = generate_pseudo_labels(base_model, orig_train, text_feats, n_clusters)

# -------------------------------------
# Pseudo-Label Dataset 정의
# -------------------------------------
class PseudoDataset(Dataset):
    def __init__(self, base_ds, pseudo_lbls):
        self.base_ds = base_ds
        self.pseudo = pseudo_lbls
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        img, _ = self.base_ds[idx]
        return img, self.pseudo[idx]

train_dataset = PseudoDataset(orig_train, pseudo_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

# -------------------------------------
# 학습용 CLIP 모델 및 옵티마이저
# -------------------------------------
clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B/32-quickgelu", pretrained="openai"
)
clip_model = clip_model.to(device)
optimizer = optim.AdamW(clip_model.parameters(), lr=2e-6, weight_decay=0.2)
loss_fn = nn.CrossEntropyLoss()

val_dataset = CustomDataset(split='val', transform=preprocess)
val_loader = DataLoader(val_dataset, batch_size=batch_size_train, shuffle=False, num_workers=2)

# -------------------------------------
# Training Loop with Iterative Re-labeling
# -------------------------------------
for epoch in range(1, num_epochs + 1):
    clip_model.train()
    epoch_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False):
        images = images.to(device)
        captions = [f"a sketch of a {class_names[l]}" for l in labels]
        tokens = open_clip.tokenize(captions).to(device)

        img_feats = clip_model.encode_image(images)
        txt_feats = clip_model.encode_text(tokens)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        logits_i2t = (img_feats @ txt_feats.T) * 100
        logits_t2i = logits_i2t.T
        target = torch.arange(images.size(0), device=device)
        loss = (loss_fn(logits_i2t, target) + loss_fn(logits_t2i, target)) * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch}/{num_epochs}] Avg Loss: {epoch_loss/len(train_loader):.4f}")

    # Validation
    clip_model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            captions = [f"a sketch of a {class_names[l]}" for l in labels]
            tokens = open_clip.tokenize(captions).to(device)
            img_feats = clip_model.encode_image(images)
            txt_feats = clip_model.encode_text(tokens)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            sims = (img_feats @ txt_feats.T) * 100
            preds = sims.argmax(dim=1)
            correct += (preds == labels.to(device)).sum().item()
            total += images.size(0)
    print(f"[Val] Epoch [{epoch}/{num_epochs}] Acc: {100*correct/total:.2f}%")

    # Iterative Re-labeling
    if epoch % relabel_interval == 0 and epoch < num_epochs:
        print(f"Reassigning Pseudo-Labels at epoch {epoch}...")
        pseudo_labels = generate_pseudo_labels(clip_model, orig_train, text_feats, n_clusters)
        train_dataset = PseudoDataset(orig_train, pseudo_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

    # Checkpoint 저장
    if epoch % relabel_interval == 0:
        ckpt = os.path.join(save_dir, f"kmean_clip_epoch_{epoch}.pth")
        torch.save(clip_model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")

# 최종 모델 저장
final_path = os.path.join(save_dir, "kmean_final.pth")
torch.save(clip_model.state_dict(), final_path)
print(f"Final model saved at: {final_path}")