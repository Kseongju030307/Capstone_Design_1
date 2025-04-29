import os
import torch
import open_clip
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from custom_dataset import CustomDataset  # 수정된 이름으로 import

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#checkpoint_path = "/media/hdd/hahyeon/open_clip/model_checkpoint/tu_clip_photo_final.pth"

# ✅ 모델 로드 (pretrained CLIP만 사용)
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B/32-quickgelu", pretrained="openai"
)
# (필요시 체크포인트 로드)
# clip_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
clip_model = clip_model.to(device)
clip_model.eval()

# ✅ 데이터셋 로딩 (custom test split)
test_dataset = CustomDataset(
    split='test',
    root_dir='/media/hdd/hahyeon/open_clip/new_dataset',
    transform=preprocess
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# ✅ 클래스 기준 텍스트 토큰화 (한 번만)
all_classes = test_dataset.categories
all_captions = [f"A Sketch of a {cls}" for cls in all_classes]
all_text_tokens = open_clip.tokenize(all_captions).to(device)

with torch.no_grad():
    all_text_features = clip_model.encode_text(all_text_tokens)
    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

# ✅ 예측 및 정답 수집
all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        images = images.to(device)

        # 이미지 피처
        img_feats = clip_model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        # 유사도 계산 & 예측
        sim = img_feats @ all_text_features.T
        preds = sim.argmax(dim=1).cpu().numpy()

        # 정답 처리
        true = labels.cpu().numpy()

        all_preds.extend(preds.tolist())
        all_targets.extend(true.tolist())

# ✅ 지표 계산
accuracy  = accuracy_score(all_targets, all_preds)
precision, recall, _, _ = precision_recall_fscore_support(
    all_targets, all_preds, average='macro', zero_division=0
)

print(f"[✓] Test Accuracy     : {accuracy  * 100:.2f}%")
print(f"[✓] Macro Precision   : {precision * 100:.2f}%")
print(f"[✓] Macro Recall      : {recall    * 100:.2f}%")
