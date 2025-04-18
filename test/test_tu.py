import os
import torch
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from tuberlin_dataset import TUBerlinDataset

# ✅ 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/media/hdd/hahyeon/open_clip/model_checkpoint/tu_clip_photo_final.pth"

# ✅ 모델 로드
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B/32-quickgelu", pretrained="openai")
clip_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
clip_model = clip_model.to(device)
clip_model.eval()

# ✅ 데이터셋 로딩 (test split 사용)
test_dataset = TUBerlinDataset(split='zeroshot', transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# ✅ 전체 클래스 기준 텍스트 토큰화 (한 번만 수행)
all_classes = test_dataset.classes
all_captions = [f"A Photo of a {cls}" for cls in all_classes]
all_text_tokens = open_clip.tokenize(all_captions).to(device)

with torch.no_grad():
    all_text_features = clip_model.encode_text(all_text_tokens)
    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

# ✅ 테스트 루프
correct = 0
total = 0

with torch.no_grad():
    test_bar = tqdm(test_loader, desc="Evaluating", leave=False)

    for images, labels in test_bar:
        images = images.to(device)

        # ✅ 이미지 피처 추출
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # ✅ 유사도 계산
        similarity = image_features @ all_text_features.T
        predicted = similarity.argmax(dim=1)

        # ✅ 정답 인덱스 처리
        if isinstance(labels[0], str):
            true_labels = torch.tensor([all_classes.index(lbl) for lbl in labels]).to(device)
        else:
            true_labels = labels.to(device)

        correct += (predicted == true_labels).sum().item()
        total += len(images)

accuracy = 100 * correct / total
print(f"[✓] Test Accuracy: {accuracy:.2f}%")
