import os
import torch
import random
from torch.utils.data import DataLoader
from quickdraw_dataset2 import QuickDrawDataset
import open_clip

# 1. 모델 및 토큰 설정
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')

# 학습된 모델을 위한 classifier head 추가
num_classes = 100
model.visual.output_dim = 768
classifier_head = torch.nn.Linear(768, num_classes)

# 모델 결합
model = torch.nn.Sequential(model.visual, classifier_head)

# 체크포인트 불러오기
checkpoint_path = "/media/hdd/hahyeon/open_clip/model_checkpoint/openclip_step_thin175000.pth"
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# 카테고리 로딩
category_file = "/media/hdd/hahyeon/open_clip/dataset/dataset_category.txt"
with open(category_file, 'r') as f:
    categories = [line.strip() for line in f.readlines()]

# 2. 데이터셋에서 랜덤 이미지 선택

test_dataset = QuickDrawDataset(split="test")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3. 예측 수행 (10개 배치만 처리)
correct = 0
total = 0
class_correct = [0] * len(categories)
class_total = [0] * len(categories)

with torch.no_grad(), torch.amp.autocast("cuda"):
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        if batch_idx >= 2000:
            break

        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)

        # 예측 결과와 정답 비교
        preds = torch.argmax(probs, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 클래스별 정확도 집계
        for i in range(len(images)):
            class_total[labels[i].item()] += 1
            if preds[i] == labels[i]:
                class_correct[labels[i].item()] += 1

            # 상위 10개 확률 및 라벨 출력
            if i == 0: 
                top10_probs, top10_idx = torch.topk(probs[i], 4)
                print(f"🎯 실제 라벨: {categories[labels[i].item()]}")
                print("🔍 예측된 상위 10개 클래스 및 확률:")
                for idx, prob in zip(top10_idx, top10_probs):
                    print(f"{categories[idx.item()]}: {prob.item():.4f}")
                print("-----------------------")

# 4. 정확도 계산 및 출력
accuracy = (correct / total) * 100
print(f"✅ (총 {total}개 이미지) 테스트 데이터 정확도: {accuracy:.2f}% ")

# 5. 클래스별 정확도 출력
print("📌 클래스별 정확도:")
for i, category in enumerate(categories):
    if class_total[i] > 0:
        class_acc = (class_correct[i] / class_total[i]) * 100
        print(f"{category}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"{category}: 데이터 없음")
        
print("📌 정확도가 85% 이하인 클래스:")
low_accuracy_classes = []
for i, category in enumerate(categories):
    if class_total[i] > 0:
        class_acc = (class_correct[i] / class_total[i]) * 100
        if class_acc <= 85:
            low_accuracy_classes.append(f"{category}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

# 결과 출력
if low_accuracy_classes:
    print("\n".join(low_accuracy_classes))
else:
    print("모든 클래스가 85% 초과의 정확도를 가집니다! 🚀")