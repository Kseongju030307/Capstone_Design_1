import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from torchvision.utils import save_image
import open_clip

# ====== 설정 경로 ======
images_path = "/media/hdd/hahyeon/Capstone_Design_1/test_image/images"
labels_path = "/media/hdd/hahyeon/Capstone_Design_1/test_image/labels"
category_file = "/media/hdd/hahyeon/open_clip/dataset/dataset_category_original.txt"
checkpoint_path = "/media/hdd/hahyeon/open_clip/model_checkpoint/openclip_step_thin175000.pth"
save_dir = '/media/hdd/hahyeon/open_clip/ppt_image'
output_dir = "cropped_objects"
yolo_model_path = "/media/hdd/hahyeon/Capstone_Design_1/yolov11/weight/best.pt"

# ====== Category 불러오기 ======
with open(category_file, 'r') as f:
    categories = [line.strip() for line in f.readlines()]

# ====== 모델 로드 (YOLO + OpenCLIP) ======
yolo_model = YOLO(yolo_model_path)

clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
clip_model.visual.output_dim = 768
classifier_head = torch.nn.Linear(768, 100)  # 클래스 수에 맞게 조정
model = torch.nn.Sequential(clip_model.visual, classifier_head)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ====== Transform 정의 ======
transform = transforms.Compose([
    # 이미지 리사이즈 (224x224)
    transforms.Lambda(lambda img: img.resize((224, 224), Image.NEAREST)),

    # 첫 번째 블러 + 이진화
    transforms.Lambda(lambda img: Image.fromarray(
        cv2.threshold(
            cv2.GaussianBlur(np.array(img), (7, 7), 0),
            220, 255, cv2.THRESH_BINARY)[1] 
    )),

    # 두 번째 블러 + 이진화
    transforms.Lambda(lambda img: Image.fromarray(
        cv2.threshold(
            cv2.GaussianBlur(np.array(img), (5, 5), 0),
            230, 255, cv2.THRESH_BINARY)[1] 
    )),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])
])

os.makedirs(save_dir, exist_ok=True)

# ====== 정확도 계산 ======
total_matched = 0
total_expected = 0

for idx in range(100):
    image_name = f"composite_{idx:03d}.png"
    label_name = f"composite_{idx:03d}.txt"

    image_path = os.path.join(images_path, image_name)
    label_path = os.path.join(labels_path, label_name)

    image = cv2.imread(image_path)

    # YOLO로 객체 탐지
    results = yolo_model(image, verbose=False)[0]
    cropped_images = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        cropped_images.append(pil_image)

    # 정답 인덱스 로드
    with open(label_path, 'r') as f:
        true_labels = [int(line.strip()) for line in f.readlines()]

    matched = 0  # 이 이미지에서 맞춘 개수

    # cropped 이미지들을 분류하고 정답과 비교
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        for i, cropped_img in enumerate(cropped_images):
            image_tensor = transform(cropped_img).unsqueeze(0).to(device)
            save_image(image_tensor, os.path.join(save_dir, f"{idx:03d}_{i}.png"))

            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()

            if pred_class in true_labels:
                matched += 1

    # 이 composite 이미지의 정확도 계산
    image_accuracy = matched / len(true_labels) if true_labels else 0
    print(f"[{image_name}] Accuracy: {image_accuracy:.2%} ({matched}/{len(true_labels)})")

    total_matched += matched
    total_expected += len(true_labels)

# 전체 평균 정확도
overall_accuracy = total_matched / total_expected if total_expected > 0 else 0
print(f"\n✅ Overall Accuracy: {overall_accuracy:.2%} ({total_matched}/{total_expected})")