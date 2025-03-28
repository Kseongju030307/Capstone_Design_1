import os
import torch
import random
from PIL import Image
from torchvision import transforms
import open_clip
from ultralytics import YOLO
import cv2
from torchvision.utils import save_image
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration

image_path = "/media/hdd/hahyeon/Capstone_Design_1/test_image/test5.jpg"
image_path = "/media/hdd/hahyeon/BLIP/sketch.png"

def image_crop():
    # 1. YOLOv11 모델 로드
    model = YOLO("/media/hdd/hahyeon/Capstone_Design_1/yolov11/weight/best.pt")  # finetuning된 모델 경로로 변경

    # 2. 이미지 로드
    image = cv2.imread(image_path)

    # 3. 이미지 탐지
    results = model(image)[0]  # 결과는 리스트로 반환되므로 [0]으로 첫 번째 결과 가져옴

    # 4. 바운딩 박스 기반으로 객체 크롭
    output_dir = "cropped_objects"
    os.makedirs(output_dir, exist_ok=True)

    images = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        images.append(pil_image)
        
    return images


def image_classification(images):
    category_file = "/media/hdd/hahyeon/open_clip/dataset/dataset_category.txt"
    with open(category_file, 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    
    # 1. 모델 및 토큰 설정
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')

    # 학습된 모델을 위한 classifier head 추가
    num_classes = 100
    model.visual.output_dim = 768
    classifier_head = torch.nn.Linear(768, num_classes)

    # 모델 결합
    model = torch.nn.Sequential(model.visual, classifier_head)

    # 체크포인트 불러오기
    checkpoint_path = "/media/hdd/hahyeon/open_clip/model_checkpoint/openclip_step90000.pth"
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(
            cv2.resize(
                cv2.threshold(
                    cv2.GaussianBlur(np.array(img), (15, 15), 0),  # 첫 번째 블러 적용
                    240, 255, cv2.THRESH_BINARY_INV  # 임계값 이진화
                )[1],
                (224, 224),  # 리사이즈 (보간 → 선 두껍게)
                interpolation=cv2.INTER_NEAREST
            )
        )),
        transforms.Lambda(lambda img: Image.fromarray(
            cv2.GaussianBlur(np.array(img), (27, 27), 0)  # 두 번째 블러 적용
        )),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])
    ])

    # 이미지 분류 및 출력
    save_dir = '/media/hdd/hahyeon/open_clip/ppt_image'
    os.makedirs(save_dir, exist_ok=True)
    predictCategories = []

    # 이미지 분류 및 저장
    with torch.no_grad(), torch.amp.autocast(device_type=device):
        for idx, image in enumerate(images):
            image_tensor = transform(image).unsqueeze(0).to(device)

            # 이미지 저장 (정규화 전 상태로 저장하기 위해 Normalize 이전 텐서 저장)
            # → transform 내 Normalize를 제외하고 따로 처리하거나, 저장 시 비정규화해줘야 함
            save_image(image_tensor, os.path.join(save_dir, f'transformed_image_{idx+1}.png'))

            # 분류
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            pred_prob = probs[0, pred_class].item()

            print(f"Image {idx + 1}: Predicted Class = {categories[pred_class]}, Probability = {pred_prob:.4f}")
            predictCategories.append(categories[pred_class])
            
    return predictCategories
            
            
def generate_caption(predictCategories):
    # 모델과 프로세서 로드
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto").to("cuda")

    # 이미지 불러오기
    image = Image.open(image_path).convert("RGB")

    # 질문 입력 및 처리
    question = f"This image contains {predictCategories[0]}, {predictCategories[1]}, and {predictCategories[2]}. 
        Please describe the image focusing on these objects and the overall scene."
    prompt = f"Question: {question} Answer:" 
    inputs = processor(image, return_tensors="pt").to("cuda")

    # 모델 예측 수행 (더 유연한 생성 방식 적용)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=10  # 답변 5개 생성
    )

    # 결과 출력 (질문 제외하고 답변만 출력)
    for i, output in enumerate(outputs):
        answer = processor.decode(output, skip_special_tokens=True).strip()
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        print(f"Answer {i+1}: {answer}")

images = image_crop()
predictCategories = image_classification(images)
generate_caption(predictCategories)