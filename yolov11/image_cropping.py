from ultralytics import YOLO
import cv2
import os

# 1. YOLOv11 모델 로드
model = YOLO(r"F:\종합설계\yolo_newdata\runs\detect\train\weights\best.pt")  # finetuning된 모델 경로로 변경

# 2. 이미지 로드
image_path = r"F:\종합설계\test\combined_2.jpg"
image = cv2.imread(image_path)

# 3. 이미지 탐지
results = model(image)[0]  # 결과는 리스트로 반환되므로 [0]으로 첫 번째 결과 가져옴

# 4. 바운딩 박스 기반으로 객체 크롭
output_dir = "cropped_objects"
os.makedirs(output_dir, exist_ok=True)

for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)

    cropped = image[y1:y2, x1:x2]
    output_path = os.path.join(output_dir, f"object_{i+1}_{class_name}.jpg")
    cv2.imwrite(output_path, cropped)

print(f"{len(results.boxes)} objects saved in '{output_dir}'")
