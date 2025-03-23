from ultralytics import YOLO
import cv2
import os

# 1. YOLOv11 모델 로드
model = YOLO(r"F:\종합설계\yolo_newdata\runs\detect\train\weights\best.pt")

# 2. 이미지 로드
image_path = r"C:\Users\gky03\OneDrive\바탕 화면\Capstone_Design_1\yolov11\a1.png"
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

# 3. 이미지 탐지
results = model(image)[0]

# 4. 저장 폴더 준비
output_dir = "cropped_objects"
os.makedirs(output_dir, exist_ok=True)

count = 0  # 저장된 객체 수

for i, box in enumerate(results.boxes):
    conf = float(box.conf[0])
    if conf < 0.6:
        continue  # 60% 미만은 스킵

    # 바운딩 박스 좌표 추출 및 출력
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    print(f"[{i}] BBox 좌표: ({x1}, {y1}), ({x2}, {y2}) → 크기: {x2 - x1} x {y2 - y1}")

    # 클래스 이름 추출
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)

    # 객체 부분 crop
    cropped = image[y1:y2, x1:x2]
    output_path = os.path.join(output_dir, f"object_{count+1}_{class_name}.jpg")
    cv2.imwrite(output_path, cropped)

    # 원본 이미지에서 해당 영역 흰색으로 덮기
    image[y1:y2, x1:x2] = (255, 255, 255)

    print(f"Object {count+1}: Class = {class_name}, Confidence = {conf:.2f}")
    count += 1

# 5. 덮인 원본 이미지 저장
cv2.imwrite("masked_image.jpg", image)

print(f"\n✅ 총 {count}개의 객체가 저장되었고, 'masked_image.jpg'도 함께 저장되었습니다.")
