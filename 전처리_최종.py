import cv2
import numpy as np
import os
from PIL import Image


def get_average_line_thickness(binary_image):
    #이미지의 평균 선 굵기 측정
    edges = cv2.Canny(binary_image, 50, 150)  # Canny 엣지 검출
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness_list = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        thickness_list.append(min(w, h))  # 가로/세로 중 짧은 길이를 선 굵기로 가정
    
    if thickness_list:
        return int(np.mean(thickness_list))  # 평균 선 굵기 반환
    return 1  # 기본값 1px

def remove_salt_and_pepper_noise(image, kernel_size=3, iterations=6):
    # 미디언 필터를 여러 번 적용하여 Salt-and-Pepper 노이즈 제거
    denoised_image = image
    for _ in range(iterations):
        denoised_image = cv2.medianBlur(denoised_image, kernel_size)
    return denoised_image

def adjust_contrast(image, alpha=0.6, beta=0):
    # alpha: 대비, beta: 밝기
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


'''전처리이미지 출력함수'''
def preprocess_sketch(image_path, output_path, target_thickness=5):
    if not os.path.exists(image_path):
        print("파일이 존재하지 않습니다. 경로를 확인하세요.")
        return
    
    # 이미지 읽기 (한글 경로 지원)
    stream = open(image_path, "rb")
    bytes_data = bytearray(stream.read())
    np_array = np.asarray(bytes_data, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if image is None:
        print("이미지를 불러오지 못했습니다.")
        return
    
    # === 이미지 크기 조정 (640px, 비율 유지) ===
    height, width = image.shape[:2]
    new_size = 640

    if width > height:
        new_width = new_size
        new_height = int((new_width / width) * height)
    else:
        new_height = new_size
        new_width = int((new_height / height) * width)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # === 흑백 변환 ===
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # === 노이즈 제거 ===
    gray = remove_salt_and_pepper_noise(gray)  # Salt-and-Pepper 노이즈 제거

    # === 명암 대비 조정 ===
    gray = adjust_contrast(gray)  # 명암 대비 조정

    # === 블러 처리리 ===
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # === 적응형 이진화 ===
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # === 줄 제거 (Morphological Operations) ===
    kernel = np.ones((1, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    kernel = np.ones((3, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # === 현재 선 굵기 측정 ===
    current_thickness = get_average_line_thickness(binary)
    thickness_ratio = target_thickness / max(current_thickness, 1)  # 0으로 나누는 것 방지

    if thickness_ratio > 1:
        # 선이 너무 얇으면 팽창
        kernel_dilate = np.ones((7, 7), np.uint8)
        binary = cv2.dilate(binary, kernel_dilate, iterations=2)  # 팽창 횟수 증가
    elif thickness_ratio < 1:
        # 선이 너무 두꺼우면 침식
        kernel_erode = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel_erode, iterations=1)

    # === RGB 변환 ===
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # === 저장 ===
    result = Image.fromarray(binary_rgb)
    result.save(output_path)
    print(f"Processed image saved at: {output_path}")


'''이미지 출력코드입니다
preprocess_sketch(r"input이미지.jpg", r"output이미지.jpg")
'''