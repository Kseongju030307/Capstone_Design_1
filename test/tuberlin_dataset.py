import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class TUBerlinDataset(Dataset):
    def __init__(self, split='train', transform=None, seed=42):
        self.default_data_dir = "/media/hdd/hahyeon/open_clip/dataset/tu_berlin_dataset"
        self.default_category_file = "/media/hdd/hahyeon/open_clip/dataset/dataset_category.txt"

        self.zeroshot_data_dir = "/media/hdd/hahyeon/open_clip/dataset/tu_berlin_zeroshot"
        self.zeroshot_category_file = "/media/hdd/hahyeon/open_clip/dataset/dataset_category_zeroshot.txt"

        # ✅ 고정 시드를 설정해 재현성 보장
        random.seed(seed)

        if split == 'zeroshot':
            self.data_dir = self.zeroshot_data_dir
            category_file = self.zeroshot_category_file
        else:
            self.data_dir = self.default_data_dir
            category_file = self.default_category_file

        # ✅ 카테고리 목록 로드
        with open(category_file, 'r') as f:
            self.categories = [line.strip() for line in f.readlines()]

        self.data = []
        self.labels = []

        for class_idx, category in enumerate(self.categories):
            category_dir = os.path.join(self.data_dir, category)
            if not os.path.isdir(category_dir):
                print(f"⚠️ 폴더 없음: {category_dir}")
                continue

            image_paths = [os.path.join(category_dir, fname)
                           for fname in os.listdir(category_dir)
                           if fname.endswith('.png')]

            random.shuffle(image_paths)
            total = len(image_paths)
            train_end = int(0.8 * total)
            val_end = int(0.9 * total)

            if split == 'train':
                split_paths = image_paths[:train_end]
            elif split == 'val':
                split_paths = image_paths[train_end:val_end]
            elif split == 'test':
                split_paths = image_paths[val_end:]
            elif split == 'zeroshot':
                split_paths = image_paths  # 모든 이미지 사용
            else:
                raise ValueError("split must be one of ['train', 'val', 'test', 'zeroshot']")

            self.data.extend(split_paths)
            self.labels.extend([class_idx] * len(split_paths))  # 정수 라벨 저장

        self.classes = self.categories

        self.transform = transform or transforms.Compose([
            transforms.Lambda(lambda img: img.resize((256, 256), Image.NEAREST)),
            transforms.Lambda(lambda img: Image.fromarray(
                cv2.threshold(
                    cv2.GaussianBlur(np.array(img), (7, 7), 0),
                    220, 255, cv2.THRESH_BINARY
                )[1]
            )),
            transforms.Lambda(lambda img: Image.fromarray(
                cv2.threshold(
                    cv2.GaussianBlur(np.array(img), (5, 5), 0),
                    230, 255, cv2.THRESH_BINARY
                )[1]
            )),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.458, 0.408],
                                 std=[0.269, 0.261, 0.276])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label
