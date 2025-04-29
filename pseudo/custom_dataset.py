import os
import random
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(
        self,
        split='train',                 # 'train', 'val', or 'test'
        root_dir='/media/hdd/hahyeon/open_clip/new_dataset',
        transform=None,
        seed=42,
        test_val_ratio=0.5            # 비율 설정 (validation : test)
    ):
        self.root_dir = root_dir
        self.split = split
        random.seed(seed)

        # 카테고리 목록 로드
        cat_file = os.path.join(self.root_dir, 'categories.txt')
        with open(cat_file, 'r') as f:
            self.categories = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {c: i for i, c in enumerate(self.categories)}

        # 파일 리스트 초기화
        self.data = []  # (img_path, label_idx)

        # train split
        if split == 'train':
            txts = ['quickdraw_train.txt', 'sketch_train.txt']
            for txt in txts:
                path_txt = os.path.join(self.root_dir, txt)
                with open(path_txt, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 2:
                            continue
                        rel_path, lbl = parts[0], int(parts[1])
                        img_path = os.path.join(self.root_dir, rel_path)
                        self.data.append((img_path, lbl))

        # val/test split: test.txt 파일들 결합 후 분할
        elif split in ['val', 'test']:
            txts = ['quickdraw_test.txt', 'sketch_test.txt']
            temp = []
            for txt in txts:
                path_txt = os.path.join(self.root_dir, txt)
                with open(path_txt, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 2:
                            continue
                        rel_path, lbl = parts[0], int(parts[1])
                        img_path = os.path.join(self.root_dir, rel_path)
                        temp.append((img_path, lbl))
            # shuffle and split
            random.shuffle(temp)
            n_total = len(temp)
            n_val = int(n_total * test_val_ratio)
            if split == 'val':
                self.data = temp[:n_val]
            else:
                self.data = temp[n_val:]
        else:
            raise ValueError("split must be one of ['train','val','test']")

        # transform 정의
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.458, 0.408],
                                std=[0.269, 0.261, 0.276])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
