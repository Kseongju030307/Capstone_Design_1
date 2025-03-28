import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch

class QuickDrawDataset(Dataset):
    def __init__(self, split='train', transform=None, max_samples_per_class=None):
        self.data_dir = "/media/hdd/hahyeon/open_clip/dataset/quickdraw_dataset"
        self.category_file = "/media/hdd/hahyeon/open_clip/dataset/dataset_category.txt"

        with open(self.category_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.data = []
        self.labels = []

        for idx, category in enumerate(self.classes):
            file_path = os.path.join(self.data_dir, f"{category}.npy")
            if os.path.exists(file_path):
                sketches = np.load(file_path)

                if max_samples_per_class is not None:
                    sketches = sketches[:max_samples_per_class]
                else:
                    sketches = sketches[:80000]

                for sketch in sketches:
                    self.data.append(sketch)
                    self.labels.append(idx)

        # ✅ 셔플
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined)
        self.data = list(self.data)
        self.labels = list(self.labels)

        # ✅ 8:1:1 split
        total_size = len(self.data)
        train_end = int(total_size * 0.8)
        val_end = int(total_size * 0.9)

        if split == 'train':
            self.data = self.data[:train_end]
            self.labels = self.labels[:train_end]
        elif split == 'val':
            self.data = self.data[train_end:val_end]
            self.labels = self.labels[train_end:val_end]
        elif split == 'test':
            self.data = self.data[val_end:]
            self.labels = self.labels[val_end:]
        else:
            raise ValueError(f"split must be one of ['train', 'val', 'test'], got {split}")

        # ✅ transform 설정
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.LANCZOS),
            transforms.Lambda(lambda img: img.point(lambda p: 0 if p > 220 else 255)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sketch = self.data[idx].reshape(28, 28)
        label = self.labels[idx]

        sketch_img = Image.fromarray(sketch.astype(np.uint8), mode='L').convert("RGB")

        if self.transform:
            sketch_img = self.transform(sketch_img)

        return sketch_img, label
