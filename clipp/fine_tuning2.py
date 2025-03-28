import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from quickdraw_dataset2 import QuickDrawDataset
from tqdm import tqdm
import open_clip

save_dir = "/media/hdd/hahyeon/open_clip/model_checkpoint"

def clip_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    logits_per_image = image_features @ text_features.T / temperature
    logits_per_text = text_features @ image_features.T / temperature

    labels = torch.arange(len(image_features), device=image_features.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2

class ClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def fine_tuning2():
    # 1. 모델 로드
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # 모델의 텍스트 인코더 파라미터 freeze
    for name, param in model.named_parameters():
        if 'text_model' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 100개 클래스 분류기 추가
    classifier = ClassificationHead(input_dim=768, num_classes=100).to(device)  # ViT-L-14의 feature dim은 768

    # 2. 데이터 로드
    train_dataset = QuickDrawDataset(split='train', transform=preprocess)
    val_dataset = QuickDrawDataset(split='val', transform=preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ✅ 텍스트 임베딩 캐싱
    print("🚀 텍스트 임베딩 캐싱 중...")
    text_embeddings = {}
    with torch.no_grad():
        for class_name in train_dataset.classes:
            token = tokenizer([class_name]).to(device)
            text_feat = model.encode_text(token)
            text_embeddings[class_name] = text_feat  # shape: (1, dim)
    print("✅ 텍스트 임베딩 완료.")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),  # 분류기도 학습
        lr=5e-5, 
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200000, last_epoch=-1)

    num_epochs = 3  # 원하면 더 늘릴 수 있음
    global_step = 0  # Initialize global_step

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        total_cls_loss = 0.0
        model.train()
        classifier.train()
        progress_bar = tqdm(train_dataloader, desc=f"[Train | Epoch {epoch}]", unit="batch")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # 텍스트 임베딩을 미리 캐시된 값으로 대체
            text_features = torch.cat([text_embeddings[train_dataset.classes[label]] for label in labels], dim=0).to(device)

            image_features = model.encode_image(images)

            # CLIP loss
            loss_clip = clip_loss(image_features, text_features)

            # 100개 클래스 분류 loss 추가
            logits = classifier(image_features)
            loss_cls = F.cross_entropy(logits, labels)

            # 총 loss = CLIP loss + 분류 loss
            loss = loss_clip + loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss_clip.item()
            total_cls_loss += loss_cls.item()
            global_step += 1
            progress_bar.set_postfix(
                clip_loss=f"{total_loss / len(train_dataloader):.6f}",
                cls_loss=f"{total_cls_loss / len(train_dataloader):.6f}"
            )

            if global_step % 5000 == 0:
                save_path = os.path.join(save_dir, f"openclip_step_loss{global_step}.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                }, save_path)
                print(f"✅ 모델 저장 완료: {save_path}")

        scheduler.step()

        # ✅ Validation 루프
        model.eval()
        classifier.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                # 텍스트 임베딩을 미리 캐시된 값으로 대체
                text_features = torch.cat([text_embeddings[val_dataset.classes[label]] for label in labels], dim=0).to(device)

                image_features = model.encode_image(images)

                # CLIP loss
                loss_clip = clip_loss(image_features, text_features)

                # 100개 클래스 분류 loss
                logits = classifier(image_features)
                loss_cls = F.cross_entropy(logits, labels)

                val_loss += loss_clip.item()
                val_cls_loss += loss_cls.item()

                # 정확도 계산
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_cls_loss = val_cls_loss / len(val_dataloader)
        accuracy = 100 * correct / total

        print(f"📊 [Epoch {epoch}] Validation Loss: {avg_val_loss:.6f}, Classification Loss: {avg_val_cls_loss:.6f}, Accuracy: {accuracy:.2f}%")

    # ✅ 최종 모델 저장
    save_path = os.path.join(save_dir, "openclip_final2.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
    }, save_path)
    print(f"✅ 최종 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    fine_tuning2()
