import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from torchvision.datasets import ImageFolder
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import warnings
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置超参数
batch_size = 1  # 增加batch_size有助于提高训练效率
learning_rate = 0.00005
num_epochs = 50
num_classes = 3  # 三分类

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小以适应ViT模型
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加颜色抖动
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)  # 添加随机擦除
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载数据集
train_dataset = ImageFolder(root=r'data/train', transform=transform)
test_dataset = ImageFolder(root=r'data/val', transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化ViT模型配置
config = ViTConfig(
    num_hidden_layers=12,  # 根据需要调整模型大小
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_size=768,
    num_labels=num_classes,  # 设置正确的类别数
    # hidden_dropout_prob=0.1,  # 添加 dropout
    # attention_probs_dropout_prob=0.1  # 添加 dropout
)

# 初始化模型
model = ViTForImageClassification(config)

# 加载预训练模型
pretrained_model_path = r'model\VIT-pretrain model\imagenet21k+imagenet2012_ViT-B_16-224.pth'  # 预训练模型路径
pretrained_dict = torch.load(pretrained_model_path, map_location=device)
model_dict = model.state_dict()

# 过滤掉不匹配的键值对（例如，分类头可能不同）
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# 更新模型的状态字典
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss
    avg_loss = running_loss / len(data_loader)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate metrics for each class
    sensitivities = []
    specificities = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return accuracy, sensitivities, specificities, precisions, recalls, f1_scores, avg_loss

# 修改训练循环
print(f"Using device: {device}")
model.to(device)

# Initialize lists to store losses
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    # Evaluate training set
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, test_loader, device)
    
    # Unpack metrics
    train_acc, train_sens, train_spec, train_prec, train_rec, train_f1, train_loss = train_metrics
    val_acc, val_sens, val_spec, val_prec, val_rec, val_f1, val_loss = val_metrics
    
    val_losses.append(val_loss)
    
    print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
    print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    print('\nValidation Results:')
    print(f'Accuracy: {val_acc*100:.2f}%')
    for i in range(num_classes):
        print(f'Class {i}: ', end='')
        print(f'Sensitivity: {val_sens[i]*100:.2f}% ', end='')
        print(f'Specificity: {val_spec[i]*100:.2f}% ', end='')
        print(f'Precision: {val_prec[i]*100:.2f}% ', end='')
        print(f'Recall: {val_rec[i]*100:.2f}% ', end='')
        print(f'F1-Score: {val_f1[i]*100:.2f}%')
    print('-' * 50)
    
    scheduler.step()

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('./loss_curve.png')
plt.close()

# 保存模型
torch.save(model.state_dict(), './model/transformer_model_3class_finetuned.pth')