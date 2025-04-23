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
batch_size = 8  # 增加batch_size有助于提高训练效率
learning_rate = 0.0001
num_epochs = 50
num_classes = 3  # 三分类

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小以适应ResNet模型
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加颜色抖动
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 加入归一化
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)  # 添加随机擦除
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_dataset = ImageFolder(root=r'data/train_shuffle', transform=transform)
test_dataset = ImageFolder(root=r'data/val_shuffle', transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化ResNet18模型
model = models.resnet18(pretrained=True)

# 添加dropout层
for name, module in model.named_children():
    if isinstance(module, nn.Sequential):
        for n, m in module.named_children():
            if isinstance(m, nn.BatchNorm2d):
                # 在每个BatchNorm2d层后添加dropout
                new_sequential = nn.Sequential(
                    m,
                    nn.Dropout(p=0.3)  # 可以调整dropout率(0-1之间)
                )
                setattr(module, n, new_sequential)

# 修改最后的全连接层，加入dropout
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # 在全连接层前添加dropout
    nn.Linear(model.fc.in_features, num_classes)
)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

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
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均loss
    avg_loss = running_loss / len(data_loader)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 计算每个类别的评估指标
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
        recall = sensitivity  # recall 等同于 sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    return accuracy, sensitivities, specificities, precisions, recalls, f1_scores, avg_loss

# 初始化存储loss的列表
train_losses = []
val_losses = []

# 修改训练循环
print(f"Using device: {device}")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    # 评估训练集和验证集
    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, test_loader, device)
    
    # 解包评估指标
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

# 绘制loss曲线
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