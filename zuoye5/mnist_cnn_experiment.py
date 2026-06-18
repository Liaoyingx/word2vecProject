import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import pandas as pd

# ------------------ 模型定义 ------------------
class BaselineCNN(nn.Module):
    def __init__(self, activation='relu', use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128*7*7, 10)

        # 激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('Unsupported activation')

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = self.act(x)

        x = x.view(-1, 128*7*7)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x

# ------------------ 训练与测试 ------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def run_experiment(exp_name, model_config, epochs=10, device='cpu'):
    print(f"\n===== Running: {exp_name} =====")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = BaselineCNN(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_acc': []}
    for epoch in range(1, epochs+1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        history['train_loss'].append(loss)
        history['test_acc'].append(acc)
        print(f"Epoch {epoch:2d} | Loss: {loss:.4f} | Test Acc: {acc:.2f}%")
    return history

# ------------------ 主程序 ------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 实验配置列表
    experiments = [
        ('Baseline (ReLU)', {'activation': 'relu', 'use_dropout': False, 'use_batchnorm': False}),
        ('Sigmoid', {'activation': 'sigmoid', 'use_dropout': False, 'use_batchnorm': False}),
        ('Tanh', {'activation': 'tanh', 'use_dropout': False, 'use_batchnorm': False}),
        ('ReLU + Dropout', {'activation': 'relu', 'use_dropout': True, 'use_batchnorm': False}),
        ('ReLU + BatchNorm', {'activation': 'relu', 'use_dropout': False, 'use_batchnorm': True}),
    ]

    results = []
    for name, config in experiments:
        hist = run_experiment(name, config, epochs=10, device=device)
        # 记录最终准确率和达到98%的轮数
        final_acc = hist['test_acc'][-1]
        try:
            epoch_98 = next(i+1 for i, acc in enumerate(hist['test_acc']) if acc >= 98.0)
        except StopIteration:
            epoch_98 = None
        results.append({
            'Experiment': name,
            'Final Test Acc (%)': final_acc,
            'Epoch to reach 98%': epoch_98,
            'Final Loss': hist['train_loss'][-1]
        })

    # 输出汇总表
    df = pd.DataFrame(results)
    print("\n========== Summary ==========")
    print(df.to_string(index=False))

    # 可选：保存详细损失/准确率曲线（省略绘图代码，但可添加）