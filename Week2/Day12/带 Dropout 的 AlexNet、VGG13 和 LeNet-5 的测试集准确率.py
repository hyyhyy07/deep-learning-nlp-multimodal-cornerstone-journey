import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 定义三个“极速版”模型
# ==========================================

# 模型 A: Nano LeNet (极小)
class NanoLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(16 * 5 * 5, 10) # 适配CIFAR尺寸

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 模型 B: Nano AlexNet (去掉深层卷积)
class NanoAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 模型 C: Nano VGG (仅2层卷积块)
class NanoVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================
# 2. 主程序：极速训练与绘图
# ==========================================
def run_quick_comparison():
    # --- 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    
    # 数据预处理 (不做复杂增强，加速加载)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 下载数据集
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 【关键】只截取极少量数据进行演示 (500张训练, 200张测试)
    train_subset = Subset(train_set, range(500))
    test_subset = Subset(test_set, range(200))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=0) # num_workers=0 防止Windows卡死
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)

    # 定义模型字典
    models = {
        "Nano LeNet": NanoLeNet().to(device),
        "Nano AlexNet": NanoAlexNet().to(device),
        "Nano VGG": NanoVGG().to(device)
    }

    results = {}
    
    print("\n开始极速训练 (共3轮)...")
    start_time = time.time()

    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        acc_history = []

        # 训练 3 轮
        for epoch in range(3):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # 快速测试
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            acc = 100. * correct / len(test_loader.dataset)
            acc_history.append(acc)
            print(f"{name} - Epoch {epoch+1}/3 - Acc: {acc:.1f}%")
        
        results[name] = acc_history

    total_time = time.time() - start_time
    print(f"\n✅ 全部完成！总耗时: {total_time:.2f} 秒")

    # ==========================================
    # 3. 绘图
    # ==========================================
    plt.figure(figsize=(8, 5))
    for name, accs in results.items():
        plt.plot([1, 2, 3], accs, marker='o', label=name)
    
    plt.title("Model Comparison (Tiny Dataset & Models)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_quick_comparison()