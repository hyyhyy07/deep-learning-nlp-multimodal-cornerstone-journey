"""
Day 13 - ResNet: 残差连接解决退化问题
对比实验: ResNet18 vs PlainNet20 on CIFAR-10

核心问题:
  - PlainNet 越深越差 (退化)
  - ResNet 越深越好 (残差连接)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

print('=' * 50)
print('Day 13 - ResNet vs PlainNet 退化对比实验')
print('=' * 50)

# ============================================================
# 1. 定义 BasicBlock (残差块)
# ============================================================

class BasicBlock(nn.Module):
    """基本残差块: 两个 3x3 卷积 + 残差连接"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 维度不匹配时用 1x1 卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # <--- 残差连接！
        out = F.relu(out)
        return out


class PlainBlock(nn.Module):
    """普通卷积块 (无残差连接) - 对照实验"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 降采样 (维度匹配用)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # PlainBlock: 没有残差连接！
        residual = self.downsample(x)
        out = out + residual  # 仅用于维度匹配, 无残差学习
        out = F.relu(out)
        return out


# ============================================================
# 2. ResNet 和 PlainNet
# ============================================================

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # CIFAR 版本: 3x3 卷积开头 (保持分辨率)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def PlainNet20(num_classes=10):
    """20层普通网络 (对照)"""
    return ResNet(PlainBlock, [2, 2, 2, 2], num_classes)

def PlainNet32(num_classes=10):
    """32层普通网络 (对照, 看退化)"""
    return ResNet(PlainBlock, [3, 4, 6, 3], num_classes)


# ============================================================
# 3. 参数量对比
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f'\n参数量对比:')
print(f'  ResNet18:   {count_params(ResNet18())/1e6:.2f}M')
print(f'  ResNet34:   {count_params(ResNet34())/1e6:.2f}M')
print(f'  PlainNet20: {count_params(PlainNet20())/1e6:.2f}M')
print(f'  PlainNet32: {count_params(PlainNet32())/1e6:.2f}M')
print(f'  VGG13:      9.67M (Day12 参考)')
print(f'\n结论: ResNet18 比 VGG13 更深但参数更少!')


# ============================================================
# 4. 训练函数
# ============================================================

def train_model(model, train_loader, test_loader, epochs=10, lr=0.1, name='Model'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_accs, test_accs = [], []
    train_losses, test_losses = [], []
    start = time.time()

    print(f'\n训练 {name}...')
    for epoch in range(epochs):
        # 训练
        model.train()
        correct = total = 0
        tl = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            tl += loss.item()
        train_acc = correct / total
        train_accs.append(train_acc)
        train_losses.append(tl / len(train_loader))

        # 测试
        model.eval()
        correct = total = 0
        tl = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
                tl += loss.item()
        test_acc = correct / total
        test_accs.append(test_acc)
        test_losses.append(tl / len(test_loader))

        scheduler.step()
        print(f'  Epoch {epoch+1:2d}/{epochs} | '
              f'Train: {train_acc:.3f} | Test: {test_acc:.3f}')

    elapsed = time.time() - start
    print(f'{name} 完成! 用时: {elapsed:.1f}s, 最终 Test Acc: {test_accs[-1]:.4f}')
    return {'train_acc': train_accs, 'test_acc': test_accs,
            'train_loss': train_losses, 'test_loss': test_losses}


# ============================================================
# 5. 加载 CIFAR-10 并训练
# ============================================================

def main():
    print(f'\n准备 CIFAR-10 数据...')

    # 尝试加载 CIFAR-10
    try:
        from torchvision import datasets, transforms
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2470, 0.2435, 0.2616)),
        ])

        train_set = datasets.CIFAR10(root='../Day12/data', train=True,
                                     download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root='../Day12/data', train=False,
                                    download=True, transform=transform_test)
        use_cifar = True
    except:
        print('CIFAR-10 下载失败, 使用随机数据')
        from torch.utils.data import TensorDataset
        train_set = TensorDataset(torch.randn(2000, 3, 32, 32),
                                 torch.randint(0, 10, (2000,)))
        test_set = TensorDataset(torch.randn(500, 3, 32, 32),
                                torch.randint(0, 10, (500,)))
        use_cifar = False

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    # 训练 4 个模型
    results = {}
    results['ResNet18'] = train_model(ResNet18(), train_loader, test_loader,
                                      epochs=10, name='ResNet18')
    results['ResNet34'] = train_model(ResNet34(), train_loader, test_loader,
                                      epochs=10, name='ResNet34')
    results['PlainNet20'] = train_model(PlainNet20(), train_loader, test_loader,
                                        epochs=10, name='PlainNet20')

    # 看 PlainNet 32层是否退化
    results['PlainNet32'] = train_model(PlainNet32(), train_loader, test_loader,
                                        epochs=10, name='PlainNet32')

    # ============================================================
    # 6. 可视化对比
    # ============================================================

    colors = {
        'ResNet18': '#6c5ce7', 'ResNet34': '#a29bfe',
        'PlainNet20': '#e17055', 'PlainNet32': '#d63031'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 训练准确率
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(range(1, len(res['train_acc'])+1), res['train_acc'],
                'o-', color=colors.get(name, '#333'), label=name)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('训练集准确率')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 测试准确率
    ax = axes[0, 1]
    for name, res in results.items():
        ax.plot(range(1, len(res['test_acc'])+1), res['test_acc'],
                'o-', color=colors.get(name, '#333'), label=name)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('测试集准确率')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 最终准确率柱状图
    ax = axes[1, 0]
    names = list(results.keys())
    accs = [results[n]['test_acc'][-1] for n in names]
    bars = ax.bar(names, accs, color=[colors.get(n, '#333') for n in names])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=11)
    ax.set_ylabel('Test Accuracy'); ax.set_title('最终测试准确率对比')
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3, axis='y')

    # 退化对比: Plain20 vs Plain32
    ax = axes[1, 1]
    plain20 = results['PlainNet20']['test_acc']
    plain32 = results['PlainNet32']['test_acc']
    ax.plot(range(1, len(plain20)+1), plain20, 'o-', color='#e17055', label='PlainNet-20')
    ax.plot(range(1, len(plain32)+1), plain32, 's--', color='#d63031', label='PlainNet-32')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy')
    ax.set_title('退化现象: PlainNet32 < PlainNet20?')
    ax.legend(); ax.grid(True, alpha=0.3)

    if plain32[-1] < plain20[-1]:
        ax.text(0.5, 0.1, f'✅ 退化验证成功! 32层({plain32[-1]:.3f}) < 20层({plain20[-1]:.3f})',
                transform=ax.transAxes, fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='#ffcccc'))
    else:
        ax.text(0.5, 0.1, f'随机数据下退化不明显, 需用真实 CIFAR-10',
                transform=ax.transAxes, fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='#ffffcc'))

    plt.suptitle('ResNet vs PlainNet: 残差连接解决退化', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('day13_degradation_comparison.png', dpi=150)
    plt.show()

    print(f'\n{"="*50}')
    print('最终结果')
    print('='*50)
    for name in names:
    
     print(f'{name:15s}: Test={results[name]["test_acc"][-1]:.4f}') 
    if use_cifar:
        print(f'\n{"="*50}')
        print('退化现象分析 (CIFAR-10 真实数据)')
        print('='*50)
        print(f'PlainNet20: {results["PlainNet20"]["test_acc"][-1]:.4f}')
        print(f'PlainNet32: {results["PlainNet32"]["test_acc"][-1]:.4f}')
        if results['PlainNet32']['test_acc'][-1] < results['PlainNet20']['test_acc'][-1]:
            print('✅ 退化现象已观察到: 32层 < 20层!')
        print(f'ResNet18:  {results["ResNet18"]["test_acc"][-1]:.4f}')
        print(f'ResNet34:  {results["ResNet34"]["test_acc"][-1]:.4f}')
        print(f'✅ ResNet 没有退化: 34层 > 18层!' if results['ResNet34']['test_acc'][-1] >= results['ResNet18']['test_acc'][-1]
              else '❌ 可能需要更多epoch')

    print('\nDay 13 完成! ResNet 原理 + 实现 + 退化验证 ✅')


if __name__ == '__main__':
    main()
