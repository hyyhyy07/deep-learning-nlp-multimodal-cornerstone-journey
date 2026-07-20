# 以后所有 PyTorch 项目都是这个结构

#```python
# 1. 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)  # 输出层
        return x

model = MyModel()

# 2. 定义损失函数和优化器
criterion = nn.MSELoss()        # 回归用 MSE
# criterion = nn.BCELoss()      # 二分类用 BCE
# criterion = nn.CrossEntropyLoss()  # 多分类用交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 比 SGD 更好

# 3. 训练循环 (这个结构永远不变!)
for epoch in range(epochs):
    y_pred = model(X_train)         # 前向
    loss = criterion(y_pred, y_train)  # 算损失

    optimizer.zero_grad()           # 清零梯度
    loss.backward()                 # 反向传播
    optimizer.step()                # 更新参数