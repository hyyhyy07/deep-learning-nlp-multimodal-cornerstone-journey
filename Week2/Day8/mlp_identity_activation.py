# 1. 定义无激活函数的模型配置
# 即使我们给很多神经元，如果没有非线性激活，它也学不到复杂边界
linear_mlp_config = (50, 50) 

# 2. 初始化模型
# activation='identity': 相当于 f(x) = x，即没有激活函数
mlp_linear = MLPClassifier(
    hidden_layer_sizes=linear_mlp_config, 
    activation='identity', 
    max_iter=2000, 
    random_state=42
)

# 3. 训练模型
print(f"正在训练无激活网络 (Identity) {linear_mlp_config}...")
mlp_linear.fit(X_train, y_train)

# 4. 评估结果
train_score_lin = mlp_linear.score(X_train, y_train)
test_score_lin = mlp_linear.score(X_test, y_test)

print("-" * 30)
print(f"模型结构: {linear_mlp_config} (Identity)")
print(f"训练集准确率: {train_score_lin:.4f}")
print(f"测试集准确率: {test_score_lin:.4f}")
print("-" * 30)

# 预期结果分析：
# 准确率通常会显著下降（可能在 90% 左右徘徊，很难达到 ReLU 的 97%+）。
# 结论：无论网络多宽或多深，如果中间没有非线性变换（如 ReLU/Sigmoid），
# 整个网络在数学上等价于单层线性模型（逻辑回归），无法处理复杂的非线性关系。