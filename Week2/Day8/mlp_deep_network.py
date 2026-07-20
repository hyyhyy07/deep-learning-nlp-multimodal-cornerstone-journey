# 1. 定义深层网络配置
deep_config = (100, 100, 100)

# 2. 初始化模型
# max_iter=3000: 深层网络收敛可能较慢，适当增加迭代次数
mlp_deep = MLPClassifier(
    hidden_layer_sizes=deep_config, 
    activation='relu', 
    max_iter=3000, 
    random_state=42
)

# 3. 训练模型
print(f"正在训练深层网络 {deep_config}...")
mlp_deep.fit(X_train, y_train)

# 4. 评估并打印结果
train_score = mlp_deep.score(X_train, y_train)
test_score = mlp_deep.score(X_test, y_test)

print("-" * 30)
print(f"模型结构: {deep_config}")
print(f"训练集准确率: {train_score:.4f}")
print(f"测试集准确率: {test_score:.4f}")
print(f"差距 (过拟合程度): {train_score - test_score:.4f}")
print("-" * 30)

# 预期结果分析：
# 你会发现训练集准确率可能非常高（接近 1.0），但测试集准确率可能反而不如之前的简单模型（如 (20,) 或 (50,)）。
# 这就是典型的“过拟合”：模型死记硬背了训练数据，但在没见过的数据上表现变差。

