# 🧠 35 天 ML + DL 完整大纲

> 5 周，从 NumPy 到扩散模型，一条线 ✅ 全 35 天完结

---

## 全景

```
第1周  Day 1-7   经典 ML          ████████████ ✅
第2周  Day 8-14  神经网络         ████████████ ✅
第3周  Day 15-21 序列 + Transformer ████████████ ✅
第4周  Day 22-28 预训练 + 生成模型  ████████████ ✅
第5周  Day 29-35 实战 + 收尾       ████████████ ✅
```

---

## 第1周 · 经典 ML 手写

| Day | 内容 | 核心 | 状态 |
|-----|------|------|:----:|
| **1** | NumPy + Pandas + Matplotlib | 向量化、广播、数据探索 | ✅ |
| **2** | 线性回归 | 手写梯度下降、损失曲线 | ✅ |
| **3** | 逻辑回归 | Sigmoid + 交叉熵 + 决策边界 | ✅ |
| **4** | 决策树 | 基尼系数、可视化、过拟合 | ✅ |
| **5** | 集成学习 | 随机森林 vs XGBoost vs Bagging vs Boosting | ✅ |
| **6** | K-Means + PCA | 无监督聚类、降维、Elbow method | ✅ |
| **7** | 周复习 | 六模型大乱斗，对比报告 | ✅ |

---

## 第2周 · 神经网络基础

| Day | 内容 | 核心问题 | 状态 |
|-----|------|---------|:----:|
| **8** | 多层感知器 (MLP) | 从逻辑回归到神经网络，激活函数结构 | ✅ |
| **9** | 前向 + 反向传播 | 链式法则，梯度穿过隐藏层 | ✅ |
| **10** | PyTorch 入门 | Tensor、Autograd、nn.Module | ✅ |
| **11** | LeNet-5 | 卷积核、池化、感受野 | ✅ |
| **12** | AlexNet / VGG | 网络加深的收益与代价 | ✅ |
| **13** | ResNet | 残差连接——为什么152层不会退化 | ✅ |
| **14** | 迁移学习 | 预训练 ResNet 微调自己的数据集 | ✅ |

---

## 第3周 · 序列与 Transformer

| Day | 内容 | 核心问题 | 状态 |
|-----|------|---------|:----:|
| **15** | RNN | 隐藏状态、BPTT、梯度消失 | ✅ |
| **16** | LSTM | 三个门：遗忘、输入、输出 | ✅ |
| **17** | GRU + Seq2Seq | LSTM 简化 + 编码器-解码器 | ✅ |
| **18** | Attention (Bahdanau) | 解码器不再只看最后一个隐藏状态 | ✅ |
| **19** | **Transformer** | Self-Attention、Multi-Head、位置编码 | ✅ |
| **20** | Transformer 实战 | HuggingFace 微调文本分类 | ✅ |
| **21** | 周复习 | RNN → LSTM → Transformer 演化对比 | ✅ |

---

## 第4周 · 预训练与生成

| Day | 内容 | 核心问题 | 状态 |
|-----|------|---------|:----:|
| **22** | BERT | MLM 预训练 + 双向编码 | ✅ |
| **23** | GPT-1/2 | 自回归 + 下一个词预测 | ✅ |
| **24** | nanoGPT | Karpathy，从头写一个 GPT | ✅ |
| **25** | ViT | Transformer 做图像分类 | ✅ |
| **26** | CLIP | 图文对比学习、多模态 | ✅ |
| **27** | GAN | 生成器 vs 判别器、DCGAN | ✅ |
| **28** | Stable Diffusion | 扩散模型、文生图原理 | ✅ |

---

## 第5周 · 实战与收尾

| Day | 内容 | 核心 | 状态 |
|-----|------|------|:----:|
| **29** | Kaggle 表格数据 | EDA、特征工程、模型集成 | ✅ |
| **30** | Kaggle NLP | TF-IDF、LSTM 文本分类 | ✅ |
| **31** | YOLO 目标检测 | 网格+锚框、NMS、预训练检测 | ✅ |
| **32** | ONNX 模型部署 | PyTorch → ONNX → 跨平台推理 | ✅ |
| **33** | 整理 GitHub + 写技术博客 | Profile README、仓库整理、博客写作 | ✅ |
| **34** | 方向选择 | CV / NLP / 多模态 / RL 对比 | ✅ |
| **35** | 最终总结 + 知识地图 | 全路线回顾、技能树、知识图谱 | ✅ |

---

## 文件夹结构

```
python深度学习/
├── 35天ML+DL学习大纲.md           ← 完整路线（全部 ✅）
├── 学习进度.md                   ← 进度跟踪（完结）
│
├── Week1/       Day 1-7  ✅ 经典 ML
│   ├── Day1/  ~  Day7/
│   └── ML_Week1_计划.md
├── Week2/       Day 8-14 ✅ 神经网络
├── Week3/       Day 15-21 ✅ 序列 + Transformer
├── Week4/       Day 22-28 ✅ 预训练 + 生成
└── Week5/       Day 29-35 ✅ 实战收尾
```

---

## 完成路线图

```
35-Day Journey
═══════════════════════════════════════

  NumPy ─→ 线性回归 ─→ 逻辑回归 ─→ 决策树 ─→ 集成学习 ─→ K-Means/PCA
    │
    ├──→ MLP ─→ 反向传播 ─→ PyTorch ─→ LeNet-5 ─→ AlexNet/VGG
    │      └──→ ResNet ─→ 迁移学习
    │
    ├──→ RNN ─→ LSTM ─→ GRU/Seq2Seq ─→ Attention
    │      └──→ Transformer ─→ 微调实战
    │
    ├──→ BERT ─→ GPT ─→ nanoGPT ─→ ViT ─→ CLIP ─→ GAN ─→ Diffusion
    │
    └──→ Kaggle ─→ YOLO ─→ ONNX ─→ GitHub+Blog ─→ 方向选择
           └──→ 最终总结 🎉
```

---

## 数字速查

| 你在找 | 对应的 |
|--------|--------|
| Day 1-7 | Week1 经典 ML ✅ |
| Day 8-14 | Week2 神经网络 ✅ |
| Day 15-21 | Week3 Transformer ✅ |
| Day 22-28 | Week4 BERT/GPT/Diffusion ✅ |
| Day 29-35 | Week5 实战收尾 ✅ |
| 总计 | **35 天 · 35 个 notebook · 全部完成 🎉** |

---

> **完结于 2026 年 7 月 20 日**
> 从 NumPy 的第一行代码，到 Stable Diffusion 的完整实现
