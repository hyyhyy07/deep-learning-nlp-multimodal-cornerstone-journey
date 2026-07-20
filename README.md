# 🧠 深度学习 · NLP 与多模态学习路线

> 从线性回归到 Stable Diffusion，35 天循序渐进的深度学习实践之旅。
> 聚焦 **自然语言处理** 与 **多模态学习**，用代码理解每一个模型。

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Days](https://img.shields.io/badge/35%20Days-Complete-2ea44f)](#-学习路线总览)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#)

---

## 📑 目录

- [为什么有这个项目](#-为什么有这个项目)
- [学习路线总览](#-学习路线总览)
- [路线图：从 ML 到多模态](#-路线图从-ml-到多模态)
  - [第一周 · 经典机器学习](#第一周--经典机器学习)
  - [第二周 · 神经网络与卷积](#第二周--神经网络与卷积)
  - [第三周 · 序列模型与 Transformer（NLP 核心）](#第三周--序列模型与-transformer-nlp-核心)
  - [第四周 · 预训练与多模态](#第四周--预训练与多模态)
  - [第五周 · 实战部署与方向选择](#第五周--实战部署与方向选择)
- [NLP 专题：从 RNN 到 GPT](#-nlp-专题从-rnn-到-gpt)
- [多模态专题：CLIP、ViT 与扩散模型](#-多模态专题clipvit-与扩散模型)
- [项目结构](#-项目结构)
- [如何开始](#-如何开始)
- [推荐学习资源](#-推荐学习资源)
- [关于作者](#-关于作者)

---

## 🎯 为什么有这个项目

这个仓库记录了从 **机器学习基础 → 深度学习 → NLP → 多模态** 的完整学习过程。

大多数教程的问题在于：**孤立讲模型，不讲演化脉络**。这里每一行代码都在回答一个问题——*为什么这个模型比上一个更好？*

如果你是以下人群之一，这个 repo 可能适合你：

- **转行者**：有 Python 基础，想系统进入 DL/NLP 方向
- **在校学生**：学完理论但缺少动手实现的经验
- **在职工程师**：想补全 CV 之外的多模态和 NLP 知识
- **面试准备者**：需要从 RNN 到 Transformer 到 CLIP 的完整知识链

每个知识点都遵循一个模式：
```
为什么需要它 → 核心思想 → 手写实现 → 可视化对比 → 与同类模型的差异
```

---

## 🗺️ 学习路线总览

| 周次 | 主题 | 天数 | 重点 |
|------|------|------|------|
| Week 1 | 经典机器学习 | Day 1–7 | NumPy, 回归, 决策树, 聚类 |
| Week 2 | 神经网络与 CNN | Day 8–14 | MLP, LeNet, AlexNet, ResNet |
| **Week 3** | **序列模型 & Transformer** | **Day 15–21** | **RNN, LSTM, Attention, Transformer ← NLP 核心** |
| **Week 4** | **预训练 & 多模态** | **Day 22–28** | **BERT, GPT, ViT, CLIP, Stable Diffusion ← 多模态核心** |
| Week 5 | 实战 & 部署 | Day 29–35 | Kaggle, YOLO, ONNX, 方向选择 |

> **NLP 主线**：Day 15 (RNN) → Day 16 (LSTM) → Day 17 (GRU+Seq2Seq) → Day 18 (Attention) → Day 19 (Transformer) → Day 22 (BERT) → Day 23 (GPT) → Day 24 (nanoGPT)
>
> **多模态主线**：Day 25 (ViT) → Day 26 (CLIP) → Day 27 (GAN) → Day 28 (Diffusion)

---

## 🧭 路线图：从 ML 到多模态

### 第一周 · 经典机器学习

打好数据科学校基础。用 NumPy 手写梯度下降，理解模型到底在优化什么。

| Day | 主题 | 核心内容 |
|-----|------|---------|
| 1 | 科学计算三板斧 | NumPy 矩阵运算 + Pandas 数据处理 + Matplotlib 可视化 |
| 2 | 线性回归 | 手写梯度下降、损失曲面可视化、学习率对比 |
| 3 | 逻辑回归 | Sigmoid 函数、交叉熵损失、决策边界 |
| 4 | 决策树 | 基尼系数、信息增益、过拟合与剪枝 |
| 5 | 集成学习 | 随机森林 vs XGBoost，偏差-方差权衡 |
| 6 | 无监督学习 | K-Means 聚类 + PCA 降维，从标准化到降维可视化 |
| 7 | 周复习 | 周知识图谱 + 综合练习 |

### 第二周 · 神经网络与卷积

从神经元堆叠到残差网络 —— 理解深度如何改变模型能力。

| Day | 主题 | 核心内容 |
|-----|------|---------|
| 8 | 多层感知机 | 激活函数对比（ReLU / Sigmoid / Tanh）、无激活 = 线性 |
| 9 | 反向传播 | 链式法则手工推导、梯度流动可视化 |
| 10 | PyTorch 入门 | Tensor + Autograd + 模块化项目结构 |
| 11 | LeNet-5 | 卷积/池化/感受野，手写数字识别的经典 |
| 12 | AlexNet / VGG | 网络加深的代价——**退化问题** 的发现 |
| 13 | ResNet | **残差连接** 解决退化，PlainNet vs ResNet 对比实验 |
| 14 | 迁移学习 | 预训练 ResNet 微调、冻结/解冻策略 |

### 第三周 · 序列模型与 Transformer（NLP 核心）

**这是 NLP 学习的核心章节。** 从处理序列数据开始，一路走到 Transformer 架构。

| Day | 主题 | 核心内容 |
|-----|------|---------|
| 15 | **RNN** | 隐藏状态、BPTT、梯度消失、正弦波预测 + 字符级文本生成 |
| 16 | **LSTM** | 遗忘门/输入门/输出门、细胞状态、RNN vs LSTM 对比实验 |
| 17 | **GRU + Seq2Seq** | 更新门/重置门、RNN/LSTM/GRU 三兄弟对决、Seq2Seq 反转字符串 |
| 18 | **Attention 机制** | Bahdanau 加性注意力、信息瓶颈可视化、三种 Attention 对比 |
| 19 | **Transformer** | Self-Attention、Multi-Head、位置编码、完整 Encoder 从零实现 |
| 20 | Transformer 实战 | HuggingFace DistilBERT 微调、情感分类全流程 |
| 21 | 周复习 | 六种序列模型同台对决 + NLP 知识图谱 |

关键对比实验：

```
RNN  ←  梯度消失严重，长序列记忆差
LSTM ←  门控机制缓解梯度问题，长程记忆显著提升
GRU  ←  简化版 LSTM，参数更少，效果相当
Seq2Seq + Attention ←  解决信息瓶颈，对齐输入输出
Transformer          ←  彻底去掉循环，并行化 + 全局注意力
```

### 第四周 · 预训练与多模态

从纯文本到图像再到图文联合 —— 多模态学习的核心组件。

| Day | 主题 | 核心内容 |
|-----|------|---------|
| 22 | **BERT** | MLM 掩码语言模型、双向注意力、句子表示 |
| 23 | **GPT** | 自回归语言模型、下一个词预测、zero-shot 能力 |
| 24 | **nanoGPT** | 从零手写 GPT 训练 loop，理解 LLM 的底层机制 |
| 25 | **ViT** | Transformer 做图像分类、Patch Embedding、位置编码 |
| 26 | **CLIP** | 图文对比学习、双塔结构、零样本分类 |
| 27 | GAN | 生成器 vs 判别器对抗训练、DCGAN |
| 28 | Stable Diffusion | 扩散模型原理、前向加噪 → 逆向去噪、文生图流程 |

多模态模型之间的关系：

```
ViT  ←  把 Transformer 从文本搬到图像（图像编码器）
CLIP ←  把文本 + 图像拉到同一个语义空间（双塔对齐）
GAN  ←  生成逼真图像（对抗训练）
Diffusion ←  可控图像生成（逐步去噪，SOTA 范式）
```

### 第五周 · 实战部署与方向选择

把学到的知识落地到真实场景。

| Day | 主题 | 核心内容 |
|-----|------|---------|
| 29 | Kaggle 表格数据 | EDA、特征工程、模型集成 |
| 30 | Kaggle NLP | TF-IDF + LSTM 文本分类比赛 |
| 31 | YOLO 目标检测 | 锚框、网格、NMS 非极大值抑制 |
| 32 | ONNX 部署 | PyTorch → ONNX → Runtime 推理加速 |
| 33 | 技术写作 | GitHub Profile README + 仓库整理 |
| 34 | 方向选择 | CV / NLP / 多模态 / RL 横向对比 |
| 35 | 最终总结 | 知识地图、技能树、持续学习路线 |

---

## 🔤 NLP 专题：从 RNN 到 GPT

本仓库的 NLP 主线覆盖了现代 NLP 的完整演化链：

```
统计方法          ←  TF-IDF (Day 30)
序列模型          ←  RNN → LSTM → GRU (Day 15-17)
对齐机制          ←  Attention (Day 18)
并行架构          ←  Transformer (Day 19)
预训练编码器      ←  BERT (Day 22)
自回归生成        ←  GPT / nanoGPT (Day 23-24)
```

每个关键节点都配有 **可运行的 Jupyter Notebook**，包含：

- 模型结构的手工实现（非仅仅调包）
- 训练过程的可视化（loss 曲线、注意力权重热力图）
- 与前一模型的定量对比（相同任务下的效果差异）
- 调试版 notebook（带探索性修改和错误尝试记录）

---

## 🖼️ 多模态专题：CLIP、ViT 与扩散模型

多模态是 2024–2026 年 AI 最活跃的方向之一。本仓库涵盖：

| 能力 | 模型 | Notebook |
|------|------|----------|
| 图像理解 | ViT (Vision Transformer) | `Week4/Day25/ML_Day25_ViT.ipynb` |
| 图文对齐 | CLIP (Contrastive Language-Image Pre-training) | `Week4/Day26/ML_Day26_CLIP.ipynb` |
| 图像生成 | DCGAN | `Week4/Day27/ML_Day27_GAN.ipynb` |
| 可控生成 | Stable Diffusion | `Week4/Day28/ML_Day28_Diffusion.ipynb` |

**CLIP 是理解现代多模态架构的关键** —— 它首次证明了文本和图像可以在同一个语义空间中对齐，后续的 DALL·E、Flamingo、GPT-4V 等多模态大模型都建立在类似的思想之上。

---

## 📂 项目结构

```
python-deep-learning-study-schedule/
├── 35天ML+DL学习大纲.md          # 完整学习计划
├── 学习进度.md                   # 学习进度追踪
├── .gitignore
├── README.md
│
├── Week1/  (经典机器学习)
│   ├── Day1/  ~  Day7/
│   └── ...
│
├── Week2/  (神经网络与CNN)
│   ├── Day8/  ~  Day14/
│   └── ...
│
├── Week3/  (序列模型 & Transformer)    ← NLP 核心
│   ├── Day15/  RNN                        # 含 RNN + debug 双版本
│   ├── Day16/  LSTM
│   ├── Day17/  GRU & Seq2Seq
│   ├── Day18/  Attention
│   ├── Day19/  Transformer
│   ├── Day20/  Transformer 实战 (HuggingFace)
│   └── Day21/  周复习
│
├── Week4/  (预训练 & 多模态)              ← 多模态核心
│   ├── Day22/  BERT
│   ├── Day23/  GPT
│   ├── Day24/  nanoGPT (从零实现)
│   ├── Day25/  ViT
│   ├── Day26/  CLIP
│   ├── Day27/  GAN
│   └── Day28/  Stable Diffusion
│
└── Week5/  (实战 & 部署)
    ├── Day29/  Kaggle 表格数据
    ├── Day30/  Kaggle NLP
    ├── Day31/  YOLO 目标检测
    ├── Day32/  ONNX 模型部署
    ├── Day33/  GitHub 博客
    ├── Day34/  方向选择
    └── Day35/  最终总结
```

每个 Day 目录的结构：
```
DayXX/
├── ML_DayXX_Topic.ipynb          # 正式版 notebook
├── ML_DayXX_Topic_debug.ipynb    # 调试探索版（含试错记录）
├── data/                         # (gitignored) 下载的数据集
└── *.png                         # 训练可视化截图
```

---

## 🚀 如何开始

### 前置要求

- Python 3.10+
- Jupyter Notebook / Jupyter Lab
- PyTorch 2.0+

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/hyyhyy07/deep-learning-nlp-multimodal-cornerstone-journey.git
cd python-deep-learning-study-schedule

# 安装核心依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn
pip install transformers datasets  # Week 3-4 需要
```

然后从任意 Day 开始：

```bash
jupyter notebook Week3/Day15/ML_Day15_RNN.ipynb
```

**推荐学习顺序**：按 Day 编号顺序阅读，NLP 方向可重点看 Week 3 + Week 4；多模态方向可从 Day 25 开始。

---

## 📚 推荐学习资源

### 顶尖大学课程

| 资源 | 适用阶段 | 备注 |
|------|---------|------|
| [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) | Week 2-3 | Stanford 经典 CV 课程，含卷积网络与视觉理解全部内容 |
| [CS224n: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) | Week 3-4 | Stanford NLP 课程，与本路线高度互补 |
| [CS229: Machine Learning](https://cs229.stanford.edu/) | Week 1-2 | Stanford ML 基础课，Andrew Ng 经典讲义 + 习题 |
| [CS 182: Deep Learning (Berkeley)](https://cs182.github.io/) | Week 2-4 | Berkeley 深度学习课程，理论与实践并重 |
| [MIT 6.S191: Intro to Deep Learning](https://introtodeeplearning.com/) | Week 1-3 | MIT 深度学习入门，覆盖 RNN/Transformer/生成模型 |
| [MIT 6.S094: Deep Learning for Self-Driving Cars](https://selfdrivingcars.mit.edu/) | Week 4-5 | 将 DL 应用于自动驾驶的实战课程 |
| [CMU 11-785: Intro to Deep Learning](https://deeplearning.cs.cmu.edu/) | Week 2-4 | CMU 深度学习课程，数学推导深入 |
| [Stanford CS330: Multi-Task & Meta Learning](https://cs330.stanford.edu/) | Week 4-5 | 多任务学习与元学习，进阶话题 |
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/) | Week 5 | 涵盖 MLOps、部署、实验管理等工程实践 |

### 视频与 YouTube 系列

| 资源 | 适用阶段 | 备注 |
|------|---------|------|
| [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Week 1-2 | 直觉最强的数学可视化系列 |
| [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | Week 2-4 | 从反向传播手写到 nanoGPT，本仓库的灵感来源之一 |
| [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) | Week 1-3 | 统计学和 ML 概念讲得最清晰的频道 |
| [Machine Learning with Phil](https://www.youtube.com/@MachineLearningwithPhil) | Week 2-4 | 代码实现类教程，从线性回归到 GAN 全面覆盖 |
| [Aladdin Persson](https://www.youtube.com/@AladdinPersson) | Week 3-5 | PyTorch 实现各类模型的高质量教程 |
| [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) | Week 3-5 | 论文精读频道，涵盖 NLP 和多模态 SOTA 论文 |
| [The AI Epiphany](https://www.youtube.com/@theaiepiphany) | Week 3-5 | 深度视觉、多模态和生成模型的深度解读 |
| [DeepLearning.AI — YouTube](https://www.youtube.com/@Deeplearningai) | Week 1-5 | Andrew Ng 团队的频道，覆盖从入门到前沿 |
| [Arxiv Insights](https://www.youtube.com/@ArxivInsights) | Week 3-4 | 论文动画讲解，注意力机制和 Transformer 动画令人印象深刻 |
| [Samuel Albanie](https://www.youtube.com/@SamuelAlbanie) | Week 3-5 | 计算机视觉和多模态研究的深度讲座 |

### 必读论文与博客

| 资源 | 适用阶段 | 备注 |
|------|---------|------|
| [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) | Day 19 | Transformer 逐行实现（PyTorch） |
| [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | Day 19 | Transformer 原始论文，NLP 的分水岭 |
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | Day 22 | BERT 论文，双向预训练的里程碑 |
| [Improving Language Understanding by Generative Pre-Training (GPT)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Day 23 | GPT 原始论文 |
| [An Image is Worth 16x16 Words: Transformers for Image Recognition (ViT)](https://arxiv.org/abs/2010.11929) | Day 25 | ViT 论文，CV 领域的 Transformer 革命 |
| [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | Day 26 | 多模态对比学习的开山之作 |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | Day 28 | DDPM 原始论文，扩散模型的基石 |
| [High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)](https://arxiv.org/abs/2112.10752) | Day 28 | Stable Diffusion 论文 |
| [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) | Day 20 | 知识蒸馏的经典案例 |
| [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) | Day 26 | 同上，多模态必读 |
| [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) | Week 5 | 神经网络剪枝的重要理论 |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | Day 13 | ResNet 论文，残差连接的起源 |

### 在线课程与交互式教程

| 资源 | 适用阶段 | 备注 |
|------|---------|------|
| [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course) | Day 20-22 | BERT/GPT 实战微调，使用 Transformers 库 |
| [HuggingFace Audio Course](https://huggingface.co/learn/audio-course) | Week 4-5 | 音频与语音领域的深度学习 |
| [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course) | Week 5 | 深度强化学习入门 |
| [fast.ai Practical Deep Learning](https://course.fast.ai/) | Week 1-3 | 自顶向下的实践派深度学习课程，代码优先 |
| [fast.ai From Deep Learning Foundations to Stable Diffusion](https://course.fast.ai/Lessons/lesson.html) | Week 4-5 | fast.ai 最新版，涵盖扩散模型 |
| [D2L: Dive into Deep Learning](https://d2l.ai/) | Week 1-5 | Amazon 出品，交互式教材 + 代码（MXNet/PyTorch/TensorFlow） |
| [Neural Networks from Scratch](https://nnfs.io/) | Week 1-3 | 从零构建神经网络，不依赖框架 |
| [Google — Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) | Week 1-2 | Google 官方 ML 速成课程 |

### 交互式可视化与工具

| 资源 | 适用阶段 | 备注 |
|------|---------|------|
| [LLM Visualization](https://bbycroft.net/llm) | Day 23-24 | GPT 内部机制 3D 可视化，交互式，视觉震撼 |
| [TensorFlow Playground](https://playground.tensorflow.org/) | Day 8-9 | 在线神经网络沙盒，直观理解激活函数、层数的影响 |
| [Distill.pub](https://distill.pub/) | Week 2-5 | 高水准的交互式 ML 科普期刊，每篇都是艺术品 |
| [CNN Explainer](https://poloclub.github.io/cnn-explainer/) | Day 11-12 | 交互式 CNN 可视化，中文友好 |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Day 19 | Transformer 内部交互可视化 |
| [Embedding Projector](https://projector.tensorflow.org/) | Day 22-23 | 高维嵌入向量的 3D 可视化 |

### 经典书籍（免费在线版）

| 资源 | 适用阶段 | 备注 |
|------|---------|------|
| [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) | Week 2-4 | "花书"，深度学习圣经，数学推导完备 |
| [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/) | Week 3-4 | NLP 领域最权威的教科书，持续更新中 |
| [Understanding Deep Learning (S. Prince)](https://udlbook.github.io/udlbook/) | Week 2-5 | 2023 年新书，图解精美，涵盖 Transformer 和多模态 |
| [The Elements of Statistical Learning (Hastie et al.)](https://hastie.su.domains/ElemStatLearn/) | Week 1-2 | 统计学习经典，偏理论但极其深刻 |
| [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html) | Week 5 | RL 领域的标准教材 |
| [Mathematics for Machine Learning (Deisenroth et al.)](https://mml-book.github.io/) | Week 1-2 | ML 需要的线性代数、概率论和微积分 |
| [Natural Language Processing with Transformers (Tunstall et al.)](https://transformersbook.com/) | Day 20-24 | HuggingFace 官方出品的 Transformer 实战书 |
| [Probabilistic Machine Learning (Murphy)](https://probml.github.io/pml-book/) | Week 1-4 | 2022 年巨著，覆盖所有现代 ML 方法 |

### 综合平台与社区

| 资源 | 适用阶段 | 备注 |
|------|---------|------|
| [Papers With Code](https://paperswithcode.com/) | Week 2-5 | 论文 + 代码 + Benchmark 一站式追踪 SOTA |
| [The Gradient](https://thegradient.pub/) | Week 3-5 | 高质量深度学习博客杂志 |
| [Sebastian Ruder's Blog](https://ruder.io/) | Week 3-5 | NLP 和多任务学习的前沿分析 |
| [Lilian Weng's Blog (OpenAI)](https://lilianweng.github.io/) | Week 3-5 | 系统性的技术博客，涵盖 LLM、多模态、扩散模型 |
| [Jay Alammar's Blog](https://jalammar.github.io/) | Week 3-4 | 最好的技术图解博客之一，BERT/Transformer 图解不可错过 |
| [Chris Olah's Blog](https://colah.github.io/) | Week 2-5 | Distill 创始人，RNN 和注意力机制的可视化先驱 |
| [GitHub Daily — Deep Learning](https://github.com/topics/deep-learning) | Week 1-5 | GitHub 每日热门 DL 项目 |
| [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) | Week 1-5 | Reddit ML 社区，论文讨论和职业建议 |
| [Hacker News — AI Section](https://news.ycombinator.com/) | Week 1-5 | 科技圈的 AI 热点讨论 |

---

## ✍️ 关于作者

这个项目是个人学习的完整记录。每个 notebook 都是边学边写的结果——

**原则**：
- **能手写就不调包** —— 只有手写一遍才能真正理解
- **能对比就不单测** —— 每个模型都有对照实验，好与坏都要知道为什么
- **能可视化就不空谈** —— loss 曲线、权重热力图、决策边界，看得见才算数

如果你也在学习深度学习的路上，欢迎 Star、Fork 或提 Issue 讨论。

---

<p align="center">
  <a href="#-深度学习--nlp-与多模态学习路线">⬆ 回到顶部</a>
</p>
