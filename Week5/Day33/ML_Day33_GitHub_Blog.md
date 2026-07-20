# Day 33 - 整理 GitHub + 写技术博客

> 目标: 把 35 天学习成果包装成作品, 让面试官一眼看到你的水平
>
> 代码写得再好, 没人看到 = 没写

---

## 为什么 GitHub + 博客很重要?

```
你写完了 35 天代码                             面试官视角
  ├── 本地硬盘躺着的 .ipynb          ──►    ❌ 看不到
  ├── 杂乱无章的仓库                 ──►    ❌ 不想看
  └── 没有博客记录学习过程            ──►    ❌ 不知道你多强

整理之后:
  ├── GitHub 主页清晰展示             ──►    ✅ "这人代码功底不错"
  ├── 每个项目有 README + 演示图      ──►    ✅ "文档写得清楚"
  └── 技术博客记录学习思考            ──►    ✅ "有总结能力，能沟通"
```

**现实:** 70% 的简历因为 GitHub 空白被直接刷掉
**加分:** 一份整洁的 GitHub + 2-3 篇技术博客 = 面试机会 + 50%

---

## 1. GitHub Profile README — 你的首页

> GitHub 上建一个跟你用户名同名的仓库, 它会显示在 profile 顶部

```python
# ============================================
# Profile README 模板
# ============================================

profile_template = '''
# 👋 Hi, I'm [你的名字]

> 数据科学 / 深度学习 学习者

## 🛠 技术栈
Python | PyTorch | Scikit-learn | Transformer | CV | NLP

## 📌 精选项目
- [35 天 ML+DL 完整路线](link) — 从 NumPy 到 Stable Diffusion
- [Kaggle 实战](link) — 房价预测 / NLP 分类
- [YOLO 目标检测](link) — 从零手写 NMS + 预训练模型

## 📊 GitHub Stats

![GitHub Stats](
https://github-readme-stats.vercel.app/api?username=你的用户名&show_icons=true
)

## 📝 最新博客
- [Transformer 为什么需要 Self-Attention?](link)
- [从 GAN 到 Stable Diffusion: 生成模型进化史](link)
'''

print('生成的 README 内容:')
print(profile_template)
print()
print('更多 stats 卡片: https://github.com/anuraghazra/github-readme-stats')
print('README 在线生成器: https://readme.so/')
```

```text
Profile README 效果:
  别人点进你 GitHub 主页 → 第一眼看到:
    - 你叫什么、会什么
    - 你的精选项目 (带链接)
    - 你的 GitHub 活跃度 (stats 卡片)
    - 你的博客文章

这就是你的"技术名片"
```

---

## 2. 仓库命名 — 好名字让项目被找到

> 坏的命名让人不想点开, 好的命名让人一眼看懂

```python
# ============================================
# 对比: 坏的 vs 好的仓库命名
# ============================================

bad_names = [
    'ceshi123',
    'testfinal_v2',
    'my_model',
    'untitled.ipynb',
    'pythonProject42',
]

good_names = [
    '35-days-ml-dl-roadmap',      # 35 天学习路线
    'pytorch-transformer-lab',    # Transformer 实现
    'kaggle-house-prices',        # Kaggle 实战
    'yolo-object-detection',      # YOLO 目标检测
    'bert-sentiment-classifier',  # BERT 情感分类
]

print('❌ 坏的命名:')
for n in bad_names:
    print(f'   {n}  ← 不知道是什么')

print()
print('✅ 好的命名:')
for n in good_names:
    print(f'   {n}')

print()
print('原则: 英文小写 + 连字符, 让人一看就知道项目内容')
```

```text
每个仓库必备 5 件套:
  1. README.md        ← 项目说明 (必须!)
  2. LICENSE          ← 开源协议 (推荐 MIT)
  3. requirements.txt ← 依赖清单
  4. demo/            ← 演示截图 / GIF
  5. .gitignore       ← 忽略无关文件
```

---

## 3. 写 README — 项目的"脸面"

> README 决定别人要不要点进去看

```python
# ============================================
# README 完整模板 (直接复制修改)
# ============================================

readme_template = '''
# 项目名称

<!-- 项目截图 / GIF — 放在顶部最吸睛 -->
![Demo](demo/screenshot.png)

## 📖 简介
用一句话说明这个项目是什么。
"从零实现 Transformer, 包含 Self-Attention、Multi-Head、位置编码,
  并在 IMDB 情感分类上达到 85%+ 准确率。"

## ✨ 特点
- 🧠 纯 NumPy 实现: 不依赖深度学习框架
- 📊 完整可视化: 注意力权重热力图、训练曲线
- 🚀 PyTorch 版: 可切换到 GPU 训练

## 🗂 项目结构
```
├── src/
│   ├── attention.py      # Self-Attention 实现
│   ├── transformer.py    # Transformer 完整模型
│   └── train.py          # 训练脚本
├── notebooks/
│   └── demo.ipynb        # 交互式演示
├── requirements.txt
└── README.md
```

## 🚀 快速开始
```bash
git clone https://github.com/你的用户名/项目名.git
cd 项目名
pip install -r requirements.txt
python src/train.py
```

## 📊 结果
| 模型 | 参数 | 准确率 |
|:----|:----:|:-----:|
| Transformer (NumPy) | 1.2M | 85.2% |
| Transformer (PyTorch) | 1.2M | 87.6% |
| BERT-base | 110M | 93.5% |

## 📚 参考
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📄 协议
MIT © 2024 [你的名字]
'''

print('README 模板就绪, 复制到你的仓库即可!')
```

```text
README 要点:
  - 第一屏放截图 + 一句话简介
  - 项目结构让人一目了然
  - 快速开始让人能马上跑起来
  - 结果表格展示成绩
```

---

## 4. 精选代表作 — 3 个 > 30 个

> 不需要把所有 day 都推成独立仓库, 精选最有代表性的

```python
# ============================================
# 从 35 天中选出 4 个代表作
# ============================================

projects = [
    {
        'name': 'Transformer 从零实现',
        'highlights': 'Self-Attention + Multi-Head + 位置编码手写',
        'showcases': '算法理解深度',
        'source': 'Day 19-20'
    },
    {
        'name': 'Kaggle 实战',
        'highlights': 'EDA + 特征工程 + 模型调参 + 提分曲线',
        'showcases': '实战能力',
        'source': 'Day 29-30'
    },
    {
        'name': 'YOLO 目标检测',
        'highlights': 'NMS 手写 + 可视化 + 预训练调用',
        'showcases': 'CV 能力',
        'source': 'Day 31'
    },
    {
        'name': 'Stable Diffusion 扩散模型',
        'highlights': '扩散过程可视化 + 去噪网络实现',
        'showcases': '生成模型理解',
        'source': 'Day 28'
    },
]

# 用表格格式展示
header = f'| {"项目":<25} | {"亮点":<40} | {"适合展示":<18} |'
sep = f'|{"":-<27}|{"":-<42}|{"":-<20}|'
print(header)
print(sep)
for p in projects:
    row = f'| {p["name"]:<25} | {p["highlights"]:<40} | {p["showcases"]:<18} |'
    print(row)
```

```text
每个代表作仓库需要:
  demo/
  ├── architecture.png      # 模型架构图
  ├── training_curve.png    # 训练曲线
  ├── results.png           # 最终结果
  └── demo.gif              # (可选) 运行演示
```

---

## 5. 写技术博客 — 为什么写?

> 教是最好的学 — 费曼学习法

```python
# ============================================
# 写博客的好处 & 平台选择
# ============================================

benefits = {
    '面试': '展示你的沟通能力 + 技术深度',
    '学习': '教是最好的学 (费曼学习法)',
    '记录': '方便以后自己回顾',
    '影响力': '积累粉丝/关注 → 机会找上门',
}

platforms = {
    '知乎': '长文技术深度, 推荐首发',
    '掘金': '排版漂亮, 技术氛围好',
    'CSDN': 'SEO 好, 搜索流量大',
    '公众号': '私域, 朋友圈分享',
    'Medium': '英文, 国外技术圈',
}

print('📝 写博客的好处:')
for k, v in benefits.items():
    print(f'  {k:<8} → {v}')

print()
print('🌐 平台选择 (推荐国内):')
for k, v in platforms.items():
    print(f'  {k:<8} → {v}')
```

---

## 6. 博客写什么? — 从 35 天里选

> 选 3-5 个最值得写的主题

```python
# ============================================
# 推荐博客主题 (面试高频 + 加分)
# ============================================

must_write = [
    '1. Transformer 详解 + 手写实现',
    '   Self-Attention / Multi-Head / 位置编码',
    '',
    '2. 从 ResNet 看残差连接为什么 work',
    '   退化问题 / 残差块 / 梯度流动可视化',
    '',
    '3. 从 RNN 到 LSTM 到 Transformer 的演化',
    '   梯度消失 / 长程依赖 / Attention 的优势',
]

bonus_topics = [
    '4. GAN vs 扩散模型: 生成模型的两种范式',
    '5. Kaggle 实战: 从 EDA 到提交的完整流程',
    '6. BERT vs GPT: 双向编码 vs 自回归解码',
]

print('🔴 必写 (面试高频):')
for line in must_write:
    print(f'  {line}')

print()
print('🟢 加分 (展示广度):')
for t in bonus_topics:
    print(f'  {t}')
```

---

## 7. 博客黄金结构 — 照着写

> 好的技术博客 = 1 个核心观点 + 3 张图 + 10 行关键代码 + 20 分钟阅读时间

```python
# ============================================
# 技术博客"六段式"模板
# ============================================

article_template = '''
# Transformer 为什么需要 Self-Attention? 从零手写实现

## 1. 问题: RNN 有什么问题?
- 顺序计算: 不能并行
- 梯度消失: 长程依赖难学
- 一句话: RNN 是串行的, 我们需要并行的

## 2. 核心思想: Self-Attention
- 每个词"看"所有词, 决定谁更重要
- 公式: Attention(Q,K,V) = softmax(QK^T/√d)V
- [插入注意力权重热力图]

## 3. 手写实现 (关键代码)
def self_attention(Q, K, V):
    scores = Q @ K.T / math.sqrt(K.shape[-1])
    weights = softmax(scores, dim=-1)
    return weights @ V

## 4. 完整架构
- Scaled Dot-Product → Multi-Head → Transformer
- [插入模型架构图]
- LayerNorm + 残差连接 + 位置编码

## 5. 实验结果
| 指标 | RNN | LSTM | Transformer |
| 训练时间 | 10min | 12min | 3min |
| 准确率 | 72% | 81% | 87% |

## 6. 总结
- Self-Attention 用并行解决了 RNN 的串行瓶颈
- Multi-Head 从不同角度理解句子
- Transformer = Self-Attention + 位置编码 + FFN

## 参考资料
- Attention Is All You Need (https://arxiv.org/abs/1706.03762)
'''

print('博客六段式: 问题 → 思想 → 代码 → 架构 → 实验 → 总结')
```

```text
标题技巧:
  数字:   "3 种 Attention 机制对比"
  问题:   "Transformer 为什么需要 Multi-Head?"
  对比:   "GAN vs 扩散模型: 谁更强?"
  实战:   "Kaggle 房价预测: 从 0.5 到 0.2 的提分之路"
```

---

## 8. GitHub Pages — 免费作品集

> 用 GitHub 自带的 Pages 功能, 零成本建个人网站

```python
# ============================================
# GitHub Pages 搭建方案对比
# ============================================

options = [
    ('GitHub Pages + Jekyll', '最简单', [
        '1. 建仓库: 用户名.github.io',
        '2. 选主题: https://pages.github.com/themes/',
        '3. 写内容: _posts/ 目录放博客',
        '4. 部署: push 自动部署'
    ]),
    ('用模板', '快速', [
        '推荐: https://github.com/sbthemes/modern-resume-theme',
        'fork 仓库 → 改配置 → 上线'
    ]),
    ('Next.js / Hugo / Astro', '进阶', [
        '自己搭建, 自由度最高',
        '部署到 Vercel / Netlify 免费'
    ]),
]

print('GitHub Pages 作品集方案:')
print()
for name, difficulty, steps in options:
    print(f'  【{name}】({difficulty})')
    for s in steps:
        print(f'    {s}')
    print()
```

---

## 9. GitHub 贡献图 — 绿点不断

> 保持提交节奏, 让 contribution graph 看起来活跃

```python
# ============================================
# 保持提交节奏的方法
# ============================================

tips = [
    '每天写一行代码 → commit → 绿点不断',
    '把学习笔记也用 git 管理, 每天提交',
    '参与开源项目 → 贡献 PR',
    '用 GitHub Actions 自动更新 README (但不推荐纯刷)',
]

print('保持 GitHub 活跃的方法:')
for t in tips:
    print(f'  ✅ {t}')

print()
print('推荐资源:')
print('  https://github.com/rzashakeri/beautify-github-profile')
print('  https://github.com/abhisheknaiidu/awesome-github-profile-readme')
```

```yaml
# .github/workflows/update-readme.yml
# GitHub Actions 自动更新 README (每周日更新)
name: Update README
on:
  schedule:
    - cron: '0 0 * * 0'
  workflow_dispatch:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate README
        run: python scripts/generate_readme.py
      - name: Commit changes
        run: |
          git config --global user.name 'bot'
          git config --global user.email 'bot@github.com'
          git add README.md
          git commit -m 'auto: update README' || exit 0
          git push
```

---

## 10. 把项目写进简历

> 简历上每个项目都要可验证 — 直接链接到 GitHub

```text
把 GitHub 项目写进简历:
  ├── 项目名 (超链接到 GitHub)
  ├── 一句话: "从零实现 Transformer, 包含完整训练 + 可视化"
  ├── 关键技术: PyTorch / NumPy / Matplotlib
  └── 结果数据: "理解 + 手写 + 论文阅读"

实际例子:

┌─────────────────────────────────────────────────┐
│ 项目: Transformer 从零实现                        │
│ 链接: github.com/xxx/transformer-lab              │
│ 描述: 从论文到手写实现 Self-Attention,            │
│       Multi-Head, 位置编码, 在 IMDB              │
│       情感分类上用 2 小时达到 85% 准确率            │
│ 技术: Python, PyTorch, NumPy, Matplotlib          │
└─────────────────────────────────────────────────┘
```

---

## Day 33 完成

### 今天学到了什么

| 概念 | 一句话 |
|:----|:------|
| **Profile README** | 同名仓库显示在主页顶部，是你的技术名片 |
| **仓库命名** | 英文小写 + 连字符，让人一眼看懂 |
| **README 五件套** | README + LICENSE + requirements.txt + demo/ + .gitignore |
| **代表作精选** | 精选 3-4 个高质量项目，好过 30 个杂乱项目 |
| **技术博客** | 六段式结构: 问题 → 思想 → 代码 → 架构 → 实验 → 总结 |
| **GitHub Pages** | 用户名.github.io 零成本建个人网站 |
| **贡献图** | 每天 commit, 保持绿点 |

## 35 天完成状态

```
Day 1-7:   经典 ML    ✅ 算法基础
Day 8-14:  神经网络    ✅ 深度学习入门
Day 15-21: 序列模型    ✅ Transformer
Day 22-28: 预训练生成  ✅ BERT/GPT/ViT/CLIP/GAN/Diffusion
Day 29-30: Kaggle     ✅ 实战
Day 31:    YOLO       ✅ 目标检测
Day 32:    ONNX       ✅ 模型部署
Day 33:    GitHub+博客 ✅ 今天 (作品包装)
Day 34:    方向选择    ← 明天 (CV/NLP/多模态/RL)
Day 35:    最终总结    ← 后天 (知识地图)
```

## 作业

---

### 作业 1: 创建你的 GitHub Profile README

**位置:** 第 1 节, `profile_template`

用模板创建同名仓库 README, 改成你自己的内容

<details>
<summary>📖 点击查看答案</summary>

```text
步骤:
1. 新建仓库: 用户名/用户名 (必须跟你的 GitHub 用户名完全一致)
2. 创建 README.md, 粘贴第 1 节的模板
3. 替换内容:
   - [你的名字] → 你的真实名字或昵称
   - 技术栈 → 你实际用过的技术
   - 项目链接 → 你的真实仓库地址
   - GitHub Stats 中的 username → 你的用户名
4. commit + push
5. 去 https://github.com/你的用户名 看效果

效果: 别人点进你主页, 第一眼就知道你是谁、会什么、做过什么
```

</details>

---

### 作业 2: 写一篇技术博客大纲

**位置:** 第 7 节, `article_template`

选一个你学得最好的主题, 按六段式结构写大纲

<details>
<summary>📖 点击查看答案</summary>

```text
推荐主题 (从 35 天里选):
- 第 3 周学得最好 → "Transformer 从零实现"
- 第 4 周学得最好 → "扩散模型原理 + 手写去噪"
- 实战最有感觉 → "Kaggle 从 EDA 到 Top 10%"

写作步骤:
1. 选 1 个主题 (一篇只讲一个概念)
2. 写六段式大纲 (问题 → 思想 → 代码 → 架构 → 实验 → 总结)
3. 准备配图 (模型架构图 / 实验结果 / 对比图)
4. 只放核心代码 (10-20 行, 不要全贴)
5. 写正文 (先写 80%, 再反复打磨)
6. 发布到知乎 / 掘金

衡量标准:
  一篇好的技术博客 = 1 个核心观点 + 3 张图 + 10 行关键代码 + 20 分钟阅读时间
```

</details>

---

### 作业 3: 给你 35 天学习仓库加一个 README

**位置:** `d:\hyy\Desktop\边城\python深度学习\`

参考第 3 节的模板, 给整个学习路线写一个总览 README

<details>
<summary>📖 点击查看答案</summary>

```markdown
# 🧠 35 天 ML + DL 完整学习路线

从 NumPy 到手写 Transformer, 从 BERT 到 Stable Diffusion

## 路线总览

| 周 | 内容 | 天数 |
|:--|:----|:---:|
| Week1 | 经典 ML (线性回归 → 集成学习) | Day 1-7 |
| Week2 | 神经网络 (MLP → ResNet → 迁移学习) | Day 8-14 |
| Week3 | 序列模型 (RNN → Attention → Transformer) | Day 15-21 |
| Week4 | 预训练生成 (BERT → GPT → ViT → GAN → 扩散模型) | Day 22-28 |
| Week5 | 实战收尾 (Kaggle → YOLO → ONNX → GitHub+博客) | Day 29-35 |

## 技术栈

Python · PyTorch · NumPy · Scikit-learn · HuggingFace · Matplotlib

## 精选项目

- [Transformer 从零实现](link) — Self-Attention / Multi-Head / 位置编码全手写
- [YOLO 目标检测](link) — NMS 手写 + 预训练模型
- [Kaggle 实战](link) — 房价预测 + NLP 分类

## 学习方式

- 每天一个 Jupyter Notebook, 代码 + 可视化 + 笔记
- 每个概念都有对比实验和图表
- 作业带答案, 方便自测

## 进度

完成 33/35 天, 最后 2 天收尾
```

</details>
