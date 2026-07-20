# Day 34 - 方向选择: CV / NLP / 多模态 / RL

> 目标: 了解深度学习四大方向, 找到最适合你的那一条路
>
> 35 天学完了基础, 现在是时候选一个方向深入了

---

## 为什么需要选方向?

```
35 天之前:                          35 天之后:
  你不知道 AI 能做什么               你已经知道所有方向的基本概念
  所有方向都陌生                     但每个方向都只学了皮毛

选方向的意义:
  不是"选错了就完蛋"                而是"先集中精力突破一个"
  广度已经有了 (35 天)              深度是下一步 (方向选择)

类比学乐器:
  35 天 = 你学会了识谱、按和弦、打拍子
  Day 34 = 你决定先练吉他, 还是先练钢琴
  不是另一个不能学, 是集中突破更有效
```

---

## 1. 全景: 四大方向

```python
# ============================================
# 四大方向总览
# ============================================

directions = {
    'CV': {
        'full_name': 'Computer Vision / 计算机视觉',
        'core_question': '让机器看懂图像和视频',
        'timeline': '2012 AlexNet → 2015 ResNet → 2020 ViT → 至今',
        'market': '自动驾驶 / 安防 / 医疗影像 / ARVR',
    },
    'NLP': {
        'full_name': 'Natural Language Processing / 自然语言处理',
        'core_question': '让机器理解和生成语言',
        'timeline': '2018 BERT → 2020 GPT-3 → 2023 GPT-4 → 至今',
        'market': '搜索 / 对话系统 / 翻译 / 内容生成',
    },
    'Multimodal': {
        'full_name': '多模态学习',
        'core_question': '让机器同时理解文本+图像+语音',
        'timeline': '2021 CLIP → 2023 GPT-4V → 2025 原生多模态',
        'market': '文生图 / 视频理解 / 机器人 / 搜索',
    },
    'RL': {
        'full_name': 'Reinforcement Learning / 强化学习',
        'core_question': '让机器通过试错学会做决策',
        'timeline': '2013 DQN → 2017 AlphaGo → 2023 RLHF',
        'market': '游戏 / 机器人控制 / 推荐系统 / 自动化',
    },
}

print(f'{"方向":<15} {"核心问题":<30} {"时间线":<40} {"应用场景":<30}')
print('-' * 115)
for k, v in directions.items():
    print(f'{k:<15} {v["core_question"]:<30} {v["timeline"]:<40} {v["market"]:<30}')
```

---

## 2. CV 方向 — 计算机视觉

> 让机器"看懂"世界: 图像分类 → 目标检测 → 分割 → 3D 视觉

```python
# ============================================
# CV 方向详解
# ============================================

cv_info = {
    '核心技能': [
        'CNN 架构: ResNet / EfficientNet / ConvNeXt',
        '目标检测: YOLO / DETR / Faster R-CNN',
        '图像分割: UNet / Mask R-CNN / SAM',
        '3D 视觉: 点云 / NeRF / 3DGS',
        '视频理解: 行为识别 / 目标跟踪',
    ],
    '推荐项目': [
        '用 YOLO + 自定义数据集做目标检测',
        '用 SAM 做分割一切 demo',
        '图像分类竞赛 (Kaggle 猫狗大战)',
        '自制人脸检测 + 马赛克工具',
    ],
    '就业方向': [
        '自动驾驶感知算法工程师',
        '安防 + 智慧城市视觉算法',
        '医疗影像 AI 工程师',
        'ARVR 计算机视觉工程师',
    ],
}

print('🔴 CV 方向')
print(f'{"核心技能":<20} {"推荐项目":<40} {"就业方向":<30}')
print('-' * 90)
skills = cv_info['核心技能']
projects = cv_info['推荐项目']
jobs = cv_info['就业方向']
for i in range(max(len(skills), len(projects), len(jobs))):
    s = skills[i] if i < len(skills) else ''
    p = projects[i] if i < len(projects) else ''
    j = jobs[i] if i < len(jobs) else ''
    print(f'{s:<20} {p:<40} {j:<30}')
```

```text
CV 面试高频题:
  - 1×1 卷积的作用? (降维 / 通道信息融合)
  - BatchNorm 的原理? (训练 vs 推理有什么区别)
  - 目标检测的 two-stage vs one-stage?
  - FPN (特征金字塔) 解决了什么问题?

CV 的独特优势:
  - 效果直观: 画个框 / 标个分割, 老板看得懂
  - 数据好找: ImageNet / COCO / OpenImages
  - 竞赛多: Kaggle CV 竞赛常年活跃
```

---

## 3. NLP 方向 — 自然语言处理

> 让机器"读懂"语言: 分类 → 生成 → 对话 → 推理

```python
# ============================================
# NLP 方向详解
# ============================================

nlp_info = {
    '核心技能': [
        'Transformer 架构: Encoder / Decoder / Seq2Seq',
        '预训练模型: BERT / GPT / LLaMA / Qwen',
        'NLP 任务: 分类 / 序列标注 / 文本生成 / 问答',
        '推理优化: KV-Cache / 量化 / Speculative Decoding',
        'Agent / RAG: 工具调用 / 知识检索增强',
    ],
    '推荐项目': [
        '用 BERT 做情感分析 / 命名实体识别',
        '用 LLM 搭建一个 RAG 问答系统',
        '微调一个小模型 (LLaMA / Qwen) 做特定任务',
        '做一个 AI 对话机器人 (Streamlit 部署)',
    ],
    '就业方向': [
        'NLP 算法工程师 (搜索 / 推荐)',
        '大模型应用开发工程师 (LLMOps)',
        '对话系统 / 智能客服',
        'AI 内容生成 (AIGC)',
    ],
}

print('🟡 NLP 方向')
print(f'{"核心技能":<40} {"推荐项目":<40} {"就业方向":<30}')
print('-' * 110)
skills = nlp_info['核心技能']
projects = nlp_info['推荐项目']
jobs = nlp_info['就业方向']
for i in range(max(len(skills), len(projects), len(jobs))):
    s = skills[i] if i < len(skills) else ''
    p = projects[i] if i < len(projects) else ''
    j = jobs[i] if i < len(jobs) else ''
    print(f'{s:<40} {p:<40} {j:<30}')
```

```text
NLP 面试高频题:
  - Self-Attention 的时间复杂度? (O(n²d))
  - 为什么需要位置编码? (无位置信息 = 词袋模型)
  - LayerNorm vs BatchNorm 为什么 NLP 用 LN?
  - 大模型的幻觉问题怎么缓解? (RAG / prompt 工程)

NLP 的独特优势:
  - 应用最广: 几乎每个产品都有文本处理需求
  - 门槛相对低: 不需要 GPU 也能跑小模型
  - 大模型时代最火的方向, 岗位最多
```

---

## 4. 多模态方向 — 融合多种信息

> 让机器同时理解文本 + 图像 + 语音 + 视频

```python
# ============================================
# 多模态方向详解
# ============================================

mm_info = {
    '核心技能': [
        'CLIP: 图文对比学习双编码器',
        'Cross-Attention: 不同模态的信息融合',
        'Diffusion Model: 文生图 / 图生视频',
        '多模态大模型: GPT-4V / Gemini / Qwen-VL',
        '模态对齐: 如何让图像和文本共享语义空间',
    ],
    '推荐项目': [
        '用 CLIP 做图文检索 (text → image search)',
        '用 Stable Diffusion 做可控文生图',
        '做一个图片问答系统 (VQA)',
        '用 Whisper 做语音识别 + 翻译',
    ],
    '就业方向': [
        '多模态算法工程师 (搜索 / 推荐)',
        'AIGC 算法工程师 (文生图 / 视频生成)',
        '自动驾驶多模态感知',
        'AI 内容审核 (图像 + 文本联合)',
    ],
}

print('🟢 多模态方向')
print(f'{"核心技能":<40} {"推荐项目":<40} {"就业方向":<30}')
print('-' * 110)
skills = mm_info['核心技能']
projects = mm_info['推荐项目']
jobs = mm_info['就业方向']
for i in range(max(len(skills), len(projects), len(jobs))):
    s = skills[i] if i < len(skills) else ''
    p = projects[i] if i < len(projects) else ''
    j = jobs[i] if i < len(jobs) else ''
    print(f'{s:<40} {p:<40} {j:<30}')
```

```text
多模态面试高频题:
  - CLIP 的对比学习损失函数? (InfoNCE Loss)
  - 图文对齐为什么需要 hard negative mining?
  - 扩散模型的引导方式: classifier-free vs classifier-based?
  - 多模态融合的早期融合 vs 晚期融合?

多模态的独特优势:
  - 最前沿: 2025-2026 年最火的方向
  - 天花板高: 同时要懂 CV + NLP
  - 大厂刚需: 搜索 / 视频 / 自动驾驶都需要多模态
```

---

## 5. RL 方向 — 强化学习

> 让机器通过试错学会做决策: 游戏 → 机器人 → RLHF

```python
# ============================================
# RL 方向详解
# ============================================

rl_info = {
    '核心技能': [
        'MDP: 马尔可夫决策过程形式化',
        'Q-Learning / DQN: 基于价值的方法',
        'Policy Gradient / PPO: 基于策略的方法',
        '环境交互: Gymnasium / MuJoCo 模拟器',
        'RLHF: 人类反馈强化学习 (大模型训练关键)',
    ],
    '推荐项目': [
        '用 DQN 玩 Atari 游戏 (CartPole / Breakout)',
        '用 PPO 训练一个神经网络学会走路',
        '实现一个简单的推荐系统 bandit 算法',
        '理解 RLHF 的 reward model 怎么训练',
    ],
    '就业方向': [
        '游戏 AI 算法工程师',
        '机器人控制算法工程师',
        '推荐系统强化学习 (流量调控 / 拍卖)',
        '大模型 RLHF 算法工程师 (最稀缺)',
    ],
}

print('🔵 RL 方向')
print(f'{"核心技能":<40} {"推荐项目":<40} {"就业方向":<30}')
print('-' * 110)
skills = rl_info['核心技能']
projects = rl_info['推荐项目']
jobs = rl_info['就业方向']
for i in range(max(len(skills), len(projects), len(jobs))):
    s = skills[i] if i < len(skills) else ''
    p = projects[i] if i < len(projects) else ''
    j = jobs[i] if i < len(jobs) else ''
    print(f'{s:<40} {p:<40} {j:<30}')
```

```text
RL 面试高频题:
  - Exploration vs Exploitation 的权衡?
  - On-policy vs Off-policy 的区别? (PPO vs DQN)
  - Reward shaping 的重要性?
  - RLHF 的 3 阶段: SFT → Reward Model → PPO?

RL 的独特优势:
  - 独一无二: CV/NLP 都能用 DL 做, RL 是独立范式
  - RLHF 是目前大模型对齐的关键技术
  - 做的人相对少, 竞争没那么激烈
```

---

## 6. 横向对比 — 四选一

> 从多个维度对比四大方向

```python
# ============================================
# 四大方向对比矩阵
# ============================================

import numpy as np

# 评分: 1-5 (越高越好)
comparison = {
    '方向': ['CV', 'NLP', '多模态', 'RL'],
    '入门难度':    [3, 2, 4, 5],   # 越低越容易
    '岗位数量':    [4, 5, 3, 2],   # 越高越多
    '薪资水平':    [3, 4, 5, 4],   # 预估
    '发展潜力':    [3, 4, 5, 4],   # 未来 3 年
    '数学要求':    [3, 2, 3, 5],   # 数学门槛
    'GPU 需求':    [4, 2, 4, 3],   # 硬件门槛
    '乐趣程度':    [4, 3, 5, 4],   # 主观
}

print(f'{"方向":<10} {"入门难度":<10} {"岗位数量":<10} {"薪资水平":<10} ",
      f'{"发展潜力":<10} {"数学要求":<10} {"GPU需求":<10} {"乐趣程度":<10}')
print('-' * 80)
for i in range(len(comparison['方向'])):
    row = [comparison[k][i] for k in comparison.keys() if k != '方向']
    stars = '  '.join(['★' * v + '☆' * (5-v) for v in row])
    print(f'{comparison["方向"][i]:<10} {stars}')

print()
print('解读:')
print('  NLP: 入门最容易, 岗位最多, 大模型时代首选')
print('  多模态: 发展潜力最大, 薪资最高, 但深度学习要求')
print('  CV: 入门适中, 岗位稳定, 适合喜欢视觉效果')
print('  RL: 门槛最高, 岗位少但竞争小, 适合有数学功底')
```

```text
一句话总结每个方向:
  CV        = "让机器看见"
  NLP       = "让机器读懂"
  多模态    = "让机器同时看见 + 读懂"
  RL        = "让机器学会做决定"
```

---

## 7. 怎么选? — 自我测评

> 答案不在外部, 在你自己的兴趣 + 背景 + 目标

```python
# ============================================
# 方向选择自测题
# ============================================

questions = [
    ('你更喜欢哪个?', [
        'A. 看结果 — 画框 / 分割图, 视觉上直观',
        'B. 玩语言 — 写文章 / 对话 / 翻译',
        'C. 都感兴趣 — 不想只做一个模态',
        'D. 爱策略 — 游戏 / 决策 / 优化问题',
    ]),
    ('你数学基础?', [
        'A. 够用 — 矩阵 / 求导没问题',
        'B. 很强 — 概率统计 / 凸优化都学过',
        'C. 一般 — 能看懂公式但不太深',
    ]),
    ('你想做什么样的工作?', [
        'A. 业务落地 — 做产品 / 工程化',
        'B. 研究探索 — 发论文 / 搞新技术',
        'C. 都想 — 先落地再深挖',
    ]),
]

for q, options in questions:
    print(q)
    for opt in options:
        print(f'  {opt}')
    print()
```

```text
结果参考:

选 A 多 → CV:        你喜欢视觉反馈, 想做看得见效果的事
选 B 多 → NLP:       你对话言敏感, 大模型时代 NLP 是最宽的路
选 C 多 → 多模态:    你好奇不同模态的交互, 适合前沿研究
选 D 多 → RL:        你喜欢策略和博弈, 适合游戏 / 机器人 / 量化

实用建议:
  1. 如果你还没毕业 → 选 NLP 或 多模态 (岗位最多)
  2. 如果你在找工作 → NLP > CV > 多模态 > RL (按岗位数量)
  3. 如果你在搞研究 → 多模态 > RL > NLP > CV (按前沿程度)
  4. 如果你只是兴趣 → 选你觉得最酷的那个 (热爱才能坚持)
```

---

## 8. 选定方向后怎么深入?

> 选方向只是开始, 关键是下一步的行动

```python
# ============================================
# 深入一个方向的路线图
# ============================================

roadmap = {
    'Phase 1': '巩固基础 (1-2 周)',
    'Phase 2': '精读经典论文 (2-4 周)',
    'Phase 3': '复现经典项目 (2-4 周)',
    'Phase 4': '做自己的项目 (4-8 周)',
    'Phase 5': '竞赛 / 开源贡献 / 实习 (持续)',
}

phases_details = {
    'Phase 1': [
        '把你选择的那个方向在 35 天里对应的 Day 重新看一遍',
        '确保每个代码块都能独立写出来 (不照抄)',
        '整理笔记, 画知识图谱',
    ],
    'Phase 2': [
        '找该方向 5-10 篇经典论文, 精读 (+ 代码)',
        'CV: 从 AlexNet → ResNet → ViT → ConvNeXt',
        'NLP: 从 Transformer → BERT → GPT → LLaMA',
        '多模态: CLIP → BLIP2 → LLaVA → GPT-4V',
        'RL: DQN → PPO → SAC → RLHF',
    ],
    'Phase 3': [
        '复现 2-3 个经典项目的核心逻辑',
        '不要只跑通, 要改代码 + 做实验',
        '记录实验结果: 改了什么参数? 效果怎么变?',
    ],
    'Phase 4': [
        '找一个你感兴趣的实际问题',
        '用你学到的技术解决它',
        '写 README + 技术博客',
    ],
    'Phase 5': [
        '参加 Kaggle 竞赛 (该方向)',
        '给开源项目提 PR (修 bug / 加 feature)',
        '找实习 / 做项目 → 积累简历',
    ],
}

print('深入路线图:')
print()
for phase, title in roadmap.items():
    print(f'  {phase}: {title}')
    for step in phases_details[phase]:
        print(f'    → {step}')
    print()
```

---

## 9. 推荐学习资源

> 每个方向的经典资源, 少走弯路

```python
# ============================================
# 四大方向资源推荐
# ============================================

resources = {
    'CV': {
        '课程': 'Stanford CS231n (李飞飞团队, 经典必看)',
        '论文': 'ResNet / ViT / YOLO / SAM',
        '框架': 'torchvision / OpenCV / MMDetection / Detectron2',
        '数据集': 'ImageNet / COCO / OpenImages / ADE20K',
        '竞赛': 'Kaggle CV 竞赛 / LVIS / ECCV 挑战赛',
    },
    'NLP': {
        '课程': 'Stanford CS224n / Stanford CS324 (LLM)',
        '论文': 'BERT / GPT-3 / LLaMA / InstructGPT',
        '框架': 'HuggingFace Transformers / LangChain / vLLM',
        '数据集': 'GLUE / SQuAD / Natural Questions / C4',
        '竞赛': 'Kaggle NLP / SCROLLS / BIG-Bench',
    },
    'Multimodal': {
        '课程': 'Stanford CS231n + CS224n (两者结合)',
        '论文': 'CLIP / BLIP2 / LLaVA / Flamingo / Qwen-VL',
        '框架': 'HuggingFace Transformers / OpenCLIP / Diffusers',
        '数据集': 'LAION-5B / COCO Captions / VizWiz',
        '竞赛': 'HuggingFace Community / VQA Challenge',
    },
    'RL': {
        '课程': 'David Silver RL / UC Berkeley CS285 / Spinning Up',
        '论文': 'DQN / PPO / SAC / RLHF (InstructGPT)',
        '框架': 'Gymnasium / Stable-Baselines3 / rl4lm',
        '数据集': 'Atari / Mujoco / DM Control',
        '竞赛': 'NeurIPS 竞赛 (MineRL / Procgen)',
    },
}

for direction, res in resources.items():
    print(f'  【{direction}】')
    print(f'    课程: {res["课程"]}')
    print(f'    论文: {res["论文"]}')
    print(f'    框架: {res["框架"]}')
    print(f'    数据集: {res["数据集"]}')
    print(f'    竞赛: {res["竞赛"]}')
    print()
```

```text
通用资源 (所有方向):
  - 论文阅读: arxiv.org / paperswithcode.com / s2orc
  - 代码复现: huggingface.co/papers / github.com/resources-flow
  - 面试准备: leetcode / 牛客网 / 各方向八股文
```

---

## 10. 最后 — 不用焦虑

> 选方向不是"一锤子买卖"

```python
# ============================================
# 关于方向选择的真相
# ============================================

truths = [
    '方向选了不是不能换 — 很多算法工程师工作 2 年后转了方向',
    '你选的方向可能 3 年后就不存在了 — 但学到的能力一直在',
    '最好的方向 = 你愿意花周末时间研究的方向',
    '在任何一个方向做到前 20%, 都比在所有方向做到前 80% 好',
    '基础能力 (Python / 数学 / 工程) 比方向更重要',
    '35 天给了你"全栈"视野, 这是很多人没有的优势',
]

print('关于方向选择的 6 个真相:')
print()
for i, t in enumerate(truths, 1):
    print(f'  {i}. {t}')
```

```text
给未来的你:
  今天选了 CV            并不是放弃了 NLP
  今天选了 NLP           不代表不能做多模态项目
  今天选了 多模态         3 年后可能发现新方向
  今天选了 RL            你学会的"如何学习"比 RL 本身更重要

35 天最大的收获不是学会了什么模型,
而是你证明了: 你能用 35 天从 0 到走进深度学习的大门。
这种学习能力, 远比今天选哪个方向重要。
```

---

## Day 34 完成

### 今天学到了什么

| 概念 | 一句话 |
|:----|:------|
| **CV** | 让机器看见, 图像分类 → 目标检测 → 分割 |
| **NLP** | 让机器读懂语言, BERT → GPT → 大模型应用 |
| **多模态** | 融合文本+图像+语音, 2025-2026 最前沿 |
| **RL** | 让机器学会做决策, 游戏 → 机器人 → RLHF |
| **选择方法** | 看兴趣 + 看岗位 + 看数学基础 |
| **深入路线** | 巩固 → 精读论文 → 复现 → 做项目 → 竞赛 |

## 35 天完成状态

```
Day 1-7:   经典 ML    ✅ 算法基础
Day 8-14:  神经网络    ✅ 深度学习入门
Day 15-21: 序列模型    ✅ Transformer
Day 22-28: 预训练生成  ✅ BERT/GPT/ViT/CLIP/GAN/Diffusion
Day 29-30: Kaggle     ✅ 实战
Day 31:    YOLO       ✅ 目标检测
Day 32:    ONNX       ✅ 模型部署
Day 33:    GitHub+博客 ✅ 作品包装
Day 34:    方向选择    ✅ 今天
Day 35:    最终总结    ← 明天 (知识地图)
```

## 作业

---

### 作业 1: 做方向选择自测

**位置:** 第 7 节, 选择自测题

回答自测题, 看看你第一反应选什么

<details>
<summary>📖 点击查看答案</summary>

```text
参考结果:
  A 多 → CV
  B 多 → NLP
  C 多 → 多模态
  D 多 → RL

建议:
  第一反应选什么就是什么 — 别纠结
  如果你在 A 和 B 之间犹豫 → 选多模态 (两个都涉及)
  如果你觉得都差不多 → NLP (岗位最多, 最稳)
```

</details>

---

### 作业 2: 列出你的下一步计划

**位置:** 第 8 节, 深入路线图

按 Phase 1-5 写出你的具体行动计划

<details>
<summary>📖 点击查看答案</summary>

```text
示例计划 (选了 NLP 方向):

Phase 1 (1-2 周):
  → 重新看 Day 19 (Transformer) + Day 22 (BERT) + Day 23 (GPT)
  → 确保能手写 Self-Attention 的代码
  → 整理 NLP 知识图谱

Phase 2 (2-4 周):
  → 精读 BERT 论文 + GPT-2 论文
  → 用 HuggingFace 跑 3 个不同的 NLP 任务
  → 读 InstructGPT 论文 (RLHF 基础)

Phase 3 (2-4 周):
  → 复现一个简单的 GPT 训练
  → 做一个 RAG 问答系统 (LangChain)
  → 微调一个小模型

Phase 4 (4-8 周):
  → 做一个垂直领域的 chatbot
  → 部署到 HuggingFace Spaces
  → 写技术博客

Phase 5 (持续):
  → 参加 Kaggle NLP 竞赛
  → 给开源项目贡献
  → 准备面试
```

</details>

---

### 作业 3: 看一门该方向的公开课第一课

**位置:** 第 9 节, 资源推荐

选定方向后, 去把推荐的课程第一课看完

<details>
<summary>📖 点击查看答案</summary>

```text
链接直达:
  CS231n (CV): https://www.youtube.com/watch?v=vT1JzSTHric
  CS224n (NLP): https://www.youtube.com/watch?v=rmVRLeT26bo
  David Silver (RL): https://www.youtube.com/watch?v=2pWv7GOvuf0
  CS285 (RL 进阶): https://www.youtube.com/watch?v=JHcTsai60Ng

看第一课就够了:
  → 了解课程风格适不适合你
  → 看 outline 是否符合你的目标
  → 如果第一课看不下去 → 换一个资源
  → 如果第一课觉得"这就是我想要的" → 坚持看完
```

</details>
