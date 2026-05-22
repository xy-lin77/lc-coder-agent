# RL 后训练学习笔记

| 章节 | 主题 | 文件 |
|------|------|------|
| 第 0 章 | 强化学习基础（MC / TD / DQN / PG / AC / TRPO / PPO / GRPO） | [0-rl-basics.md](notes/0-rl-basics.md) |
| 第 1 章 | LLM 语境下的 V、Q 和 Advantage | [1-llm-rl.md](notes/1-llm-rl.md) |
| 第 2 章 | PPO / DPO / GRPO 概览与对比 | [2-ppo-dpo-grpo.md](notes/2-ppo-dpo-grpo.md) |
| 第 3 章 | GAE + PPO-Clip 详解 | [3-gae-ppo-clip.md](notes/3-gae-ppo-clip.md) |

**[本项目 README](../README.md)**（SFT + GRPO 代码推理后训练方案）

---

<details>
<summary><b>第 0 章 · 强化学习基础</b> &nbsp;—&nbsp; MC · TD · SARSA · Q-learning · DQN · Policy Gradient · REINFORCE · AC · TRPO · PPO · GRPO</summary>

# 强化学习基础

## 分类维度

**value-based vs policy-based**

| | value-based | policy-based |
|---|---|---|
| 学什么 | 学价值函数 $V(s)$ 或 $Q(s,a)$，用价值间接推导动作 | 直接学策略 $\pi_\theta(a\|s)$，输出动作概率 |
| 代表算法 | MC、TD、SARSA、Q-learning、DQN | Policy Gradient、REINFORCE、PPO、GRPO |
| 特点 | 有明确的价值估计，可解释；动作空间大时效率低 | 天然支持连续/大动作空间；无显式价值，方差较高 |

**on-policy vs off-policy**

| | on-policy | off-policy |
|---|---|---|
| 数据来源 | 只能用**当前策略**自己采集的数据更新 | 可以用**任意策略**采集的历史数据更新 |
| 代表算法 | MC、TD、SARSA、PPO、GRPO | Q-learning、DQN |
| 特点 | 数据与策略强绑定，更新后旧数据作废；稳定但样本效率低 | 可复用历史数据，样本效率高；但需处理分布偏移 |

---

## 蒙特卡洛（MC）

- 分类：value-based，on-policy
- 原理：从完整回合中获取每一步的回报 $G_t$，以所有访问该状态时 $G_t$ 的均值估计 $V(s)$，无 Bellman 递推
- 直觉：下完一整盘棋，赢了给所有经过的位置记 +1，输了记 -1，下 100 盘后取平均。只靠完整棋局统计，下到一半不能更新。

---

## 时序差分学习（TD）

- 分类：value-based，on-policy
- 原理：结合 MC 采样与 DP 递推，每步更新一次

$$V(s_t) \leftarrow V(s_t) + \alpha\bigl[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\bigr]$$

- 直觉：不用下完整盘，走一步就用"这步奖励 + 下一位置的价值"修正当前位置的估计，比 MC 更高效。

---

## SARSA

- 分类：value-based，on-policy
- 原理：学习动作价值函数 $Q(s,a)$，更新依赖五元组 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl[r + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\bigr]$$

- 直觉：按"保守策略"走了 $a$，得到奖励，再按同一策略走 $a'$，只用这组真实走法更新。想换策略就必须重新收集数据。

---

## Q-learning

- 分类：value-based，off-policy
- 原理：直接学最优 $Q^*(s,a)$，更新时取下一状态所有动作的最大值

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl[r + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)\bigr]$$

- 直觉：可以看任何人的下棋录像，学习时不管别人怎么走，只关心下一步最优走法的价值，数据来源灵活。

---

## DQN（深度 Q 网络）

- 分类：value-based，off-policy
- 原理：用神经网络拟合 $Q(s,a)$，引入经验回放与目标网络稳定训练

$$L = \mathbb{E}\bigl[(r + \gamma \max_{a'} Q'(s_{t+1},a') - Q(s_t,a_t))^2\bigr]$$

- 直觉：把棋盘画面输入网络，输出每个落子位置的 Q 值。存下所有对局经验，随机抽取训练，靠网络泛化没见过的棋局。

---

## Policy Gradient（策略梯度）

- 分类：policy-based，on-policy
- 原理：直接优化策略 $\pi_\theta(a|s)$，最大化期望回报

$$\nabla J(\theta) = \mathbb{E}\bigl[\nabla \log \pi_\theta(a|s) \cdot G_t\bigr]$$

- 直觉：直接输出落子概率分布，赢了就把这盘所有走法的概率调高，输了调低，不需要先估价值。

---

## REINFORCE

- 分类：policy-based，on-policy
- 原理：蒙特卡洛策略梯度，完整回合结束后用 $G_t$ 更新

$$\theta \leftarrow \theta + \alpha \nabla \log \pi_\theta(a_t|s_t) \cdot G_t$$

- 直觉：下完整盘后回溯每一步，赢了就提高每步选择的概率。实现最简单，但方差较大。

---

## Actor-Critic（AC）

- 分类：actor-critic 混合，on-policy
- 原理：Actor 输出动作概率，Critic 用 TD 误差 $\delta = r + \gamma V(s') - V(s)$ 评价动作好坏，Actor 用 $\delta$ 替代 $G_t$ 更新
- 直觉：Actor 负责落子，Critic 站旁边每走一步就打分，Actor 根据打分调整概率。不用等下完整盘，比 REINFORCE 更稳。

---

## TRPO

- 分类：policy-based，on-policy
- 原理：用共轭梯度 + 线搜索保证单调策略提升，KL 散度约束新旧策略差异
- 直觉：更新时严格限制策略变化幅度，确保新模型效果不会突变，每步更新都能保证不降，适合大模型安全训练。

---

## PPO（近端策略优化）

- 分类：policy-based，on-policy（clip 版）
- 原理：简化 TRPO，用裁剪函数替代 KL 约束，可多次复用同一批数据

$$L(\theta) = \mathbb{E}\bigl[\min\bigl(r_t(\theta) \cdot A_t,\ \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t\bigr)\bigr]$$

- 直觉：对同一批对话数据反复小幅度更新模型，不让模型学得太激进，兼顾效果与稳定性，是 RLHF 最常用算法。

---

## GRPO（分组相对策略优化）

- 分类：policy-based，on-policy
- 原理：去掉 Critic，对同一 prompt 采样 $G$ 条回复，用组内 reward 归一化直接得到 Advantage

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

- 直觉：给模型同一个问题，生成多个回答，reward 高于组内均值的回答提升概率，低于均值的降低。无需 Critic，训练更轻量，适合数学、代码等答案差异大的任务。

</details>

---

<details>
<summary><b>第 1 章 · LLM 语境下的 V、Q 和 Advantage</b> &nbsp;—&nbsp; 状态价值 · 动作价值 · 优势函数 · Bellman 方程 · Critic 归因</summary>

# 理解 LLM 语境下的 V, Q 和 Advantage

## 建立直觉：类比"下棋"

强化学习的场景可以抽象成：

```
一个智能体（Agent）在环境中不断做决策，
每次决策后环境给出反馈（reward），
目标是最大化长期累积 reward。
```

对应到 LLM：
```
Agent = 语言模型
环境 = 人类偏好 / 规则评分器
决策 = 每次生成一个 token
Reward = 最终对整个回复的打分
```

---

## RL 核心概念

| 符号 | 名称 | 含义 | LLM 对应 |
|------|------|------|---------|
| `s` | State 状态 | 当前所处的情境 | prompt + 已生成的 token |
| `a` | Action 动作 | 在当前状态下的决策 | 生成下一个 token |
| `r` | Reward 奖励 | 执行动作后的即时反馈 | 最终打分（稀疏） |
| `π` | Policy 策略 | 给定状态，选择动作的规则 | 语言模型本身 |
| `γ` | Discount factor | 对未来 reward 的折扣系数，取值 (0,1) | 通常 0.99；体现"远期 reward 不如近期确定"，γ 越小越短视 |
| `G_t` | Return 回报 | 从 t 时刻起，未来所有 reward 的折扣累加：`r_t + γ·r_{t+1} + γ²·r_{t+2} + ...` | LLM 中 reward 集中在末尾，G_t 退化为终端分数乘以折扣系数 |

---

## V 函数：一个状态"值多少钱"

```
V^π(s) = 从状态 s 出发，按策略 π 行动，期望能拿到的 G_t
```

**下棋类比：**
```
开局（s_0）：V = 0.5    （五五开）
你走出好棋（s_5）：V = 0.8   （你占优）
对手走出神之一手（s_8）：V = 0.2  （你处于劣势）
```

**LLM 里的 V：**
```
s_t = "今天天气真的很____"（已生成到这里）
V(s_t) ≈ 0.7   意思是：从这个上下文继续生成，预计最终能得到 0.7 分的 reward
```

Critic 神经网络 = 学 V 函数的模型，输入 $s_t$，输出预测的 $V(s_t)$。

---

## Q 函数：一个（状态+动作）值多少钱

```
Q^π(s, a) = 在状态 s 执行动作 a，然后按策略 π 走下去，期望能拿到的 G_t

V(s) = Σ_a π(a|s) · Q(s,a)   （对所有可能动作加权平均）
```

---

## Advantage：这个动作比平均水平好多少

```
A^π(s, a) = Q(s, a) - V(s)

A > 0：这个动作比平均水平好，应该强化
A < 0：这个动作比平均水平差，应该抑制
A = 0：这个动作和平均水平一样，无需调整
```

**为什么不直接用 Q，要用 Advantage：**
```
开局状态：Q(好棋) = 0.8,  Q(差棋) = 0.6   → 差距 0.2
残局状态：Q(好棋) = 0.99, Q(差棋) = 0.3   → 差距 0.69
```
不同状态的绝对价值差异很大，用 Advantage 归一化后才能公平比较，梯度更稳定。

---

## Bellman 方程：RL 核心等式

$$V(s_t) = r_t + \gamma \cdot V(s_{t+1})$$

TD error 就是 Advantage 的单步估计：

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) = r_t + \gamma \cdot V(s_{t+1}) - V(s_t) = \delta_t$$

---

## 概念关系总图

```
环境给出稀疏 Reward
         ↓
G_t = 折扣累积 Reward（Return）
         ↓
    ┌────┴────┐
    V(s)     Q(s,a)
  状态价值   状态-动作价值
    └────┬────┘
         ↓
   A(s,a) = Q - V
     动作的相对好坏
         ↓
   PPO 用 A 来更新 Policy：
   强化 A>0 的 token
   抑制 A<0 的 token
```

---

## LLM 完整图景

```
生成序列：x → y_1 → y_2 → y_3 → ... → y_T → Reward = 0.8

Critic 估计每步的 V：
  V(x)          = 0.5
  V(x,y_1)      = 0.55
  V(x,y_1,y_2)  = 0.6
  ...
  V(x,...,y_T)  = 0.8

计算每步的 Advantage（TD error）：
  A_1 = r_1 + γ·V(s_2) - V(s_1) = 0 + 0.99·0.6 - 0.55 = +0.044
  A_T = r_T + 0 - V(s_T)        = 0.8 - 0.8 = 0

PPO 更新：
  A_t > 0 → 增大生成 y_t 的概率
  A_t < 0 → 减小生成 y_t 的概率

Critic 的核心价值：把末端的一个分数，通过 Bellman 方程
逐步往前归因，让每个 token 都知道"自己对最终结果贡献了多少"。
```

</details>

---

<details>
<summary><b>第 2 章 · PPO / DPO / GRPO 概览</b> &nbsp;—&nbsp; 四模型架构 · 工程实现 · 交互流程 · 三算法对比</summary>

# PPO

## 1. 四个模型

| 模型 | 核心作用 | 额外改造 | 训练目标 & 损失函数 |
| ---- | -------- | -------- | ------------------- |
| Actor | SFT训练后的模型，待优化的目标生成模型 | 无 | 最大化动作优势，同时裁剪策略更新幅度，防止模型单次更新过大、训练崩塌。使用PPO-Clip损失 $L = \mathbb{E}\bigl[\min\bigl(r_t A_t,\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t\bigr)\bigr]$，其中 $r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}$ |
| Critic | SFT训练后的模型，预测从当前token位置到序列结束的未来累计总收益期望（$V_t$），用于计算优势函数 | 在每一个Token的隐层状态后拼接Value预测头 | 拟合真实时序回报与GAE估值，使用MSE损失 $L = \mathbb{E}\big[(V_t - V_t^{target})^2\big]$ |
| Reference | 冻结的SFT训练模型，固定输出分布，用于计算KL惩罚项，约束Actor更新幅度 | 无 | 无 |
| Reward Model | 偏好打分模型，预测单条完整问答序列的全局偏好分数 $r(x,y)$，提供原始奖励信号 | 在序列最后一个Token的隐层状态上拼接Reward打分头 | 学习人类偏好排序 $L = -\log\big(r(x, y_{winner}) - r(x, y_{loser})\big)$ |

---

## 2. 工程实现

### 学术论文（全量微调，如 InstructGPT、Llama 2）
- Reward Model = SFT backbone 全量更新 + value head
- Critic = SFT backbone 全量更新 + value head

### 工业界（PPO 权重共享 + ZeRO3 + LoRA）
1. **权重复用**：Actor 与 Reference 共享同一底座，仅通过开关 LoRA 适配器区分；Critic 与 RM 共享主干。显存常驻权重减少至2份
2. **LoRA 轻量化训练**：冻结 SFT 主干，仅训练 LoRA 适配器与专属 Value 头
3. **DeepSpeed ZeRO3**：权重分片、显存卸载、梯度分片
4. **Reward 信号**：主观对话用独立小参数 RM；代码/数学/工具调用用规则函数

---

## 3. 交互流程

1. **Actor 生成回复**：采样完整回复 $y \sim \pi_{\theta_{\text{old}}}(y \mid x)$
2. **Reward Model 打分**：输出原始奖励 $r(x, y)$
3. **Reference 计算 KL 惩罚**：$r_t = r(x,y) - \beta \cdot \text{KL}(\pi_{\theta_{\text{old}}} \parallel \pi_{\text{ref}})$
4. **Critic 逐 Token 预测 V & 计算 GAE**：输出 $V(s_t)$，回溯得到 $A_t$ 与 $V_t^{\text{target}}$
5. **更新 Actor**：$L_{\text{Actor}} = \min\left(\text{ratio} \cdot A,\ \text{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon) \cdot A\right)$
6. **更新 Critic**：$L_{\text{Critic}} = \mathbb{E}\big[(V_t - V_t^{\text{target}})^2\big]$
7. 循环迭代

---

# DPO

## 两个模型
- **Policy**（对应 PPO Actor）+ **Reference**
- 移除 RM、Critic，无价值估计、优势函数计算

## 交互流程（无强化学习循环，一步训练）

1. 输入构造：给定 `x`，采样一对回复 `(y_w, y_l)`
2. 双模型前向计算：Policy 和 Reference 分别计算 `y_w`、`y_l` 的对数概率
3. 偏好损失：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right)\right]$$

4. 直接反向传播，无 Clip、GAE、多模型交替更新

---

# GRPO

## 三个模型
- **Actor** + **Reference** + **RM**，移除 Critic

## 与 PPO 差异：Advantage 计算

PPO 用 Critic 逐 token 估 $V(s_t)$，再通过 GAE 回溯得到 $A_t$。

GRPO 对同一 prompt 采样 $G$ 条回复，用**组内 reward 的相对排名**直接得到 Advantage：

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$

## 交互流程

1. **Actor 批量采样**：对同一 prompt $x$ 采样 $G$ 条回复
2. **Reward Model 打分**：每条回复输出 $r_i$
3. **组内归一化**：计算 $\hat{A}_i$
4. **更新 Actor**（无 Critic 更新）：

$$L_{\text{GRPO}} = \min\left(\text{ratio} \cdot \hat{A}_i,\ \text{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon) \cdot \hat{A}_i\right) - \beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}})$$

5. 循环迭代

</details>

---

<details>
<summary><b>第 3 章 · GAE + PPO-Clip 详解</b> &nbsp;—&nbsp; Critic 结构 · GAE 推导 · λ 的作用 · Clip 直觉 · 完整数据流</summary>

# GAE + PPO-Clip 详解

## 一、PPO 整体训练循环

PPO 是一个迭代的 **采样 → 估计 → 更新** 循环：

```
┌─────────────────────────────────────────────────────┐
│  1. Actor 采样一批轨迹                               │
│  2. Critic 估计每个 token 的 Advantage               │
│  3. 用 Advantage 计算 PPO Loss                       │
│  4. 反向传播，更新 Actor 和 Critic 的权重             │
│  5. 回到第 1 步                                      │
└─────────────────────────────────────────────────────┘
```

---

## 二、Critic 如何预测 Advantage

### Critic 的结构

```
LLM 主干（Transformer layers）
         ↓
   hidden state h_t  （维度 d_model，如 4096）
         ↓
   Linear(d_model → 1)   ← value head，只有这一层是新加的
         ↓
   V(s_t)  （一个标量）
```

每个 token 位置都输出一个标量，代表"从这里往后的期望 return"。

### 从 V 到 Advantage：GAE 计算过程

```python
# 伪代码：一条轨迹结束后
rewards = [0, 0, 0, ..., 0, R_final]     # 只有最后一步有 reward
values  = critic(s_0, s_1, ..., s_T)     # Critic 预测每步的 V

# 第一步：计算每步的 TD error
deltas = [rewards[t] + γ * values[t+1] - values[t] for t in range(T)]

# 第二步：GAE 向前累积
gae = 0
advantages = []
for t in reversed(range(T)):
    gae = deltas[t] + γ * λ * gae        # 指数加权累积
    advantages.insert(0, gae)
```

---

## 三、GAE（Generalized Advantage Estimation）

### 两种极端估计方法

**方法一：单步 TD（高 bias，低 variance）**
$$A_t \approx \delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$$

**方法二：Monte Carlo（低 bias，高 variance）**
$$A_t \approx G_t - V(s_t) = (r_t + \gamma \cdot r_{t+1} + \ldots) - V(s_t)$$

### GAE：在两者之间插值

$$A_t^{\text{GAE}} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \ldots + (\gamma\lambda)^{T-t}\delta_T$$

### λ 的作用

```
λ = 0：A_t = δ_t                 ← 退化为单步 TD，高 bias 低 variance
λ = 1：A_t = G_t - V(s_t)        ← 退化为 Monte Carlo，低 bias 高 variance
0 < λ < 1：bias 和 variance 都适中，实践中通常取 λ = 0.95
```

### 为什么权重指数衰减

越远的 TD error，受 V 预测误差累积影响越大，可信度越低：
```
δ_t     权重 = 1          （最近，最可信）
δ_{t+1} 权重 = γλ
δ_{t+2} 权重 = (γλ)²
δ_{t+3} 权重 = (γλ)³     （越远，越不可信）
```

---

## 四、PPO-Clip 推导

### ratio 与 PPO Loss

$$\text{ratio}_t = \frac{\pi_\theta(y_t|s_t)}{\pi_{\theta_\text{old}}(y_t|s_t)}$$

$$L_{\text{CLIP}} = \mathbb{E}_t\bigl[\min\bigl(\text{ratio}_t \cdot A_t,\ \text{clip}(\text{ratio}_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t\bigr)\bigr]$$

### Clip 的直觉

```
A_t > 0（好 token）：
  ratio < 1+ε → 正常按比例鼓励，增大概率
  ratio > 1+ε → 截断，不再给额外 reward（防止更新过猛）

A_t < 0（坏 token）：
  ratio > 1-ε → 正常按比例惩罚
  ratio < 1-ε → 截断，不再额外惩罚（防止过度抑制）
```

**本质：好事不能做过头，坏事不要惩罚过度，每次更新控制在安全范围内。**

### KL 惩罚

$$L_{\text{total}} = L_{\text{CLIP}} - \beta \cdot \text{KL}[\pi_\theta \| \pi_{\text{ref}}]$$

---

## 五、完整数据流总结

```
【采样阶段】（不更新权重）
  Actor 生成序列 y_1...y_T
  记录每步的 log π_θ_old(y_t|s_t)
  Reward Model 给出最终分数 R

【估计阶段】（不更新权重）
  Critic 前向传播，输出 V(s_0)...V(s_T)
  计算 TD error δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
  GAE 累积得到每步的 A_t

【更新阶段】（权重更新）
  Actor：
    重新前向传播，得到新的 log π_θ(y_t|s_t)
    计算 ratio = exp(log π_θ - log π_θ_old)
    计算 L_CLIP → 反向传播 → 更新所有 Transformer 层权重

  Critic：
    重新前向传播，得到新的 V(s_t)
    计算 L_critic = (V(s_t) - R_t)²
    反向传播 → 更新所有层权重（含 value head）

【循环】
  用更新后的 Actor 重新采样，开始下一轮
```

Critic 负责**看懂局面**（估计 V），Actor 负责**改进决策**（更新 logit），两者交替训练，共同收敛。

</details>
