# RL 后训练学习笔记

| 章节 | 主题 | 文件 |
|------|------|------|
| 第 0 章 | 强化学习基础（MC / TD / DQN / PG / AC / TRPO / PPO / GRPO） | [0-rl-basics.md](notes/0-rl-basics.md) |
| 第 1 章 | LLM 语境下的 V、Q 和 Advantage | [1-llm-rl.md](notes/1-llm-rl.md) |
| 第 2 章 | PPO / DPO / GRPO 概览与对比 | [2-ppo-dpo-grpo.md](notes/2-ppo-dpo-grpo.md) |
| 第 3 章 | GAE + PPO-Clip 详解 | [3-gae-ppo-clip.md](notes/3-gae-ppo-clip.md) |
| 第 4 章 | DAPO / Dr.GRPO：基于 GRPO 的改动 | [4-dapo-drgrpo.md](notes/4-dapo-drgrpo.md) |
| 第 5 章 | 项目实践：Qwen2.5-7B SFT + GRPO 代码推理后训练 | [../README.md](../README.md) |

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

---

<details>
<summary><b>第 4 章 · DAPO / Dr.GRPO</b> &nbsp;—&nbsp; Clip-Higher · Dynamic Sampling · Per-Token Loss · Length Bias</summary>

# DAPO 与 Dr.GRPO：基于 GRPO 的改动

GRPO 的核心是：同一个 prompt 采样多条回答，用组内 reward 归一化得到 advantage，不训练 critic。DAPO 和 Dr.GRPO 都保留了这个基本框架，但重点修正了两个实践问题：

- **更新太保守或探索不足**：正确回答的概率被 PPO clip 过早截断，模型难以继续放大高质量轨迹。
- **长度偏置**：同一个序列级 reward 被分配到所有 token 上，loss 的归一化方式会改变长回答和短回答的梯度权重。

---

## 一、GRPO 基线

对每个 prompt 采样 $G$ 条回答 $o_1,\ldots,o_G$，每条回答得到一个 reward $r_i$。GRPO 用组内均值和标准差归一化：

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1,\ldots,r_G)}{\mathrm{std}(r_1,\ldots,r_G)}$$

每条回答内部的所有 token 共享同一个 $\hat{A}_i$。先定义第 $i$ 条回答第 $t$ 个 token 的 clipped loss：

$$c_{i,t} = \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)$$

$$\ell_{i,t} = \min(\rho_{i,t}\hat{A}_i,\ c_{i,t}\hat{A}_i)$$

再对组内回答和回答内 token 求平均：

$$L_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\ell_{i,t}$$

其中：

$$\rho_{i,t}=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$$

### GRPO 的特点

| 设计 | 作用 | 潜在问题 |
|---|---|---|
| 组内 reward 标准化 | 不需要 critic，对 reward 尺度不敏感 | 当组内 reward 方差很小时，advantage 可能被放大 |
| 每条序列先做 token 平均 | 每条回答权重相同 | 长回答和短回答的 token 梯度权重不一致 |
| 对称 clip | 上下界都是 $\epsilon$ | 对正 advantage 样本的鼓励可能过早被截断 |

---

## 二、DAPO：让 GRPO 更适合长链推理训练

DAPO 可以理解为一组针对 GRPO 的工程化改造，核心目标是提升长链推理 RL 的有效样本利用率和训练稳定性。

### 1. Clip-Higher

PPO / GRPO 常用对称裁剪：

$$\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)$$

DAPO 将上下界解耦：

$$\mathrm{clip}(\rho,1-\epsilon_{\text{low}},1+\epsilon_{\text{high}})$$

并设置更高的上界：

$$\epsilon_{\text{high}} > \epsilon_{\text{low}}$$

直觉：

- $\hat{A}>0$ 的回答是组内更好的回答，应该允许模型更充分地提高这些 token 的概率。
- 更高的 upper clip 可以减少“好样本刚开始变好就被截断”的问题。
- lower clip 保持相对保守，避免对负样本过度更新。

### 2. Dynamic Sampling

GRPO 对一个 prompt 采样 $G$ 条回答。如果 $G$ 条全对或全错，组内相对优势几乎没有有效区分度：

```text
全对：reward = [1, 1, 1, 1]  → 没有谁比谁更好
全错：reward = [0, 0, 0, 0]  → 没有可学习的正样本
有差异：reward = [1, 0, 1, 0] → 有清晰相对信号
```

DAPO 会优先保留 reward 有区分度的 prompt，过滤掉全对或全错的组，提升每个 batch 的有效梯度密度。

### 3. Token-Level Policy Gradient Loss

GRPO 常见写法是 **per-sequence average，再对 batch 平均**：

$$\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t L_{i,t}$$

DAPO 改为 **全局 token 平均**：

$$\frac{1}{\sum_i |o_i|}\sum_i\sum_t L_{i,t}$$

差异在于：GRPO 让每条回答的总权重接近相同，DAPO 让每个 token 的权重接近相同。长链推理任务中，较长回答通常包含更多推理步骤；token-level loss 可以避免长回答在序列级平均中被过度压缩。

### 4. Overlong Filtering / Soft Penalty

长链推理 RL 容易出现“越想越长”的问题。DAPO 会对超长回答进行过滤或软惩罚：

- 超过最大长度且没有给出最终答案的样本，降低 reward 或直接过滤。
- 接近长度上限的回答可以给平滑惩罚，避免模型只学会拖长推理。

---

## 三、Dr.GRPO：修正 GRPO 的长度偏置

Dr.GRPO 关注的问题更集中：**GRPO 的 loss 归一化会引入长度偏置**。

在序列级 reward 场景里，一条回答只有一个 $\hat{A}_i$，然后这个 advantage 被复制到所有 token。此时如何归一化 loss，会直接决定“长回答”和“短回答”谁的梯度更大。

### Dr.GRPO 的核心改动

Dr.GRPO 倾向于用固定长度常数归一化，而不是用每条回答自己的长度归一化：

$$L_{\text{Dr.GRPO}} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{L_{\max}}\sum_{t=1}^{|o_i|} L_{i,t}$$

其中 $L_{\max}$ 通常取最大生成长度或配置中的 response length 上限。

这样做的含义是：

- 不再让短回答因为 $1/|o_i|$ 更大而天然获得更大的单 token 权重。
- 长回答的总梯度会随有效 token 数增加，但上界由 $L_{\max}$ 控制。
- 对长链推理更中性，避免模型仅因为 loss 形式偏好短输出。

Dr.GRPO 也常与“去掉组内 std 缩放”一起讨论：只减去组内均值，避免 reward 方差很小时 advantage 被异常放大。

---

## 四、Per-Seq / Per-Token / Dr.GRPO 对比

设 $L_{i,t}$ 表示第 $i$ 条回答第 $t$ 个 token 的 clipped policy gradient loss。

| 写法 | 谁的权重相同 | 对长度的影响 | 代表 |
|---|---|---|---|
| Per-seq per-token | 每条回答权重相同 | 短回答单 token 权重大，长回答单 token 权重小 | 原始 GRPO 常见实现 |
| Per-token | 每个 token 权重相同 | 长回答因 token 更多，总权重更大 | DAPO |
| Fixed-length seq norm | 每条回答有固定上限 | 减少短回答优势，长回答不会无限放大 | Dr.GRPO |

**Per-seq per-token：**

$$\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t L_{i,t}$$

**Per-token：**

$$\frac{1}{\sum_i |o_i|}\sum_i\sum_t L_{i,t}$$

**Fixed-length seq norm：**

$$\frac{1}{G}\sum_i\frac{1}{L_{\max}}\sum_t L_{i,t}$$

一个简单例子：

```text
回答 A：长度 100，advantage > 0
回答 B：长度 1000，advantage > 0
```

- Per-seq per-token：A 和 B 的总权重接近相同，B 的每个 token 权重只有 A 的 1/10。
- Per-token：A 的总权重约为 B 的 1/10，所有 token 同权。
- Dr.GRPO：A 的总权重约为 $100/L_{\max}$，B 的总权重约为 $1000/L_{\max}$，但都受固定长度上限约束。

---

## 五、总结

| 方法 | 基于 GRPO 的主要改动 | 解决的问题 |
|---|---|---|
| GRPO | 组内相对 reward，去掉 critic | 降低显存和训练复杂度 |
| DAPO | Clip-Higher、Dynamic Sampling、token-level loss、overlong penalty | 提高有效样本利用率，增强正样本学习，适配长链推理 |
| Dr.GRPO | 固定长度归一化，常搭配去 std 缩放 | 减少长度偏置和 advantage 异常放大 |

面试表达可以压缩成一句话：

> DAPO 更像是把 GRPO 做成可扩展的长链推理训练系统，重点在采样、clip 和 token-level loss；Dr.GRPO 更像是对 GRPO 目标函数的偏差修正，重点在长度归一化和 advantage 缩放。

</details>

---

<details>
<summary><b>第 5 章 · 项目实践</b> &nbsp;—&nbsp; Qwen2.5-7B · SFT · GRPO · HumanEval · LiveCodeBench · MBPP</summary>

# GRPO 代码推理后训练项目方案

基于 Qwen2.5-7B-Instruct，通过 SFT + GRPO 两阶段后训练，提升模型代码推理能力，并在标准 benchmark 上量化效果。

**计算平台**：HKUST SuperPOD，单节点 4 块 NVIDIA H800 80GB

---

## 技术栈

| 组件 | 工具 |
|---|---|
| 基座模型 | Qwen2.5-7B-Instruct |
| SFT 框架 | LLaMA-Factory |
| GRPO 框架 | verl（字节跳动开源） |
| 评测 | HumanEval、LiveCodeBench、MBPP |
| 集群调度 | Slurm |

---

## 整体流程

```
Qwen2.5-7B-Instruct
        ↓
   Stage 1: SFT          ← 教模型用 <think> 格式推理
        ↓
   Stage 2: GRPO         ← 用代码执行结果作为 reward 强化正确推理
        ↓
     评测模型
   HumanEval / LiveCodeBench / MBPP
```

---

## Stage 1：SFT

### 目的

让模型学会链式推理格式（`<think>...</think>`），为 GRPO 阶段提供足够的 reward 信号，解决冷启动问题。

### 数据构造

- 来源：LeetCode Easy/Medium 题目（约 800 题）
- 用 DeepSeek-R1 或 GPT-4o 批量生成带 CoT 的解题过程
- 数据格式：

```json
{
  "prompt": "题目描述...",
  "response": "<think>\n分析思路...\n</think>\n\n```python\n代码实现\n```"
}
```

### 训练配置

```bash
llamafactory-cli train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --stage sft \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_target all \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --bf16 \
    --output_dir ./sft_output
```

**关键决策**：使用 LoRA 而非全参数微调，节省显存，rank=64 对 7B 模型足够。

### 资源消耗

- 4 块 H800，约 4~6 小时

---

## Stage 2：GRPO

### 核心原理

GRPO（Group Relative Policy Optimization）对每道题采样 N 个回答，用组内相对奖励代替 PPO 中的 value function，**不需要额外的 critic 网络**，显存占用更低。

### Reward 函数设计

```python
def compute_reward(response: str, test_cases: list) -> float:
    reward = 0.0

    # 格式奖励
    if "<think>" in response and "</think>" in response:
        reward += 0.1

    # 提取代码
    code = extract_code_block(response)
    if not code:
        return reward

    # 执行测试用例（核心 reward）
    passed = 0
    for test in test_cases:
        try:
            result = execute_with_timeout(code, test["input"], timeout=5)
            if result == test["expected_output"]:
                passed += 1
        except:
            pass

    reward += (passed / len(test_cases)) * 0.9
    return reward
```

**Reward Hacking 防范**：
- 设置执行超时（5秒），防止死循环
- 检测 hardcode 输出行为

### 训练配置

```bash
python -m verl.trainer.main_ppo \
    algorithm=grpo \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=./sft_output \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=2
```

| 关键超参 | 值 | 说明 |
|---|---|---|
| `rollout.n` | 8 | 每题采样 8 个回答做组内对比 |
| `kl_coef` | 0.001 | 防止模型偏离 SFT 初始化太远 |
| `temperature` | 0.8 | 保证 rollout 多样性 |
| `lr` | 1e-6 | 比 SFT 小一个量级，稳定训练 |

### 资源消耗

- 4 块 H800，约 20 小时（2 epoch）

---

## 数据规划

```
LeetCode 题目
      ↓
筛选有标准测试用例的题目（Easy + Medium 约 800 题）
      ↓
┌─────────────────────────────────────────┐
│  SFT 数据（约 800 题）                   │
│  用 DeepSeek-R1 生成 CoT 解题过程        │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  GRPO 训练集（50~100 题，有可执行测试）   │
│  GRPO 验证集（20 题，监控 pass@1 曲线）   │
└─────────────────────────────────────────┘
```

推荐现成数据集：

- `leetcode-hard-gym`：有完整测试用例
- `APPS dataset`：有单元测试，难度梯度好
- `CodeContests`（DeepMind）：竞赛题，区分度高

---

## 评测指标

| 指标 | 说明 |
|---|---|
| **pass@1** | 贪心解码通过率，最核心指标 |
| **pass@10** | 采样 10 次的上界，反映模型潜力 |

Benchmark：

- **HumanEval(+)**：基础代码能力
- **LiveCodeBench**：时效性强，防数据泄露污染
- **MBPP**：覆盖简单实用场景

---

## 总计资源消耗

| 阶段 | 墙上耗时 | GPU 数 | GPU 小时 |
|------|----------|--------|----------|
| SFT | 0.5h | 1 | 0.5 |
| GRPO | 1.0h | 4 | 4.0 |
| 评测调试 | 0.5h | 4 | 2.0 |
| **合计** | **2.0h** | — | **~6.5 GPU 小时** |

在 normal 分区每账户 96 GPU 并发限制内，可分多次 sbatch 提交。

---

## 面试技术亮点

### 1. 为什么两阶段而不是直接 GRPO？

冷启动问题。从 base 模型直接做 GRPO，模型不会生成 `<think>` 格式，reward 信号极度稀疏，训练几乎不收敛。SFT 阶段先注入推理格式，GRPO 才能有效优化。

### 2. GRPO 相比 PPO 的优势？

PPO 需要独立的 critic 网络估计 value function，对 7B 模型来说相当于同时维护两个大模型，显存压力翻倍。GRPO 用同一道题的 N 个采样回答做组内归一化，用相对奖励代替绝对 value，省去 critic，显存更友好，且对 reward 量纲不敏感。

### 3. Reward 怎么设计的，有没有遇到 Reward Hacking？

用代码执行结果作为 verifiable reward，天然避免了 reward model 偏差问题。主要防范两类 hacking：一是死循环（超时截断），二是 hardcode 输出（检测 print 直接输出答案的行为）。

### 4. KL 散度的作用？

KL 惩罚项约束 GRPO 训练后的策略不要偏离 SFT 初始化太远，防止模型为了拿高 reward 退化成不自然的输出。`kl_coef=0.001` 是较小的值，允许模型有一定自由度探索。

### 5. 训练过程中观察到了什么现象？

随着训练轮次增加：
- `<think>` 部分内容质量提升，出现更多结构化分析
- Response 长度先增后趋于稳定
- pass@1 稳步提升，pass@10 提升更快（说明模型学到了正确方向但贪心解码还有提升空间）

---

## 项目价值总结

这个项目完整覆盖了当前 LLM 后训练的核心技术链路：**数据构造 → SFT 冷启动 → GRPO 强化 → 可量化评测**，使用的技术栈（verl、GRPO、verifiable reward）与 DeepSeek-R1 的训练范式高度一致。

</details>
