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

# 蒙特卡洛（MC）

- 分类：value-based，on-policy
- 原理：从完整回合中获取每一步的回报 $G_t$，以所有访问该状态时 $G_t$ 的均值估计 $V(s)$，无 Bellman 递推
- 直觉：下完一整盘棋，赢了给所有经过的位置记 +1，输了记 -1，下 100 盘后取平均。只靠完整棋局统计，下到一半不能更新。

---

# 时序差分学习（TD）

- 分类：value-based，on-policy
- 原理：结合 MC 采样与 DP 递推，每步更新一次

$$V(s_t) \leftarrow V(s_t) + \alpha\bigl[r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\bigr]$$

- 直觉：不用下完整盘，走一步就用"这步奖励 + 下一位置的价值"修正当前位置的估计，比 MC 更高效。

---

# SARSA

- 分类：value-based，on-policy
- 原理：学习动作价值函数 $Q(s,a)$，更新依赖五元组 $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl[r + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\bigr]$$

- 直觉：按"保守策略"走了 $a$，得到奖励，再按同一策略走 $a'$，只用这组真实走法更新。想换策略就必须重新收集数据。

---

# Q-learning

- 分类：value-based，off-policy
- 原理：直接学最优 $Q^*(s,a)$，更新时取下一状态所有动作的最大值

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl[r + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t)\bigr]$$

- 直觉：可以看任何人的下棋录像，学习时不管别人怎么走，只关心下一步最优走法的价值，数据来源灵活。

---

# DQN（深度 Q 网络）

- 分类：value-based，off-policy
- 原理：用神经网络拟合 $Q(s,a)$，引入经验回放与目标网络稳定训练

$$L = \mathbb{E}\bigl[(r + \gamma \max_{a'} Q'(s_{t+1},a') - Q(s_t,a_t))^2\bigr]$$

- 直觉：把棋盘画面输入网络，输出每个落子位置的 Q 值。存下所有对局经验，随机抽取训练，靠网络泛化没见过的棋局。

---

# Policy Gradient（策略梯度）

- 分类：policy-based，on-policy
- 原理：直接优化策略 $\pi_\theta(a|s)$，最大化期望回报

$$\nabla J(\theta) = \mathbb{E}\bigl[\nabla \log \pi_\theta(a|s) \cdot G_t\bigr]$$

- 直觉：直接输出落子概率分布，赢了就把这盘所有走法的概率调高，输了调低，不需要先估价值。

---

# REINFORCE

- 分类：policy-based，on-policy
- 原理：蒙特卡洛策略梯度，完整回合结束后用 $G_t$ 更新

$$\theta \leftarrow \theta + \alpha \nabla \log \pi_\theta(a_t|s_t) \cdot G_t$$

- 直觉：下完整盘后回溯每一步，赢了就提高每步选择的概率。实现最简单，但方差较大。

---

# Actor-Critic（AC）

- 分类：actor-critic 混合，on-policy
- 原理：Actor 输出动作概率，Critic 用 TD 误差 $\delta = r + \gamma V(s') - V(s)$ 评价动作好坏，Actor 用 $\delta$ 替代 $G_t$ 更新
- 直觉：Actor 负责落子，Critic 站旁边每走一步就打分，Actor 根据打分调整概率。不用等下完整盘，比 REINFORCE 更稳。

---

# TRPO

- 分类：policy-based，on-policy
- 原理：用共轭梯度 + 线搜索保证单调策略提升，KL 散度约束新旧策略差异
- 直觉：更新时严格限制策略变化幅度，确保新模型效果不会突变，每步更新都能保证不降，适合大模型安全训练。

---

# PPO（近端策略优化）

- 分类：policy-based，on-policy（clip 版）
- 原理：简化 TRPO，用裁剪函数替代 KL 约束，可多次复用同一批数据

$$L(\theta) = \mathbb{E}\bigl[\min\bigl(r_t(\theta) \cdot A_t,\ \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t\bigr)\bigr]$$

- 直觉：对同一批对话数据反复小幅度更新模型，不让模型学得太激进，兼顾效果与稳定性，是 RLHF 最常用算法。

---

# GRPO（分组相对策略优化）

- 分类：policy-based，on-policy
- 原理：去掉 Critic，对同一 prompt 采样 $G$ 条回复，用组内 reward 归一化直接得到 Advantage

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

- 直觉：给模型同一个问题，生成多个回答，reward 高于组内均值的回答提升概率，低于均值的降低。无需 Critic，训练更轻量，适合数学、代码等答案差异大的任务。
