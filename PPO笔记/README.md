# PPO 原理深入 + GAE 详解

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

Critic 是在 LLM 主干上加一个 **value head**：

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

### Critic 的训练目标

Critic 要让自己的预测 `V(s_t)` 满足 Bellman 方程，loss 是：

```
L_critic = E_t[ (V(s_t) - R_t)² ]
```

其中 `R_t` 是实际观测到的 return（从 t 到序列末尾的折扣累积 reward）。这是个**回归问题**，Critic 用 MSE loss 拟合实际 return。

### 从 V 到 Advantage：GAE 计算过程

Advantage **不是 Critic 直接输出的**，而是事后计算的：

```python
# 伪代码：一条轨迹结束后
rewards = [0, 0, 0, ..., 0, R_final]     # 只有最后一步有 reward
values  = critic(s_0, s_1, ..., s_T)     # Critic 预测每步的 V

# 第一步：计算每步的 TD error
deltas = []
for t in range(T):
    delta_t = rewards[t] + γ * values[t+1] - values[t]
    deltas.append(delta_t)

# 第二步：GAE 向前累积
advantages = []
gae = 0
for t in reversed(range(T)):
    gae = deltas[t] + γ * λ * gae        # 指数加权累积
    advantages.insert(0, gae)
```

最终每个 token `t` 都有一个 `A_t`，这是一个**纯数值**，不是模型输出。

---

## 三、GAE（Generalized Advantage Estimation）

### 问题：Advantage 怎么估计？

只有 Critic 预测的 V，没有模型直接预测 Q，所以需要用 V 间接估计 A。有两种极端方法：

**方法一：单步 TD（高 bias，低 variance）**

```
A_t ≈ δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
```

只看一步，完全依赖 V 是否准确。V 有偏差则 A 有偏差（bias 大）。

**方法二：Monte Carlo（低 bias，高 variance）**

```
A_t ≈ G_t - V(s_t)
     = (r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T) - V(s_t)
```

用真实完整轨迹，不依赖 V 的准确性（bias 小）。但轨迹本身随机性大，梯度不稳定（variance 大）。

### GAE：在两者之间插值

把未来多步的 TD error 指数加权累加，λ 控制看多远：

```
A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ... + (γλ)^{T-t}·δ_T
```

### λ 的作用

```
λ = 0：A_t = δ_t                 ← 退化为单步 TD，高 bias 低 variance
λ = 1：A_t = G_t - V(s_t)        ← 退化为 Monte Carlo，低 bias 高 variance
0 < λ < 1：bias 和 variance 都适中，实践中通常取 λ = 0.95
```

### 为什么权重指数衰减

越远的 TD error，受 V 预测误差累积的影响越大，可信度越低，所以权重指数衰减是合理的：

```
δ_t     权重 = 1          （最近，最可信）
δ_{t+1} 权重 = γλ
δ_{t+2} 权重 = (γλ)²
δ_{t+3} 权重 = (γλ)³     （越远，越不可信）
```

---

## 四、Actor 如何被更新

### Actor 的输出

```
输入：s_t = (prompt + y_1 + ... + y_{t-1})
         ↓
   Transformer 前向传播
         ↓
   logits：维度 = vocab_size（如 128000）
         ↓
   softmax → 每个 token 的概率分布 π_θ(·|s_t)
         ↓
   采样得到 y_t，记录 π_θ(y_t|s_t)
```

### PPO Loss 推导

最朴素的想法是直接梯度上升：

```
L_naive = E_t[ π_θ(y_t|s_t) · A_t ]
```

但更新幅度不受控制，策略容易崩掉。PPO 引入 **ratio** 衡量新旧策略差距：

```
ratio_t = π_θ(y_t|s_t) / π_θ_old(y_t|s_t)
```

然后用 clip 限制 ratio 范围，防止单次更新过大（ε 通常取 0.2）：

```
L_CLIP = E_t[ min(
    ratio_t · A_t,
    clip(ratio_t, 1-ε, 1+ε) · A_t
) ]
```

### Clip 的直觉

```
A_t > 0（好 token）：
  ratio < 1+ε → 正常按比例给 reward（鼓励增大概率）
  ratio > 1+ε → 截断，不再给额外 reward（防止更新过猛）

A_t < 0（坏 token）：
  ratio > 1-ε → 正常按比例惩罚
  ratio < 1-ε → 截断，不再额外惩罚（防止过度抑制）
```

**本质**：好事不能做过头，坏事不要惩罚过度，每次更新控制在安全范围内。

### KL 惩罚

PPO 在 LLM 后训练中还加了 KL 约束，防止模型偏离 Reference 太远：

```
L_total = L_CLIP - β · KL[π_θ || π_ref]
```

---

## 五、权重更新的完整过程

**logit 变了，模型权重动了。**

```
1. 前向传播（Actor）
   输入 s_t → Transformer → logits → softmax → π_θ(y_t|s_t)
   同时记录 π_θ_old(y_t|s_t)（采样时存下来，固定不变）

2. 计算 loss
   ratio = π_θ / π_θ_old
   L_CLIP = min(ratio·A_t, clip(ratio, 0.8, 1.2)·A_t)
   loss = -mean(L_CLIP)   ← 负号因为要梯度上升

3. 反向传播
   loss.backward()
   梯度流经 softmax → logits → Transformer 所有层

4. 优化器更新权重
   optimizer.step()   ← Actor 的权重（W_q, W_k, W_v 等）全部更新
```

### logit 如何变化

```
A_t > 0（好 token），且 ratio < 1+ε 时：
  loss 对 logit[y_t] 的梯度 < 0
  optimizer 让 logit[y_t] 增大
  → softmax 后 π(y_t|s_t) 增大
  → 下次更容易采样到 y_t

A_t < 0（坏 token），且 ratio > 1-ε 时：
  loss 对 logit[y_t] 的梯度 > 0
  optimizer 让 logit[y_t] 减小
  → softmax 后 π(y_t|s_t) 减小
  → 下次更不容易采样到 y_t
```

注意：logit 是 vocab_size 维的向量，softmax 归一化，**改变一个 token 的 logit，其他 token 的概率也会相应变化**。

---

## 六、完整数据流总结

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
    计算 L_CLIP
    反向传播 → 更新 Actor 所有 Transformer 层权重

  Critic：
    重新前向传播，得到新的 V(s_t)
    计算 L_critic = (V(s_t) - R_t)²
    反向传播 → 更新 Critic 所有层权重（含 value head）

【循环】
  用更新后的 Actor 重新采样，开始下一轮
```

Critic 负责**看懂局面**（估计 V），Actor 负责**改进决策**（更新 logit），两者交替训练，共同收敛。
