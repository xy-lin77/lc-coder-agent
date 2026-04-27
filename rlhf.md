# PPO

## 1. 四个模型
- **Actor**：基于 SFT 训练后的模型，待优化的目标模型
- **Critic**：基于 SFT 模型，逻辑复杂，暂略
- **Reference**：冻结 SFT 训练完成的模型
- **Reward Model (RM)**：依托 SFT 模型，在最后一个 Token 的隐层状态上拼接 Value 头并微调
  - Loss: L = -log(r(x, y_winner) - r(x, y_loser))

---

## 2. 微调方式
### 2.1 学术标准做法（论文实现）
- **全量微调**（如 InstructGPT、Llama 2）
  - Reward Model = SFT backbone 全量更新 + value head
  - Critic = SFT backbone 全量更新 + value head
  - 原因：Backbone 需要从“生成表示”转变为“评判表示”，全量微调效果最优

### 2.2 工程实践（显存妥协）
#### 策略 1：RM 用 LoRA，Critic 共享 RM
- RM：`SFT + LoRA`（训练）→ merge 成全量权重（推理冻结）
- Critic：复用 RM 权重，仅训练 value head + LoRA
- 优势：RM/Critic 共享 Backbone，显存减半

#### 策略 2：Actor & Reference 共享，Critic & Reward 共享
- 显存仅 2 份完整权重：
  1. Actor（训练中）+ Reference 的 LoRA Adapter（或临时 disable Adapter 算 KL）
  2. Critic/Reward 共享 Backbone（冻结）+ Critic 的 value head（训练中）
- 常见于 `TRL + DeepSpeed ZeRO3` 组合

#### 策略 3：Reward Model 小模型化
- Actor：7B SFT 模型（生成回复）
- RM：1.5B 独立模型（全量微调做评判）
- 优势：RM 无需与 Actor 同规模，显存压力大幅降低

---

## 3. 交互流程
1. **Actor 生成回复**  
   y ~ π_θ(y | x)

2. **Reward Model 给回复打分**  
   r(x, y)

3. **Reference 计算 KL 惩罚**
   - KL 惩罚项：β · KL(π_θ || π_ref)
   - 最终 reward：r(x, y) - β · KL

4. **Critic 估计每个 token 的 value**
   - V(s_t)
   - Advantage：A_t = r + γ V(s_{t+1}) - V(s_t)

5. **PPO-Clip 更新 Actor**
   - L = min(ratio · A, clip(ratio, 1±ε) · A)
   - ratio = π_θ(a|s) / π_{θ_old}(a|s)

6. **同时更新 Critic**
   - 用 MSE loss 拟合 value

> 注：原生流程显存压力极大，需同时维护 4 个大模型。

---

# DPO

## 1. 两个模型
- **Policy**：待优化模型（对应 PPO Actor）
- **Reference**：冻结模型
- **移除组件**：无 Reward Model、无 Critic

---

## 2. 微调方式
### 2.1 学术标准做法
- **全量微调**：Policy 更新，Reference 冻结
- 优势：端到端优化偏好，无需奖励模型

### 2.2 工程实践
#### 策略 1：LoRA
- 仅训练 Policy 的 LoRA
- 显存显著降低

#### 策略 2：单卡方案
- Reference 仅 forward，无梯度
- 支持单卡训练

#### 策略 3：推理合并
- LoRA 合并到主模型
- 推理无额外开销

---

## 3. 交互流程（一步训练）

1. **输入构造**  
   给定 x，采样 (y_w, y_l)

2. **前向计算**
   - Policy：
     log π_θ(y_w | x), log π_θ(y_l | x)
   - Reference：
     log π_ref(y_w | x), log π_ref(y_l | x)

3. **损失函数**
   L_DPO = -E[ log σ( β ( log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x)) ) ) ]

   其中 β 为温度系数

4. **参数更新**
   - 仅更新 Policy
   - Reference 冻结
   - 无 PPO 复杂流程

> 注：DPO 仅需 2 个模型，无 RL 循环，训练更快、显存更低。
