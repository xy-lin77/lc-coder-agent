# PPO

## 1. 四个模型
- **Actor**：基于 SFT 训练后的模型，待优化的目标模型
- **Critic**：基于 SFT 模型，逻辑复杂，暂略
- **Reference**：冻结 SFT 训练完成的模型
- **Reward Model (RM)**：依托 SFT 模型，在最后一个 Token 的隐层状态上拼接 Value 头并微调
  - Loss: $$L = -\log(r(x,y_{winner}) - r(x,y_{loser}))$$

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
- 优势：RM 无需与 Actor 同规模，显存压力大幅降低（如 DeepSeek-R1 实践）

---

## 3. 交互流程
1. **Actor 生成回复**  
 $$y \sim \pi_\theta(y|x)$$

2. **Reward Model 给回复打分**  
 输出 $$r(x, y)$$

3. **Reference 计算 KL 惩罚**
 - 惩罚项： $$\beta \cdot KL[\pi_\theta ||\pi_{ref}]$$
 - 最终 reward： $$r(x, y) - \beta \cdot KL$$

4. **Critic 估计每个 token 位置的 value**
 - 输出 $$V(s_t)$$
 - 用 GAE 计算 Advantage： $$A_t = r + \gamma V(s_{t+1}) - V(s_t)$$

5. **PPO-Clip 更新 Actor**
 - 损失： $$L = \min(ratio \cdot A, \text{clip}(ratio, 1\pm\varepsilon) \cdot A)$$
 - 其中 $$ratio = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$$

6. **同时更新 Critic**
 - 用 MSE loss 拟合 value 估计值

> 注：原生流程显存压力极大，需同时在 GPU 上维护 4 个大模型（Actor / Reference / Reward / Critic），工程中通常采用上述显存优化策略。

---

# DPO

## 1. 两个模型
- **Policy**：基于 SFT 训练后的模型，待优化的目标模型（对应PPO Actor）
- **Reference**：同 PPO
- **剔除组件**：完全移除 Reward Model、Critic 两大模型，无价值估计、优势函数计算

---

## 2. 微调方式
### 2.1 学术标准做法（原生DPO）
- **全量微调**：Policy全量更新，Reference全程冻结
- 优势：无需训练奖励模型，端到端直接优化偏好，流程极简

### 2.2 工程实践（显存极致优化）
#### 策略 1：Policy用LoRA微调，Reference冻结
- 仅训练Policy的LoRA适配器，主干权重冻结
- 显存占用：仅需1份完整模型权重+小体积LoRA，相比PPO降低75%

#### 策略 2：单卡适配方案
- 推理/训练共享计算图，Reference仅前向计算无梯度
- 适配7B/13B模型单卡微调，无需ZeRO3分布式

#### 策略 3：合并推理加速
- 训练完成后将LoRA权重合并至Policy，直接部署，无额外推理开销

---

## 3. 交互流程（无强化学习循环，一步训练）

1. **输入构造**：给定指令 $x$，采样一对回复 $(y_w, y_l)$，其中 $y_w$ 为偏好优胜回复，$y_l$ 为劣等回复。

2. **双模型前向计算**
   - **Policy**：计算对数概率 $\log \pi_\theta(y_w \mid x)$、$\log \pi_\theta(y_l \mid x)$
   - **Reference**：计算对数概率 $\log \pi_{\text{ref}}(y_w \mid x)$、$\log \pi_{\text{ref}}(y_l \mid x)$

3. **核心偏好损失计算（无 KL 惩罚显式计算）**

   损失函数为：

   $$
   L_{\text{DPO}}
   =
   -\mathbb{E}_{(x,y_w,y_l)}
   \left[
   \log \sigma
   \left(
   \beta
   \left(
   \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
   -
   \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
   \right)
   \right)
   \right]
   $$

   其中 $\beta$ 为温度系数，用于平衡参考模型约束。

4. **参数更新**
   - 直接反向传播更新 **Policy**
   - **Reference** 全程冻结
   - 无 PPO 中的 Clip、GAE、多模型交替更新流程

> 注：相比 PPO 四模型并行，DPO 仅需 2 个模型且无 RL 循环，训练速度更快，显存消耗更低，是工业界常用的偏好优化方案。
