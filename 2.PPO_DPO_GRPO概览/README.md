# PPO

## 1. 四个模型

| 模型 | 核心作用 | 额外改造 | 训练目标 & 损失函数 |
| ---- | -------- | -------- | ------------------- |
| Actor | SFT训练后的模型，待优化的目标生成模型 | 无 | 最大化动作优势，同时裁剪策略更新幅度，防止模型单次更新过大、训练崩塌。使用PPO-Clip损失 <br> $L = \mathbb{E}\bigl[\min\bigl(r_t A_t,\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t\bigr)\bigr]$ <br> ，其中 $r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}$ |
| Critic | SFT训练后的模型，预测从当前token位置到序列结束的未来累计总收益期望（ $V_t$ ），用于计算优势函数 | 在每一个Token的隐层状态后拼接Value预测头 | 拟合真实时序回报与GAE估值，使用MSE损失缩小预测 $V_t$ 与实际回报偏差 <br> $L = \mathbb{E}\big[(V_t - V_t^{target})^2\big]$ |
| Reference | 冻结的SFT训练模型，固定输出分布，用于计算KL惩罚项，约束Actor更新幅度 | 无 | 无 |
| Reward Model | 偏好打分模型，预测单条完整问答序列的全局偏好分数 $r(x,y)$，提供原始奖励信号 | 在序列最后一个Token的隐层状态上拼接Reward打分头 | 学习人类偏好排序，拉大优劣回复分数差距 <br> $L = -\log\big(r(x, y_{winner}) - r(x, y_{loser})\big)$ |

---

## 2. 工程实现

### 2.1 学术论文
#### 全量微调（如 InstructGPT、Llama 2）
1. Reward Model = SFT backbone 全量更新 + value head
2. Critic = SFT backbone 全量更新 + value head
3. 原因：Backbone 需要从“生成表示”转变为“评判表示”，全量微调效果最优

### 2.2 工业界
#### 显存妥协方案：PPO 权重共享 + ZeRO3 + LoRA
1. 权重复用：Actor 与 Reference 共享同一底座，仅通过开关 LoRA 适配器区分训练与推理状态；Critic 与 Reward Model 共享主干权重。显存常驻权重减少至2份。
2. LoRA 轻量化训练：Actor、Critic、RM 均冻结 SFT 主干，仅训练 LoRA 适配器与 Critic/RM 专属 Value 头，大幅降低可训练参数量。
3. DeepSpeed ZeRO3 分布式加持：结合权重分片、显存卸载、梯度分片能力，进一步分摊多卡显存压力。
4. 独立小奖励模型：采用小规格独立 RM（如 1.5B 小模型）替代与 Actor 同尺寸权重，仅负责偏好打分。

---

## 3. 交互流程

1. **Actor 生成回复**：固定旧策略，采样完整回复 $y \sim \pi_{\theta_{\text{old}}}(y \mid x)$

2. **Reward Model 给出全局偏好分数**：输出原始奖励 $r(x, y)$

3. **Reference 计算 KL 约束惩罚**：逐 token 叠加惩罚，得到单 token 最终奖励

$$r_t = r(x,y) - \beta \cdot \text{KL}(\pi_{\theta_{\text{old}}} \parallel \pi_{\text{ref}})$$

4. **Critic 逐 Token 预测状态价值 & 计算 GAE 优势**：输出 $V(s_t)$，回溯得到优势函数 $A_t$ 与价值目标 $V_t^{\text{target}}$

5. **同时更新 Actor 和 Critic 权重**

   Actor PPO-Clip 损失：

$$L_{\text{Actor}} = \min\left(\text{ratio} \cdot A,\ \text{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon) \cdot A\right), \quad \text{ratio} = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}$$

   Critic MSE 损失：

$$L_{\text{Critic}} = \mathbb{E}\big[(V_t - V_t^{\text{target}})^2\big]$$

6. **循环迭代，进入下一轮采样更新**

---

# DPO

## 1. 两个模型
- **Policy**：SFT训练后的模型，待优化的目标生成模型（对应 PPO Actor）
- **Reference**：同 PPO
- 完全移除 RM、Critic 两大模型，无价值估计、优势函数计算
---

## 2. 交互流程（无强化学习循环，一步训练）

1. **输入构造**：给定指令 `x`，采样一对回复 `(y_w, y_l)`，其中 `y_w` 为优胜回复，`y_l` 为劣等回复

2. **双模型前向计算**：Policy 和 Reference 分别计算 `y_w`、`y_l` 的对数概率

3. **核心偏好损失计算**：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right)\right]$$

   其中 $\beta$ 为温度系数，用于平衡参考模型约束

4. **参数更新**：直接反向传播更新 Policy，Reference 全程冻结，无 Clip、GAE、多模型交替更新

> 注：相比 PPO 四模型并行，DPO 仅需 2 个模型且无 RL 循环，训练速度更快，显存消耗更低，是工业界常用的偏好优化方案。

```
