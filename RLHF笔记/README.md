# PPO

## 1. 四个模型

| 模型 | 核心作用 | 额外改造 | 训练目标 & 损失函数 |
| ---- | -------- | -------- | ------------------- |
| Actor | SFT训练后的模型，待优化的目标生成模型 | 无 | 最大化动作优势，同时裁剪策略更新幅度，防止模型单次更新过大、训练崩塌。使用PPO-Clip损失 <br> $L = \mathbb{E}\bigl[\min\bigl(r_t A_t,\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t\bigr)\bigr]$ <br> ，其中 $r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}$ |
| Critic | SFT训练后的模型，预测从当前token位置到序列结束的未来累计总收益期望（ $V_t$ ），用于计算优势函数 | 在每一个Token的隐层状态后拼接Value预测头 | 拟合真实时序回报与GAE估值，使用MSE损失缩小预测 $V_t$ 与实际回报偏差 <br> $L = \mathbb{E}\big[(V_t - V_t^{target})^2\big]$ |
| Reference | 冻结的SFT训练模型，固定输出分布，用于计算KL惩罚项，约束Actor更新幅度 | 无 | 无 |
| Reward Model | 偏好打分模型，预测单条完整问答序列的全局偏好分数 $r(x,y)$，提供原始奖励信号 | 在序列最后一个Token的隐层状态上拼接Reward打分头 | 学习人类偏好排序，拉大优劣回复分数差距 <br> $L = -\log\big(r(x, y_{winner}) - r(x, y_{loser})\big)$ |

---

## 2. 微调方式

### 2.1 学术论文实现
#### 全量微调（如 InstructGPT、Llama 2）
1. Reward Model = SFT backbone 全量更新 + value head
2. Critic = SFT backbone 全量更新 + value head
3. 原因：Backbone 需要从“生成表示”转变为“评判表示”，全量微调效果最优

### 2.2 工程实践
#### 显存妥协方案：PPO 权重共享 + ZeRO3 + LoRA
1. 权重复用设计：Actor 与 Reference 共享同一 SFT 主干底座，仅通过开关 LoRA 适配器区分训练与推理状态；Critic 与 Reward Model 共享主干权重。显存常驻权重减少至2份。
2. LoRA 轻量化训练：Actor、Critic、RM 均冻结 SFT 主干，仅训练 LoRA 适配器与 Critic/RM 专属 Value 头，大幅降低可训练参数量，减少梯度、优化器显存开销，避免全量微调显存爆炸。
3. DeepSpeed ZeRO3 分布式加持：结合权重分片、显存卸载、梯度分片能力，进一步分摊多卡显存压力，完美适配 7B/13B/34B 主流大模型。
4. 独立小尺寸奖励模型：采用小规格独立 RM（如 1.5B 小模型）替代与 Actor 同尺寸权重，无需占用大模型显存资源，仅负责偏好打分，进一步降低整体硬件成本与显存负载。

---

## 3. 交互流程

1. **Actor 生成回复**  
   `y ~ pi_theta(y | x)`

2. **Reward Model 给回复打分**  
   `r(x, y)`

3. **Reference 计算 KL 惩罚**
   - 惩罚项：`beta * KL(pi_theta || pi_ref)`
   - 最终 reward：`r(x, y) - beta * KL`

4. **Critic 估计每个 token 位置的 value**
   - 输出：`V(s_t)`
   - 用 GAE 计算 Advantage：`A_t = r + gamma * V(s_{t+1}) - V(s_t)`

5. **PPO-Clip 更新 Actor**
   - 损失：`L = min(ratio * A, clip(ratio, 1 +/- epsilon) * A)`
   - 其中：`ratio = pi_theta(a | s) / pi_theta_old(a | s)`

6. **同时更新 Critic**
   - 用 MSE loss 拟合 value 估计值

> 注：原生流程显存压力极大，需同时在 GPU 上维护 4 个大模型（Actor / Reference / Reward / Critic），工程中通常采用上述显存优化策略。

---

# DPO

## 1. 两个模型
- **Policy**：基于 SFT 训练后的模型，待优化的目标模型（对应 PPO Actor）
- **Reference**：同 PPO
- **剔除组件**：完全移除 Reward Model、Critic 两大模型，无价值估计、优势函数计算

---

## 2. 微调方式

### 2.1 学术标准做法（原生 DPO）
- **全量微调**：Policy 全量更新，Reference 全程冻结
- 优势：无需训练奖励模型，端到端直接优化偏好，流程极简

### 2.2 工程实践（显存极致优化）

#### 策略 1：Policy 用 LoRA 微调，Reference 冻结
- 仅训练 Policy 的 LoRA 适配器，主干权重冻结
- 显存占用：仅需 1 份完整模型权重 + 小体积 LoRA，相比 PPO 明显降低

#### 策略 2：单卡适配方案
- 推理/训练共享计算图，Reference 仅前向计算无梯度
- 适配 7B/13B 模型单卡微调，无需 ZeRO3 分布式

#### 策略 3：合并推理加速
- 训练完成后将 LoRA 权重合并至 Policy，直接部署，无额外推理开销

---

## 3. 交互流程（无强化学习循环，一步训练）

1. **输入构造**  
   给定指令 `x`，采样一对回复 `(y_w, y_l)`，其中 `y_w` 为偏好优胜回复，`y_l` 为劣等回复。

2. **双模型前向计算**
   - Policy：计算 `log pi_theta(y_w | x)` 和 `log pi_theta(y_l | x)`
   - Reference：计算 `log pi_ref(y_w | x)` 和 `log pi_ref(y_l | x)`

3. **核心偏好损失计算**
   - 损失函数：

```text
L_DPO = -E[
  log sigma(
    beta * (
      log(pi_theta(y_w | x) / pi_ref(y_w | x))
      -
      log(pi_theta(y_l | x) / pi_ref(y_l | x))
    )
  )
]
````

* 其中 `beta` 为温度系数，用于平衡参考模型约束

4. **参数更新**

   * 直接反向传播更新 Policy
   * Reference 全程冻结
   * 无 PPO 中的 Clip、GAE、多模型交替更新流程

> 注：相比 PPO 四模型并行，DPO 仅需 2 个模型且无 RL 循环，训练速度更快，显存消耗更低，是工业界常用的偏好优化方案。

```
