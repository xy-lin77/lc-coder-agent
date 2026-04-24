# PPO

## 1. 四个模型
- **Actor**：基于 SFT 训练后的模型
- **Critic**：基于 SFT 模型，逻辑复杂，暂略；可与 Reward 共享 Backbone
- **Reference**：采用 SFT 阶段训练完成的模型
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
   $y \sim \pi_\theta(y|x)$
2. **Reward Model 给回复打分**
   输出 $r(x, y)$
3. **Reference 计算 KL 惩罚**
   - 惩罚项：$\beta \cdot KL[\pi_\theta \ || \ \pi_{ref}]$
   - 最终 reward：$r(x, y) - \beta \cdot KL$
4. **Critic 估计每个 token 位置的 value**
   - 输出 $V(s_t)$
   - 用 GAE 计算 Advantage：$A_t = r + \gamma V(s_{t+1}) - V(s_t)$
5. **PPO-Clip 更新 Actor**
   - 损失：$L = \min(ratio \cdot A, \text{clip}(ratio, 1\pm\varepsilon) \cdot A)$
   - 其中 $ratio = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$
6. **同时更新 Critic**
   - 用 MSE loss 拟合 value 估计值

> 注：原生流程显存压力极大，需同时在 GPU 上维护 4 个大模型（Actor / Reference / Reward / Critic），工程中通常采用上述显存优化策略。
