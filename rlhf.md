# PPO 笔记

1. Actor：基于 SFT 训练后的模型。
2. Critic：同样以 SFT 模型为基底，内部逻辑相对复杂，暂不展开。
3. Reference：采用 SFT 阶段训练完成的模型。
4. Reward：依托 SFT 模型实现，在最后一个 Token 的隐层状态上拼接 Value 头并做微调；

损失公式：\(L = -\log(r(x,y_{winner}) - r(x,y_{loser}))\)。
