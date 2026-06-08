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
