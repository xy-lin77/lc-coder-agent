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

## Slurm 作业提交

### 交互式调试

```bash
srun --account=mscbdtsuperpod \
     --partition=normal \
     --nodes=1 \
     --gpus-per-node=4 \
     --time=08:00:00 \
     --pty bash
```

### 正式训练（批处理）

```bash
# train.sh
#!/bin/bash
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

python -m verl.trainer.main_ppo ...
```

```bash
sbatch train.sh
```

---

## 总计资源消耗

| 阶段 | 时间 | GPU 数 | GPU 小时 |
|---|---|---|---|
| SFT | 6h | 4 | 24 |
| GRPO | 20h | 4 | 80 |
| 评测调试 | 约 10h | 4 | 40 |
| **合计** | | | **~144 GPU 小时** |

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
