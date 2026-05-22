#!/bin/bash
# =============================================================================
# GRPO Training Job — verl + Qwen2.5-7B-Instruct on SuperPOD (H800 x4)
# =============================================================================
#SBATCH --job-name=lc-grpo
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H800:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err
#SBATCH --dependency=singleton
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_MAIL}

# ---------------------------------------------------------------------------
# Job diagnostics
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  GRPO Training Job"
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_NODELIST}"
echo "Start time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname     : $(hostname)"
echo "============================================================"

START_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# Ensure Ray is stopped on exit (success or failure)
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[INFO] Stopping Ray cluster — $(date '+%Y-%m-%d %H:%M:%S')"
    ray stop --force 2>/dev/null || true
    echo "[INFO] Ray stopped."
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source ~/.bashrc
conda activate lc-trainer

BASE_DIR=/workspace/lc-coder-agent
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"
export HF_HOME=/data/hf_cache
export WANDB_MODE=online

# NCCL
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# verl / vllm
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "${BASE_DIR}"
mkdir -p logs

# ---------------------------------------------------------------------------
# GPU sanity check
# ---------------------------------------------------------------------------
echo ""
echo "[INFO] Checking GPU visibility..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ---------------------------------------------------------------------------
# Prerequisite: SFT merged checkpoint must exist
# ---------------------------------------------------------------------------
SFT_CKPT="${BASE_DIR}/checkpoints/sft/merged"

echo "------------------------------------------------------------"
echo "[CHECK] Verifying SFT merged checkpoint at: ${SFT_CKPT}"
echo "------------------------------------------------------------"

if [ ! -d "${SFT_CKPT}" ]; then
    echo "[ERROR] SFT merged checkpoint not found at ${SFT_CKPT}" >&2
    echo "[ERROR] Please run sft.slurm first and ensure the LoRA merge step completes." >&2
    exit 1
fi

# Verify the directory is non-empty and contains model weights
if [ -z "$(ls -A "${SFT_CKPT}" 2>/dev/null)" ]; then
    echo "[ERROR] SFT merged checkpoint directory exists but is empty: ${SFT_CKPT}" >&2
    exit 1
fi

echo "[INFO] SFT checkpoint found: ${SFT_CKPT}"
ls -lh "${SFT_CKPT}" | head -20
echo ""

# ---------------------------------------------------------------------------
# Step 1: Start Ray local cluster
# ---------------------------------------------------------------------------
echo "------------------------------------------------------------"
echo "[STEP 1] Starting Ray head node — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  CPUs : 64"
echo "  GPUs : 4"
echo "------------------------------------------------------------"

ray start \
    --head \
    --num-cpus=64 \
    --num-gpus=4 \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats \
    2>&1

RAY_EXIT=$?
if [ "${RAY_EXIT}" -ne 0 ]; then
    echo "[ERROR] Failed to start Ray cluster (exit code ${RAY_EXIT})" >&2
    exit "${RAY_EXIT}"
fi

# Give Ray a moment to fully initialise
sleep 5

echo "[INFO] Ray cluster started successfully."
ray status

# ---------------------------------------------------------------------------
# Step 2: GRPO Training
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[STEP 2] Starting GRPO training — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Config      : configs/grpo/grpo_verl.yaml"
echo "  SFT ckpt    : ${SFT_CKPT}"
echo "  GPUs        : 4 x H800"
echo "------------------------------------------------------------"

python train/grpo/run_grpo.py \
    --config configs/grpo/grpo_verl.yaml \
    --num-gpus 4 \
    --sft-checkpoint "${SFT_CKPT}"

GRPO_EXIT=$?
if [ "${GRPO_EXIT}" -ne 0 ]; then
    echo "[ERROR] GRPO training failed with exit code ${GRPO_EXIT}" >&2
    exit "${GRPO_EXIT}"
fi

echo "[INFO] GRPO training completed successfully — $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------------------
# Elapsed time (cleanup/Ray stop handled by trap)
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  GRPO job finished — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapsed : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================================"

exit 0
