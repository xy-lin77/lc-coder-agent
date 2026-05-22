#!/bin/bash
# =============================================================================
# SFT Training Job — LLaMA-Factory + Qwen2.5-7B-Instruct on SuperPOD (H800 x4)
# =============================================================================
#SBATCH --job-name=lc-sft
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H800:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_MAIL}

# ---------------------------------------------------------------------------
# Job diagnostics
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  SFT Training Job"
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_NODELIST}"
echo "Start time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname     : $(hostname)"
echo "============================================================"

START_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source ~/.bashrc
conda activate lc-trainer

# Project paths
BASE_DIR=/workspace/lc-coder-agent
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

# Hugging Face cache
export HF_HOME=/data/hf_cache

# Weights & Biases
export WANDB_MODE=online

# NCCL — optimised for NVLink topology on H800
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0

# Tokenizer parallelism warning suppression
export TOKENIZERS_PARALLELISM=false

cd "${BASE_DIR}"

# Ensure log directory exists
mkdir -p logs

# ---------------------------------------------------------------------------
# GPU / CUDA sanity check
# ---------------------------------------------------------------------------
echo ""
echo "[INFO] Checking GPU visibility..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ---------------------------------------------------------------------------
# Step 1: SFT Training
# ---------------------------------------------------------------------------
echo "------------------------------------------------------------"
echo "[STEP 1] Starting SFT training — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Config  : configs/sft/qwen2_5_7b_lora_sft.yaml"
echo "  GPUs    : 4 x H800"
echo "------------------------------------------------------------"

python train/sft/run_sft.py \
    --config configs/sft/qwen2_5_7b_lora_sft.yaml \
    --num-gpus 4

SFT_EXIT=$?
if [ "${SFT_EXIT}" -ne 0 ]; then
    echo "[ERROR] SFT training failed with exit code ${SFT_EXIT}" >&2
    exit "${SFT_EXIT}"
fi

echo "[INFO] SFT training completed successfully — $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------------------
# Step 2: LoRA merge
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[STEP 2] Merging LoRA adapter into base model..."
echo "  Output : checkpoints/sft/merged"
echo "------------------------------------------------------------"

MERGE_CONFIG=configs/sft/qwen2_5_7b_lora_sft.yaml
MERGED_OUTPUT="${BASE_DIR}/checkpoints/sft/merged"

mkdir -p "${MERGED_OUTPUT}"

llamafactory-cli export \
    --model_name_or_path /data/models/Qwen2.5-7B-Instruct \
    --adapter_name_or_path "${BASE_DIR}/checkpoints/sft/lora_adapter" \
    --template qwen \
    --finetuning_type lora \
    --export_dir "${MERGED_OUTPUT}" \
    --export_size 2 \
    --export_legacy_format false

MERGE_EXIT=$?
if [ "${MERGE_EXIT}" -ne 0 ]; then
    echo "[ERROR] LoRA merge failed with exit code ${MERGE_EXIT}" >&2
    exit "${MERGE_EXIT}"
fi

echo "[INFO] LoRA merge completed — merged model saved to: ${MERGED_OUTPUT}"

# ---------------------------------------------------------------------------
# Elapsed time
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  SFT job finished — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapsed : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================================"

exit 0
