#!/bin/bash
# =============================================================================
# Data Preparation Job — download, generate, and preprocess training data
# CPU-only; no GPUs required.
# =============================================================================
#SBATCH --job-name=lc-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/data_prep_%j.out

# ---------------------------------------------------------------------------
# Fail fast on any unhandled error
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Job diagnostics
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Data Preparation Job"
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_NODELIST}"
echo "Start time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname     : $(hostname)"
echo "============================================================"

START_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# Environment setup (no GPU)
# ---------------------------------------------------------------------------
source ~/.bashrc
conda activate lc-trainer

BASE_DIR=/workspace/lc-coder-agent
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"
export HF_HOME=/data/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "${BASE_DIR}"
mkdir -p logs

# ---------------------------------------------------------------------------
# Helper: print dataset statistics for a given directory
# ---------------------------------------------------------------------------
print_dataset_stats() {
    local label="$1"
    local dir="$2"
    echo ""
    echo "--- Dataset stats: ${label} ---"
    if [ -d "${dir}" ]; then
        echo "  Directory : ${dir}"
        echo "  Size      : $(du -sh "${dir}" 2>/dev/null | cut -f1)"
        echo "  Files     : $(find "${dir}" -type f | wc -l)"
        # Count JSON/JSONL lines if present
        local jsonl_count
        jsonl_count=$(find "${dir}" -name '*.jsonl' -exec wc -l {} + 2>/dev/null \
                      | tail -1 | awk '{print $1}' || echo "0")
        if [ "${jsonl_count}" -gt 0 ]; then
            echo "  JSONL rows: ${jsonl_count}"
        fi
    else
        echo "  [WARN] Directory not found: ${dir}"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Step 1: Download raw datasets
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[STEP 1/4] Downloading datasets — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python data/download_datasets.py \
    --output-dir "${BASE_DIR}/data/raw" \
    --hf-cache "${HF_HOME}"

print_dataset_stats "raw" "${BASE_DIR}/data/raw"

# ---------------------------------------------------------------------------
# Step 2: Generate SFT data via API
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[STEP 2/4] Generating SFT data — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Load API keys from project .env (must exist; abort otherwise)
ENV_FILE="${BASE_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    echo "[ERROR] .env file not found at ${ENV_FILE}" >&2
    echo "[ERROR] Create it with OPENAI_API_KEY=<key> before running data prep." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${ENV_FILE}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY is not set in ${ENV_FILE}" >&2
    exit 1
fi

echo "[INFO] OPENAI_API_KEY loaded (length: ${#OPENAI_API_KEY})"

python data/generate_sft_data.py \
    --raw-data-dir   "${BASE_DIR}/data/raw" \
    --output-dir     "${BASE_DIR}/data/sft_raw" \
    --num-workers    "${SLURM_CPUS_PER_TASK}"

print_dataset_stats "sft_raw" "${BASE_DIR}/data/sft_raw"

# ---------------------------------------------------------------------------
# Step 3: Preprocess SFT data (tokenise, format, split)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[STEP 3/4] Preprocessing SFT data — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python data/preprocess_sft.py \
    --input-dir  "${BASE_DIR}/data/sft_raw" \
    --output-dir "${BASE_DIR}/data/sft_processed" \
    --tokenizer  /data/models/Qwen2.5-7B-Instruct \
    --num-workers "${SLURM_CPUS_PER_TASK}"

print_dataset_stats "sft_processed" "${BASE_DIR}/data/sft_processed"

# ---------------------------------------------------------------------------
# Step 4: Preprocess GRPO data (reward-model-compatible format)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[STEP 4/4] Preprocessing GRPO data — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python data/preprocess_grpo.py \
    --input-dir  "${BASE_DIR}/data/raw" \
    --output-dir "${BASE_DIR}/data/grpo_processed" \
    --tokenizer  /data/models/Qwen2.5-7B-Instruct \
    --num-workers "${SLURM_CPUS_PER_TASK}"

print_dataset_stats "grpo_processed" "${BASE_DIR}/data/grpo_processed"

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  DATA PREPARATION COMPLETE"
echo "============================================================"
echo ""
echo "Output locations:"
echo "  Raw data         : ${BASE_DIR}/data/raw"
echo "  SFT raw          : ${BASE_DIR}/data/sft_raw"
echo "  SFT processed    : ${BASE_DIR}/data/sft_processed"
echo "  GRPO processed   : ${BASE_DIR}/data/grpo_processed"
echo ""

# ---------------------------------------------------------------------------
# Elapsed time
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo "------------------------------------------------------------"
echo "  Data prep finished — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapsed : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "------------------------------------------------------------"

exit 0
