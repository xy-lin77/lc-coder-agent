#!/bin/bash
# =============================================================================
# Evaluation Job — HumanEval / MBPP / LiveCodeBench on SuperPOD (H800 x4)
# Array job: 3 tasks (0=base, 1=sft, 2=grpo)
# =============================================================================
#SBATCH --job-name=lc-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H800:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --array=0-2

# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------
MODEL_ARRAY=(
    "/data/models/Qwen2.5-7B-Instruct"          # 0 — base model
    "checkpoints/sft/merged"                      # 1 — after SFT
    "checkpoints/grpo/qwen2_5_7b_grpo/latest"    # 2 — after GRPO
)
MODEL_LABEL_ARRAY=("base" "sft" "grpo")

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_PATH="${MODEL_ARRAY[${TASK_ID}]}"
MODEL_LABEL="${MODEL_LABEL_ARRAY[${TASK_ID}]}"

# ---------------------------------------------------------------------------
# Job diagnostics
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Evaluation Job  (array task ${TASK_ID} / 2)"
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID} [array task ${SLURM_ARRAY_TASK_ID}]"
echo "Node list    : ${SLURM_NODELIST}"
echo "Start time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname     : $(hostname)"
echo "Model label  : ${MODEL_LABEL}"
echo "Model path   : ${MODEL_PATH}"
echo "============================================================"

START_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source ~/.bashrc
conda activate lc-trainer

BASE_DIR=/workspace/lc-coder-agent
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"
export HF_HOME=/data/hf_cache
export TOKENIZERS_PARALLELISM=false

cd "${BASE_DIR}"
mkdir -p logs

# Resolve relative checkpoint paths against BASE_DIR
if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="${BASE_DIR}/${MODEL_PATH}"
fi

# ---------------------------------------------------------------------------
# Skip if checkpoint not present (warn, don't fail the whole array)
# ---------------------------------------------------------------------------
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[WARN] Checkpoint not found for variant '${MODEL_LABEL}': ${MODEL_PATH}" >&2
    echo "[WARN] Skipping evaluation for task ${TASK_ID}." >&2
    exit 0
fi

echo "[INFO] Checkpoint verified: ${MODEL_PATH}"
echo ""

# ---------------------------------------------------------------------------
# GPU sanity check
# ---------------------------------------------------------------------------
echo "[INFO] GPU visibility:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ---------------------------------------------------------------------------
# Benchmark 1: HumanEval
# ---------------------------------------------------------------------------
echo "------------------------------------------------------------"
echo "[BENCH 1/3] HumanEval — ${MODEL_LABEL} — $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"

HUMANEVAL_OUT="${BASE_DIR}/eval/results/humaneval/${MODEL_LABEL}"
mkdir -p "${HUMANEVAL_OUT}"

python eval/run_humaneval.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${HUMANEVAL_OUT}/" \
    --num-samples-per-task 10 \
    --tensor-parallel-size 4

HE_EXIT=$?
if [ "${HE_EXIT}" -ne 0 ]; then
    echo "[ERROR] HumanEval failed for ${MODEL_LABEL} (exit ${HE_EXIT})" >&2
    exit "${HE_EXIT}"
fi
echo "[INFO] HumanEval done — $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------------------
# Benchmark 2: MBPP
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[BENCH 2/3] MBPP — ${MODEL_LABEL} — $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"

MBPP_OUT="${BASE_DIR}/eval/results/mbpp/${MODEL_LABEL}"
mkdir -p "${MBPP_OUT}"

python eval/run_mbpp.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${MBPP_OUT}/" \
    --tensor-parallel-size 4

MBPP_EXIT=$?
if [ "${MBPP_EXIT}" -ne 0 ]; then
    echo "[ERROR] MBPP failed for ${MODEL_LABEL} (exit ${MBPP_EXIT})" >&2
    exit "${MBPP_EXIT}"
fi
echo "[INFO] MBPP done — $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------------------
# Benchmark 3: LiveCodeBench
# ---------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
echo "[BENCH 3/3] LiveCodeBench — ${MODEL_LABEL} — $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"

LCB_OUT="${BASE_DIR}/eval/results/livecodebench/${MODEL_LABEL}"
mkdir -p "${LCB_OUT}"

python eval/run_livecodebench.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${LCB_OUT}/" \
    --tensor-parallel-size 4

LCB_EXIT=$?
if [ "${LCB_EXIT}" -ne 0 ]; then
    echo "[ERROR] LiveCodeBench failed for ${MODEL_LABEL} (exit ${LCB_EXIT})" >&2
    exit "${LCB_EXIT}"
fi
echo "[INFO] LiveCodeBench done — $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------------------
# Results summary (task 0 only — runs after all array tasks finish)
# ---------------------------------------------------------------------------
if [ "${TASK_ID}" -eq 0 ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "[SUMMARY] Waiting for all array tasks to complete..."
    echo "------------------------------------------------------------"

    # Wait for sibling tasks (tasks 1 and 2) to finish before summarising.
    # We poll for the presence of their output files as a lightweight signal.
    SUMMARY_WAIT_TIMEOUT=3600   # seconds (1 hour max wait)
    SUMMARY_POLL_INTERVAL=60    # seconds between polls
    SUMMARY_ELAPSED=0

    for OTHER_TASK in 1 2; do
        OTHER_LABEL="${MODEL_LABEL_ARRAY[${OTHER_TASK}]}"
        SENTINEL_FILE="${BASE_DIR}/eval/results/livecodebench/${OTHER_LABEL}/.done"

        echo "[INFO] Waiting for task ${OTHER_TASK} (${OTHER_LABEL}) sentinel: ${SENTINEL_FILE}"
        while [ ! -f "${SENTINEL_FILE}" ] && [ "${SUMMARY_ELAPSED}" -lt "${SUMMARY_WAIT_TIMEOUT}" ]; do
            sleep "${SUMMARY_POLL_INTERVAL}"
            SUMMARY_ELAPSED=$(( SUMMARY_ELAPSED + SUMMARY_POLL_INTERVAL ))
        done

        if [ ! -f "${SENTINEL_FILE}" ]; then
            echo "[WARN] Timed out waiting for task ${OTHER_TASK}; proceeding with available results."
        fi
    done

    echo ""
    echo "------------------------------------------------------------"
    echo "[SUMMARY] Generating final results table..."
    echo "------------------------------------------------------------"

    python eval/summarize_results.py \
        --results-dir "${BASE_DIR}/eval/results" \
        --output "${BASE_DIR}/eval/results/summary.json"

    echo ""
    echo "============================  FINAL SUMMARY  =============================="
    python -c "
import json, sys
try:
    data = json.load(open('${BASE_DIR}/eval/results/summary.json'))
    header = f\"{'Model':<10} {'HumanEval':>12} {'MBPP':>10} {'LCBench':>12}\"
    print(header)
    print('-' * len(header))
    for entry in data.get('results', []):
        print(f\"{entry.get('model','?'):<10} {entry.get('humaneval_pass@1','N/A'):>12} {entry.get('mbpp_pass@1','N/A'):>10} {entry.get('livecodebench_pass@1','N/A'):>12}\")
except Exception as e:
    print(f'Could not render summary table: {e}', file=sys.stderr)
"
    echo "==========================================================================="
fi

# Write sentinel file so task-0 knows this task is done
touch "${BASE_DIR}/eval/results/livecodebench/${MODEL_LABEL}/.done"

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
echo "  Eval job finished [${MODEL_LABEL}] — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapsed : ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================================"

exit 0
