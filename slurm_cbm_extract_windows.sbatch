#!/bin/bash
#
# SBATCH script to:
#  1) start a vLLM server (Qwen/Qwen3-VL-8B-Thinking) in a vllm env
#  2) run CBM concept extraction pointing at that local server
#
# Usage:
#   sbatch slurm_cbm_extract_windows.sbatch
#
# Notes:
# - Edit the SBATCH resources, conda env names, and paths as needed for your cluster.
# - vLLM runs on the same allocated node; the extractor hits http://127.0.0.1:8001/v1
#

#SBATCH --job-name=cbm_hmdb_concept_extraction
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu-vram-48gb
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p ./slurm_logs

############################
# User-configurable settings
############################

# Conda env namesa
VLLM_ENV="vllm"
EXTRACT_ENV="sam3"  # Environment for running concept extraction

# vLLM server settings (from the notebook cell 12-19)
VLLM_MODEL="Qwen/Qwen3-VL-8B-Thinking"
VLLM_PORT="8001"
VLLM_HOST="0.0.0.0"

# Extraction command settings
# Update these paths to match your setup
DATASET_NAME="hmdb"  # set to hmdb, ssv2, ucf101, breakfast, ...
IMAGE_DATA_DIR="./Datasets/HMDB/Image_data" #change name for other datasets
OUTPUT_ROOT="./concept_extraction_out_batch/HMDB/8B" #change name for other datasets
INSTANCES_PER_CLASS="5"
SEED="0"
WINDOW_SIZE=30

EXTRACT_SCRIPT="./utils/extract_cbm_concepts_from_image_data_windows.py"

# Optional: restrict to a single class (set empty for all)
CLASS_FILTER=""

# OpenAI-compatible client requires a key string even for local vLLM
export OPENAI_API_KEY="EMPTY"

############################
# Helpers
############################

log() { echo "[$(date '+%F %T')] $*"; }

cleanup() {
  set +e
  if [[ -n "${VLLM_PID:-}" ]]; then
    log "Stopping vLLM (pid=${VLLM_PID})"
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

############################
# Load conda
############################

# Adjust this if your cluster uses a different conda init path.
if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi
# Explicit conda init for non-interactive shell (adjust path if your conda lives elsewhere).
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  log "ERROR: conda.sh not found; update path in slurm script."
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  log "ERROR: conda not found in PATH. Edit this script to initialize conda for your cluster."
  exit 2
fi

############################
# Start vLLM server
############################

log "Activating conda env: ${VLLM_ENV}"
conda activate "${VLLM_ENV}"

VLLM_LOG="./slurm_logs/vllm-${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-nojob}.log"
log "Starting vLLM server -> ${VLLM_LOG}"

vllm serve "${VLLM_MODEL}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --allowed-local-media-path / \
  --enforce-eager \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  >"${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
log "vLLM pid=${VLLM_PID}"

conda deactivate || true

log "Waiting for vLLM to become ready on http://127.0.0.1:${VLLM_PORT}/v1/models"
READY=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "${READY}" -ne 1 ]]; then
  log "ERROR: vLLM did not become ready in time. Check ${VLLM_LOG}"
  exit 3
fi
log "vLLM is ready."

############################
# Run extraction
############################

log "Activating conda env: ${EXTRACT_ENV}"
conda activate "${EXTRACT_ENV}"

EXTRACT_LOG="./slurm_logs/extract-${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-nojob}.log"
log "Running extraction -> ${EXTRACT_LOG}"

CMD=(python3 "${EXTRACT_SCRIPT}"
  --image_data_dir "${IMAGE_DATA_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --instances_per_class "${INSTANCES_PER_CLASS}"
  --shuffle_instances --seed "${SEED}"
  --dataset_name "${DATASET_NAME}"
  --llm_server_url "http://127.0.0.1:${VLLM_PORT}/v1" --llm_model "${VLLM_MODEL}" --llm_api_key "EMPTY" --llm_max_tokens 3500
  --window_size "${WINDOW_SIZE}" --max_windows 8 --max_window_overlap 0.25
  --max_concepts 6
)

if [[ -n "${CLASS_FILTER}" ]]; then
  CMD+=(--class_filter "${CLASS_FILTER}")
fi

{
  log "Command: ${CMD[*]}"
  "${CMD[@]}"
} >>"${EXTRACT_LOG}" 2>&1

log "Extraction finished."

conda deactivate || true

log "DONE"


