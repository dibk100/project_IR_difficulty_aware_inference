#!/usr/bin/env bash
# run_all.sh — phase01 파이프라인 일괄 실행
#
#   1) run_rollouts.py        : Easy/Hard 라벨 생성  -> gsm8k_rollouts.jsonl
#   2) extract_hidden_states.py: hidden state 추출   -> gsm8k_hidden_states.npz
#   3) visualize_embeddings.py : PCA / t-SNE 시각화
#
# 어느 단계든 실패하면 즉시 중단된다.
#
# Usage:
#   bash run_all.sh
#   ./run_all.sh           # (chmod +x run_all.sh 후)

set -euo pipefail

# 스크립트 위치로 이동 (상대경로 data/, *.jsonl 기준을 맞추기 위함)
cd "$(dirname "$0")"

# python 실행 명령 (필요시 PYTHON=python3 bash run_all.sh 로 오버라이드)
PY="${PYTHON:-python}"

run_step () {
    local step_no="$1"
    local script="$2"
    echo ""
    echo "=================================================================="
    echo "[STEP ${step_no}] ${script}"
    echo "=================================================================="
    "$PY" "$script"
    echo "[STEP ${step_no}] DONE: ${script}"
}

start_ts=$(date +%s)

run_step 1 run_rollouts.py
run_step 2 extract_hidden_states.py
run_step 3 visualize_embeddings.py

elapsed=$(( $(date +%s) - start_ts ))
echo ""
echo "=================================================================="
echo "All steps completed in ${elapsed}s."
echo "=================================================================="
