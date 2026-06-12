#!/usr/bin/env bash
# run_all.sh — phase01 파이프라인 일괄 실행
#
#   1) run_rollouts.py         : Easy/Hard 라벨 생성   -> gsm8k_main_rollouts.jsonl
#   2) extract_hidden_states.py: hidden state 추출      -> gsm8k_main_hidden_states.npz
#   3) visualize_embeddings.py : PCA / t-SNE 시각화      -> pca_*.png, tsne_*.png
#   4) linear_probe.py         : linear probe 정량 평가  -> (지표 출력)
#   5) length_matched_probe.py : 길이 매칭 probe (길이 교란 통제) -> (지표 출력)
#
# 모든 산출물(jsonl/npz/png)과 각 단계의 터미널 로그는 output/ 폴더에 저장된다.
# 어느 단계든 실패하면 즉시 중단된다.
#
# Usage:
#   bash run_all.sh
#   ./run_all.sh           # (chmod +x run_all.sh 후)

set -euo pipefail

# 스크립트 위치(=phase01 루트)로 기준을 고정
ROOT="$(cd "$(dirname "$0")" && pwd)"

# 모든 출력이 모이는 폴더
OUT="${ROOT}/output"
mkdir -p "$OUT"

# python 실행 명령 (필요시 PYTHON=python3 bash run_all.sh 로 오버라이드)
PY="${PYTHON:-python}"

# 파이프라인 전체 통합 로그
RUN_LOG="${OUT}/run_all.log"

run_step () {
    local step_no="$1"
    local script="$2"
    local log="${OUT}/$(printf '%02d' "$step_no")_${script%.py}.log"

    echo ""
    echo "=================================================================="
    echo "[STEP ${step_no}] ${script}   (log: ${log})"
    echo "=================================================================="

    # cwd 를 output/ 으로 두고 실행 → 스크립트의 상대경로 산출물이 output/ 에 생성됨.
    # stdout+stderr 를 화면에 보여주면서 단계별 로그 파일에도 저장(tee).
    # set -o pipefail 덕분에 python 이 실패하면 파이프라인도 실패로 처리됨.
    ( cd "$OUT" && "$PY" "${ROOT}/${script}" ) 2>&1 | tee "$log"

    echo "[STEP ${step_no}] DONE: ${script}"
}

start_ts=$(date +%s)

# 통합 로그에도 전체 출력을 함께 기록
{
    echo "##################################################################"
    echo "# run_all.sh started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# output dir: ${OUT}"
    echo "##################################################################"

    run_step 1 run_rollouts.py
    run_step 2 extract_hidden_states.py
    run_step 3 visualize_embeddings.py
    run_step 4 linear_probe.py
    run_step 5 length_matched_probe.py

    elapsed=$(( $(date +%s) - start_ts ))
    echo ""
    echo "=================================================================="
    echo "All steps completed in ${elapsed}s."
    echo "Outputs saved under: ${OUT}"
    echo "=================================================================="
} 2>&1 | tee "$RUN_LOG"