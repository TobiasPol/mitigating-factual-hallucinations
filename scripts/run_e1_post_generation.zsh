#!/bin/zsh

set -euo pipefail

readonly REPOSITORY_ROOT="${0:A:h:h}"
cd "${REPOSITORY_ROOT}"

readonly STUDY_ROOT="artifacts/studies/qwen36-27b-mlx4-m4max48-v1"
readonly MODEL_ROOT="artifacts/models/qwen3.6-27b-mlx-4bit/c000ac2c2057d94be3fa931000c31723aac53282"
readonly GENERATION_SESSION="mfh-qwen-e1-generation"
readonly GENERATION_CHECKPOINT="${STUDY_ROOT}/checkpoints/E1-generation.json"
readonly GRADING_CHECKPOINT="${STUDY_ROOT}/checkpoints/E1-grading.json"
readonly OUTPUT_DIRECTORY="${STUDY_ROOT}/outputs/E1"
readonly GRADER_ATTEMPTS="${STUDY_ROOT}/work/E1/grader-attempts.jsonl"
readonly MAX_TRANSIENT_RESTARTS_WITHOUT_PROGRESS=5
readonly EXPECTED_SPLIT_MANIFEST="05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"
readonly EXPECTED_GRADER_MANIFEST="b3af3c847c3488d6228a47c205186caca06bca8de1cd00dd81f0b83ac73e1159"

readonly -a E1_INPUTS=(
  artifacts/splits/triviaqa-reviewed
  artifacts/graders/e1-frozen-v2
  configs/models/qwen3.6-27b-mlx-4bit.yaml
  "${MODEL_ROOT}"
  configs/models/qwen3.6-27b-mlx-4bit.snapshot.json
  "${STUDY_ROOT}/frozen/mlx-preflight.json"
  "${STUDY_ROOT}/work/E1"
  "${STUDY_ROOT}/runs/E1"
  "${STUDY_ROOT}/runs/E0"
)

checkpoint_counts() {
  local checkpoint="$1"
  local completed_key="$2"
  local expected_key="$3"
  python3 -c \
    'import json, sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value[sys.argv[2]], value[sys.argv[3]])' \
    "${checkpoint}" "${completed_key}" "${expected_key}"
}

last_attempt_is_transient() {
  python3 -c \
    'import json, sys; lines=[line for line in open(sys.argv[1], encoding="utf-8") if line.strip()]; print("true" if json.loads(lines[-1])["receipt"]["transient"] is True else "false")' \
    "${GRADER_ATTEMPTS}"
}

while tmux has-session -t "${GENERATION_SESSION}" 2>/dev/null; do
  sleep 30
done

if [[ ! -f "${GENERATION_CHECKPOINT}" ]]; then
  print -u2 -- "E1 generation ended without its external checkpoint"
  exit 1
fi

read -r generations_completed generations_expected <<< \
  "$(checkpoint_counts "${GENERATION_CHECKPOINT}" records_completed records_expected)"
if [[ "${generations_completed}" != "19800" || "${generations_expected}" != "19800" ]]; then
  print -u2 -- \
    "E1 generation stopped at ${generations_completed}/${generations_expected}; grading was not started"
  exit 1
fi

typeset -i transient_restarts_without_progress=0
while true; do
  typeset -i grades_before=0
  if [[ -f "${GRADING_CHECKPOINT}" ]]; then
    read -r grades_before _ <<< \
      "$(checkpoint_counts "${GRADING_CHECKPOINT}" grades_completed grades_expected)"
  fi

  grade_arguments=(
    "${E1_INPUTS[@]}"
    --expected-split-manifest-digest "${EXPECTED_SPLIT_MANIFEST}"
    --expected-grader-manifest-digest "${EXPECTED_GRADER_MANIFEST}"
    --checkpoint-file "${GRADING_CHECKPOINT}"
    --env-file .env
  )
  if [[ -f "${GRADING_CHECKPOINT}" ]]; then
    grade_arguments+=(--resume)
  fi

  UV_CACHE_DIR=/tmp/uv-cache-lock uv run --no-sync mfh grade-e1-openrouter \
    "${grade_arguments[@]}"

  read -r grades_completed grades_expected <<< \
    "$(checkpoint_counts "${GRADING_CHECKPOINT}" grades_completed grades_expected)"
  if [[ "${grades_completed}" == "4800" && "${grades_expected}" == "4800" ]]; then
    break
  fi

  if [[ ! -f "${GRADER_ATTEMPTS}" || "$(last_attempt_is_transient)" != "true" ]]; then
    print -u2 -- \
      "E1 grading stopped at ${grades_completed}/${grades_expected} after a permanent provider error; automatic retry is disabled"
    exit 2
  fi

  if (( grades_completed > grades_before )); then
    transient_restarts_without_progress=1
  else
    (( transient_restarts_without_progress += 1 ))
  fi
  if (( transient_restarts_without_progress > MAX_TRANSIENT_RESTARTS_WITHOUT_PROGRESS )); then
    print -u2 -- \
      "E1 grading stopped at ${grades_completed}/${grades_expected} after ${MAX_TRANSIENT_RESTARTS_WITHOUT_PROGRESS} transient restarts without progress"
    exit 3
  fi

  print -u2 -- \
    "E1 grading paused at ${grades_completed}/${grades_expected} after a transient provider error; retry ${transient_restarts_without_progress}/${MAX_TRANSIENT_RESTARTS_WITHOUT_PROGRESS} starts in 60 seconds"
  sleep 60
done

UV_CACHE_DIR=/tmp/uv-cache-lock uv run --no-sync mfh finalize-e1 \
  "${E1_INPUTS[@]}" "${OUTPUT_DIRECTORY}" \
  --expected-split-manifest-digest "${EXPECTED_SPLIT_MANIFEST}" \
  --expected-grader-manifest-digest "${EXPECTED_GRADER_MANIFEST}"

readonly OUTPUT_MANIFEST_DIGEST="$(
  python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest_digest"])' \
    "${OUTPUT_DIRECTORY}/manifest.json"
)"

UV_CACHE_DIR=/tmp/uv-cache-lock uv run --no-sync mfh verify-e1-outputs \
  "${OUTPUT_DIRECTORY}" "${STUDY_ROOT}/work/E1" "${STUDY_ROOT}/runs/E1" \
  --expected-manifest-digest "${OUTPUT_MANIFEST_DIGEST}"
