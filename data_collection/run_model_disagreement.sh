#!/usr/bin/env bash
# Run the workspace's six-model panel, then rank the shared candidate clips.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash data_collection/run_model_disagreement.sh INPUT_CSV OUTPUT_DIR" >&2
  exit 2
fi
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY before running the Gemini model}"
input_csv="$1"
output_dir="$2"
gemini_model="${GEMINI_MODEL:-gemini-3.1-flash-lite-preview}"
python_bin="${PYTHON:-python}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$output_dir"

"$python_bin" "$repo_dir/model_infer/gemini.py" \
  --model-name "$gemini_model" --input_csv "$input_csv" \
  --output_csv "$output_dir/gemini.csv" --prompt_type basic_rl

models=(internvl3_5_8b mimo_vl_7b_rl_2508 mimo_vl_7b_sft_2508
        qwen3_vl_8b_instruct qwen3_vl_8b_thinking)
prediction_files=("$output_dir/gemini.csv")
for model in "${models[@]}"; do
  "$python_bin" "$repo_dir/model_infer/vllm_inference.py" \
    --model-type "$model" --modality video --input-csv "$input_csv" \
    --output-csv "$output_dir/$model.csv" --prompt-type basic_rl \
    --fps "${FPS:-1.0}" --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  prediction_files+=("$output_dir/$model.csv")
done

"$python_bin" "$repo_dir/data_collection/rank_hard_videos.py" \
  --inputs "${prediction_files[@]}" --id-column "${ID_COLUMN:-video_id}" \
  --output "$output_dir/hard_candidates.csv"
