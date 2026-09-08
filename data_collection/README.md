# Collecting hard videos

[Repository overview](../README.md) · [Model inference](../model_infer/README.md)

The workflow is **Gemini quality assessment → candidate filtering → model disagreement → human annotation**. Run commands from the **repository root**.

| Script | Purpose |
| --- | --- |
| [gemini_data_local.py](gemini_data_local.py) | Apply the 12 quality criteria to local MP4 files |
| [gemini_data_yt.py](gemini_data_yt.py) | Apply the same criteria using YouTube video IDs |
| [run_model_disagreement.sh](run_model_disagreement.sh) | Run Gemini and five local VLMs, then rank candidates |
| [rank_hard_videos.py](rank_hard_videos.py) | Rank existing model predictions by disagreement |

## 1. Assess quality with Gemini

Set `GEMINI_API_KEY` in your environment and install the dependencies. No local GPU is needed for this step.

```bash
python -m pip install google-genai json-repair pandas tqdm

# Local MP4 files: input CSV requires file_path.
python data_collection/gemini_data_local.py \
  --input-csv data/candidates.csv --output-csv results/quality.csv

# Alternatively, YouTube videos: input CSV requires video_id.
python data_collection/gemini_data_yt.py \
  --input-csv data/youtube_candidates.csv --output-csv results/quality_youtube.csv
```

Both scripts default to **`gemini-3.1-flash-lite`** and accept `--model-name` and `--max-retries`. The original rubric scores 12 criteria from 0 to 2, including branding, overlays, temporal continuity, consequence-revealing text, camera perspective, and editing. The output preserves input columns and adds `model_output`, containing a category-score array or `Error`.

The local runner sends MP4 bytes and saves its CSV at the end. The YouTube runner sends video URLs, periodically checkpoints, and resumes nonempty outputs—including `Error`—when the output CSV has the same row count. Reuse that output only with identical input and row order; use a fresh output path to reassess everything.

## 2. Filter candidates

Parse the score arrays, exclude failed assessments, and select candidates for your domain. The research notebooks examined videos scoring 2 on all 12 criteria and used a maximum duration of 1,200 seconds. The assessment scripts export scores; they do not apply those filters automatically.

Prepare a retained-candidate CSV with local `file_path` values and unique `video_id` values for the next step. If several clips share a video ID, add a unique `clip_id` instead.

## 3. Run the disagreement panel

Follow the [vLLM setup](../model_infer/README.md#local-models-with-vllm) and set `GEMINI_API_KEY`. The launcher runs the following panel sequentially:

| Model | Checkpoint / API model | Backend |
| --- | --- | --- |
| Gemini | `gemini-3.1-flash-lite-preview` (override with `GEMINI_MODEL`) | Gemini API |
| InternVL3.5-8B | `OpenGVLab/InternVL3_5-8B` | vLLM |
| MiMo-VL-7B-RL | `XiaomiMiMo/MiMo-VL-7B-RL-2508` | vLLM |
| MiMo-VL-7B-SFT | `XiaomiMiMo/MiMo-VL-7B-SFT-2508` | vLLM |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | vLLM |
| Qwen3-VL-8B-Thinking | `Qwen/Qwen3-VL-8B-Thinking` | vLLM |

```bash
GEMINI_MODEL=gemini-3.1-flash-lite CUDA_VISIBLE_DEVICES=0 \
  bash data_collection/run_model_disagreement.sh \
  data/retained_candidates.csv results/model_disagreement
```

This writes six prediction CSVs and `hard_candidates.csv`. Every run uses the `basic_rl` yes/no prompt. Local models default to 1 fps and one GPU; configure `FPS`, `TENSOR_PARALLEL_SIZE`, and `CUDA_VISIBLE_DEVICES` as needed. Set `ID_COLUMN=clip_id` for custom clip IDs. Gemini uploads local clips to its API.

The original ranking notebook used Gemini, InternVL, MiMo-RL, MiMo-SFT, and Qwen-7B; the launcher uses the six-model panel above.

## 4. Rank and annotate

To rank predictions you already have from at least two models:

```bash
python data_collection/rank_hard_videos.py \
  --inputs results/gemini.csv results/internvl.csv results/mimo_rl.csv \
           results/mimo_sft.csv results/qwen.csv \
  --output results/hard_candidates.csv
```

Each CSV needs `video_id` and `model_output`, with the same unique IDs across files. Use `--id-column clip_id` when appropriate. A clip is excluded if any model answer cannot be parsed.

The output retains metadata from the first input and adds model votes and `model_disagreement_score`: the sample variance of yes/no votes (`yes=1`, `no=0`, `ddof=1`). Higher scores mean more disagreement; unanimous votes score zero. Compare rankings using the same model panel.

Review the highest-ranking candidates and annotate their risk cues, temporal boundaries, and eventual outcomes. The script ranks all valid clips without imposing a selection threshold; disagreement does not establish a ground-truth risk label.
