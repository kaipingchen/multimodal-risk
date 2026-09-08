# Model inference

[Repository overview](../README.md) · [Hard-video collection](../data_collection/README.md)

Run the commands below from the **repository root**. Each runner preserves input columns and adds `model_output`; failed inference is recorded as `Error`.

## Prepare the input

Create a CSV with one row per clip:

| Column | Purpose |
| --- | --- |
| `file_path` | Path to the local video clip |
| `video_id` | Unique clip identifier for disagreement and temporal evaluation |
| `riskLabel` | Ground truth (`yes`/`no`), required for evaluation |

Download annotations from [Hugging Face](https://huggingface.co/datasets/Yogesh914/RiskCueBench) and obtain source videos using its dataset card. Trim videos to the annotated risk-signal interval before inference; the runners read the whole file and do not apply annotation timestamps. Resolve missing labels before evaluation. For multiple clips from one source video, retain a unique `clip_id` and select it when ranking or evaluating.

## Local models with vLLM

Use Python 3.11+ and a compatible CUDA/PyTorch environment:

```bash
python -m pip install -r requirements.txt
python -m pip install pandas qwen-vl-utils av
mkdir -p results

python model_infer/vllm_inference.py \
  --model-type qwen2_5_vl \
  --input-csv data/benchmark.csv --output-csv results/qwen.csv \
  --prompt-type basic --modality video --fps 1.0
```

Select GPUs with `CUDA_VISIBLE_DEVICES` and set `--tensor-parallel-size` as needed. Dependencies and GPU memory requirements vary by model; the requirements file is not a fully locked reproduction environment.

The [vLLM runner](vllm_inference.py) accepts these model keys:

| Family | `--model-type` |
| --- | --- |
| Qwen | `qwen2_5_vl`, `qwen3_vl`, `qwen3_vl_8b_instruct`, `qwen3_vl_8b_thinking` |
| InternVL | `internvl_chat` (InternVL3-2B), `internvl3_5_8b` |
| MiMo | `mimo_vl_7b_rl_2508`, `mimo_vl_7b_sft_2508` |
| Other | `glm4_1v`, `hyperclovax_seed_vision`, `keye_vl1_5`, `llava-next-video`, `llava-onevision`, `ovis2_5` |

Checkpoint IDs are defined in the script. The [Apollo](apollo.py) and [VideoLLaMA3](vl3.py) research runners retain their model-specific implementations; inspect their dependencies and configuration before use.

## Gemini API

Install `google-genai` and set `GEMINI_API_KEY` in your environment. The [Gemini runner](gemini.py) uploads local clips to the API:

```bash
python -m pip install google-genai pandas tqdm
mkdir -p results

python model_infer/gemini.py \
  --model-name gemini-3.1-flash-lite \
  --input_csv data/benchmark.csv --output_csv results/gemini.csv \
  --prompt_type basic_rl
```

## Prompts and evaluation

| Prompt | Behavior / additional input |
| --- | --- |
| `basic` | Reasoning and a yes/no risk prediction |
| `basic_rl` | Yes/no risk prediction |
| `with_description` | Adds the `cleaned_text` column to the prompt |
| `mcq` | Uses the `question` column; outputs a multiple-choice answer |

Evaluate binary risk predictions with:

```bash
python evaluation/calculate_accuracy.py --input results/qwen.csv
```

See the [metric guide](../evaluation/README.md) for F1, RGA, TRD, and SCD. MCQ outputs cannot be scored with the binary F1 utility.
