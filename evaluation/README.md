# Evaluation

[Repository overview](../README.md) · [Model inference](../model_infer/README.md)

Run commands from the **repository root**:

```bash
python -m pip install -r evaluation/requirements.txt
mkdir -p results
```

| Metric | What it measures | Script |
| --- | --- | --- |
| **F1** | Binary risk prediction for the positive `yes` class | [calculate_accuracy.py](calculate_accuracy.py) |
| **RGA** | Alignment of reasoning items with annotated visual indicators | [calculate_rga.py](calculate_rga.py) |
| **TRD** | Mean absolute F1 change under shuffled, half-swapped, and reversed frames | [calculate_trd_scd.py](calculate_trd_scd.py) |
| **SCD** | Subset-size-weighted F1 with self-correction minus that without it | [calculate_trd_scd.py](calculate_trd_scd.py) |

Definitions follow [Section 3.4 of the paper](https://arxiv.org/html/2601.03369v1#S3.SS4). Run inference first, then extract reasoning annotations or prepare temporal variants as needed.

## Risk prediction F1

```bash
python evaluation/calculate_accuracy.py --input results/qwen.csv
```

Requires `riskLabel` (`yes`/`no`) and the VLM's `model_output`. Reports accuracy, confusion counts, precision, recall, and positive-class F1. Missing, `Error`, and unparseable predictions are excluded and counted; invalid labels are rejected. The parser checks answer tags, boxed answers, then the last standalone yes/no word. Check it against your model's response format and report exclusions alongside scores.

## Extract confusion counts and decision items

[analyze_reasoning.py](analyze_reasoning.py) uses **Gemini 3.1 Flash-Lite** to analyze VLM reasoning text. Set `GEMINI_API_KEY` in your environment:

```bash
python -m pip install google-genai json-repair
python evaluation/analyze_reasoning.py \
  --input-csv data/predictions.csv --output-csv results/judge_results.csv
```

Input columns: `video_id`, `thinking` (reasoning), `riskLabel`, and `pred` (model prediction). Use `--reasoning-column` and `--prediction-column` for other column names; `--model-name` and `--max-retries` are also configurable. Only text is sent to the judge.

The output retains source columns and adds:

- `model_output`: JSON with `confusion_count` and `decision_items`.
- `confusion_count`: nonnegative integer; blank for failed analyses.
- `decision_items`: JSON list of objects/items used in the reasoning.

An existing `model_output` is preserved as `original_model_output`. Failed analyses are marked `Error`; review or exclude them before scoring. Empty reasoning is recorded as a failure, not a zero confusion count. The script saves at the end and requires a separate output path.

For RGA, retain `riskVisualIndicator` and `eval` in the input so they carry into the judge results. For SCD, use the original risk prediction with the extracted count. With the default input columns, prepare both files as follows:

```python
import pandas as pd

judged = pd.read_csv("results/judge_results.csv")
valid = judged.loc[judged["confusion_count"].notna()].copy()
valid.to_csv("results/rga_input.csv", index=False)
valid["model_output"] = valid["pred"]  # Restore the risk prediction for SCD.
valid.to_csv("results/scd_input.csv", index=False)
```

Use those files in the RGA and SCD commands below. If you selected another prediction column, replace `pred` in the preparation code accordingly. Record how many judge failures were excluded.

## Reasoning Grounding Accuracy (RGA)

```bash
python evaluation/calculate_rga.py \
  --input results/rga_input.csv --output results/rga.csv
```

Requires `riskVisualIndicator` (a serialized list or comma-separated text) and **judge output** in `model_output`, such as `{"decision_items": ["traffic light", "vehicle"]}`. The script uses `all-MiniLM-L6-v2` embeddings and writes per-row RGA, top/bottom 10% CSVs, and a printed dataset mean, on a 0–100 scale.

By default, threshold fitting requires binary prediction-correctness labels in `eval`, with both classes present. For held-out evaluation, fit on calibration data and reuse the printed value with `--threshold`. Empty decision lists score zero; missing indicators produce zero similarities. **Omitting `--output` updates the input CSV in place.**

## Temporal Reasoning Difference (TRD)

```bash
python evaluation/calculate_trd_scd.py trd \
  --original results/original.csv --shuffled results/shuffled.csv \
  --swapped results/swapped.csv --reversed results/reversed.csv
```

Each CSV requires `video_id`, `riskLabel`, and `model_output`. IDs and labels must match across conditions. An unparseable prediction excludes that clip from all four runs. The JSON report includes per-condition F1, mean absolute F1 difference, and evaluated/excluded counts. Record frame-sampling settings and augmentation seeds with your results.

## Self-Correction Degradation (SCD)

```bash
python evaluation/calculate_trd_scd.py scd --input results/scd_input.csv
```

Requires `video_id`, `riskLabel`, the **original VLM prediction** in `model_output`, and a judge-extracted nonnegative integer `confusion_count`. Computes `(n_plus × F1_plus − n_minus × F1_minus) / n_valid`, where `plus` means `confusion_count > 0`. Reports subset sizes, F1 values, and exclusions; an empty subset contributes zero.

F1, TRD, and SCD use the 0–1 F1 scale; SCD can be negative. TRD/SCD require unique clip IDs and at least one valid evaluated clip. For custom IDs, put `--id-column clip_id` **before** `trd` or `scd`.
