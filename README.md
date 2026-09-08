<p align="center">
  <img src="assets/risk-mitigation.png" width="100" alt="RiskCueBench" />
</p>

# RiskCueBench

**Benchmarking Anticipatory Reasoning from Early Risk Cues in Video-Language Models**

[Paper](https://arxiv.org/abs/2601.03369) · [Hand-annotated dataset on Hugging Face](https://huggingface.co/datasets/Yogesh914/RiskCueBench)

RiskCueBench evaluates whether video-language models can anticipate incidents from early visual cues. This repository provides tools to collect challenging videos, run models, and calculate risk-prediction and reasoning metrics.

## Repository guide

| Folder | Contents |
| --- | --- |
| [data_collection/](data_collection/README.md) | Gemini quality assessment, multi-model disagreement ranking, and guidance for collecting hard videos |
| [model_infer/](model_infer/README.md) | Setup and usage for local VLMs through vLLM and Gemini API inference |
| [evaluation/](evaluation/README.md) | Reasoning annotation (confusion counts and decision items) and **F1**, **RGA**, **TRD**, **SCD** evaluation |
| [figures/](figures/) | Grounding and reasoning-length analysis notebook and its input data |
| [util/](util/) | API request helper |
| [assets/](assets/) | README artwork |

## Dataset

The [Hugging Face dataset](https://huggingface.co/datasets/Yogesh914/RiskCueBench) contains hand annotations for `carcrash`, `protest`, `arson`, and `cruise`. Its dataset card explains the fields and how to obtain source videos; downloading annotations does not download footage.

```python
from datasets import load_dataset  # pip install datasets
benchmark = load_dataset("Yogesh914/RiskCueBench")
```

See [inference preparation](model_infer/README.md#prepare-the-input) for creating risk-signal clips and input CSVs.

## Analysis

Open [figures/plots.ipynb](figures/plots.ipynb) with `figures/` as the working directory. It uses the bundled `figures/data/` inputs and requires Jupyter, matplotlib, and scipy in addition to the evaluation dependencies.
