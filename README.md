<p align="center">
    <img src="assets/risk-mitigation.png" width="100" style="margin-bottom: 0.2;"/>
</p>

<h2 align="center">RiskCueBench: Evaluating Vision Language Models for Anticipatory Risk Reasoning in Videos</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2601.03369-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.03369)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E.svg?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/Yogesh914/RiskCueBench)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</h5>

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
