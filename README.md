<p align="center">
    <img src="assets/risk-mitigation.png" width="200" style="margin-bottom: 0.2;"/>
</p>

<h2 align="center">Multimodal Risk: A VLM Benchmarking Framework for Situational Risk Reasoning</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2601.03369-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.03369)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Yogesh914/RiskCueBench)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</h5>

---

## Overview

**Multimodal Risk** is a comprehensive benchmarking framework for evaluating Vision Language Models (VLMs) on situational risk reasoning tasks. The framework assesses how well state-of-the-art VLMs can predict and reason about risky events in videos and images, including scenarios like arrests, fires, violence, and property damage.

---

## Supported Models

| Model | Parameters | Source |
|:------|:-----------|:-------|
| GLM-4.1V-9B-Thinking | 9B | [zai-org/GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) |
| HyperCLOVAX-SEED-Think | 14B | [naver-hyperclovax/HyperCLOVAX-SEED-Think-14B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B) |
| InternVL3 | 2B | [OpenGVLab/InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) |
| Keye-VL-1.5 | 8B | [Kwai-Keye/Keye-VL-1.5-8B](https://huggingface.co/Kwai-Keye/Keye-VL-1.5-8B) |
| LLaVA-NeXT-Video | 7B | [llava-hf/LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) |
| LLaVA-OneVision | 7B | [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) |
| Ovis2.5 | 2B | [AIDC-AI/Ovis2.5-2B](https://huggingface.co/AIDC-AI/Ovis2.5-2B) |
| Qwen2.5-VL | 3B | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen3-VL | 4B | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Apollo | 7B | Custom Script |
| VideoLLaMA3 | 7B | [DAMO-NLP-SG/VideoLLaMA3-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B) |

---

## Installation

### Requirements

- Python >= 3.11
- PyTorch >= 2.0
- CUDA >= 11.8

### Setup

```bash
# Create conda environment
conda create -y --name multimodal_risk python=3.12
conda activate multimodal_risk

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Basic Inference

```python
# Single video inference with Qwen2.5-VL
python model_infer/vllm_inference.py \
    --model-type qwen2_5_vl \
    --modality video
```

### Batch Dataset Inference

```bash
python model_infer/vllm_inference.py \
    --model-type qwen2_5_vl \
    --input-csv data.csv \
    --output-csv results.csv \
    --prompt-type with_description \
    --modality video
```

### Using Apollo Model

```bash
python model_infer/apollo.py \
    --input_csv input_data.csv \
    --output_csv output_results.csv \
    --prompt_type with_description
```

### Using VideoLLaMA3

```bash
python model_infer/vl3.py \
    --input_csv input_data.csv \
    --output_csv output_results.csv \
    --prompt_type basic
```

---

## Prompt Types

| Type | Description |
|:-----|:------------|
| `basic` | Simple risk prediction (Yes/No with reasoning) |
| `basic_rl` | Reasoning-only variants |


---

## Key Parameters

```
--model-type             Model architecture to use
--modality               'image' or 'video'
--prompt-type            Prompt strategy (basic, with_description, mcq, basic_rl)
--fps                    Video frame rate (default: 1.0)
--tensor-parallel-size   GPU parallelism for large models
--num-frames             Number of frames for video processing
```

---

## Project Structure

```
multimodal-risk/
├── model_infer/
│   ├── vllm_inference.py    # Main vllm inference engine (9 models)
│   ├── apollo.py            # Apollo model inference
│   └── vl3.py               # VideoLLaMA3 inference
├── figures/                  # Analysis notebooks and plots
├── assets/                   # Visual resources
└── requirements.txt
```

---

## License

This project is released under the Apache 2.0 License.
