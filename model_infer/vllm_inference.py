# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import av
import numpy as np
import random
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple

import pandas as pd
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.lora.request import LoRARequest
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser


# Prompt templates for dataset inference
PROMPTS = {
    "basic": "Predict whether a risk event is going to occur in the video. First, provide a clear, step-by-step explanation of your reasoning. Then, answer \"Yes\" or \"No\". Format your response as follows:\nReasoning: [Explanation]\nAnswer: [Yes/No]",
    "with_description": "{}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",
    "basic_rl": "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\"",
    "mcq": "Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{}"
}


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: list[int] | None = None
    lora_requests: list[LoRARequest] | None = None
    sampling_params: list[SamplingParams] | None = None


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.




# GLM-4.1V
def run_glm4_1v(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "zai-org/GLM-4.1V-9B-Thinking"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        mm_processor_kwargs={
            "size": {"shortest_edge": 12544, "longest_edge": 47040000},
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
        enforce_eager=True,
    )

    if modality == "image":
        placeholder = "<|begin_of_image|><|image|><|end_of_image|>"
    elif modality == "video":
        placeholder = "<|begin_of_video|><|video|><|end_of_video|>"

    prompts = [
        (
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            f"{placeholder}"
            f"{question}<|assistant|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B
def run_hyperclovax_seed_vision(
    questions: list[str], modality: str
) -> ModelRequestData:
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192 if modality == "image" else 16384,
        limit_mm_per_prompt={modality: 1},
    )

    messages = list()
    for question in questions:
        if modality == "image":
            """
            ocr: List the words in the image in raster order.
                Even if the word order feels unnatural for reading,
                the model will handle it as long as it follows raster order.
                e.g. "Naver, CLOVA, bigshane"
            lens_keywords: List the entity names in the image.
                e.g. "iPhone"
            lens_local_keywords: List the entity names with quads in the image.
                e.g. "[0.07, 0.21, 0.92, 0.90] iPhone"
            """
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "ocr": "",
                                "lens_keywords": "",
                                "lens_local_keywords": "",
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ]
            )
        elif modality == "video":
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ]
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=None,
    )




# InternVL
def run_internvl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "OpenGVLab/InternVL3-2B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# Keye-VL-1.5
def run_keye_vl1_5(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Kwai-Keye/Keye-VL-1.5-8B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )




# LlaVA-NeXT-Video
# Currently only support for video input
def run_llava_next_video(questions: list[str], modality: str) -> ModelRequestData:
    assert modality == "video"

    prompts = [f"USER: <video>\n{question} ASSISTANT:" for question in questions]
    engine_args = EngineArgs(
        model="llava-hf/LLaVA-NeXT-Video-7B-hf",
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# LLaVA-OneVision
def run_llava_onevision(questions: list[str], modality: str) -> ModelRequestData:
    if modality == "video":
        prompts = [
            f"<|im_start|>user <video>\n{question}<|im_end|><|im_start|>assistant\n"
            for question in questions
        ]

    elif modality == "image":
        prompts = [
            f"<|im_start|>user <image>\n{question}<|im_end|><|im_start|>assistant\n"
            for question in questions
        ]

    engine_args = EngineArgs(
        model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        max_model_len=16384,
        limit_mm_per_prompt={modality: 1},
    )

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )






# Ovis2_5
def run_ovis2_5(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "AIDC-AI/Ovis2.5-2B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        trust_remote_code=True,
        dtype="half",
        limit_mm_per_prompt={modality: 1},
    )
    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    prompts = [
        f"<|im_start|>user\n\n{placeholder}\n{question}<|im_end|>\n<|im_start|>assistant\n"
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )




# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen3-VL-Dense
def run_qwen3_vl(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "Qwen/Qwen3-VL-4B-Instruct"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def run_qwen_family(
    model_name: str,
    questions: list[str],
    modality: str,
    *,
    max_model_len: int = 8192,
    max_num_seqs: int = 1,
) -> ModelRequestData:
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )

def run_mimo_vl_rl(questions: list[str], modality: str) -> ModelRequestData:
    return run_qwen_family(
        "XiaomiMiMo/MiMo-VL-7B-RL-2508",
        questions,
        modality,
        max_model_len=8192,
    )

def run_mimo_vl_sft(questions: list[str], modality: str) -> ModelRequestData:
    return run_qwen_family(
        "XiaomiMiMo/MiMo-VL-7B-SFT-2508",
        questions,
        modality,
        max_model_len=8192,
    )

def run_qwen3_vl_8b_instruct(
    questions: list[str], modality: str
) -> ModelRequestData:
    return run_qwen_family(
        "Qwen/Qwen3-VL-8B-Instruct",
        questions,
        modality,
        max_model_len=8192,
    )

def run_qwen3_vl_8b_thinking(
    questions: list[str], modality: str
) -> ModelRequestData:
    return run_qwen_family(
        "Qwen/Qwen3-VL-8B-Thinking",
        questions,
        modality,
        max_model_len=8192,
    )

def run_internvl3_5(questions: list[str], modality: str) -> ModelRequestData:
    model_name = "OpenGVLab/InternVL3_5-8B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={modality: 1},
    )

    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}\n{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL3_5-8B
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )

model_example_map = {
    "glm4_1v": run_glm4_1v,
    "hyperclovax_seed_vision": run_hyperclovax_seed_vision,
    "internvl_chat": run_internvl,
    "internvl3_5_8b": run_internvl3_5,
    "mimo_vl_7b_rl_2508": run_mimo_vl_rl,
    "mimo_vl_7b_sft_2508": run_mimo_vl_sft,
    "qwen3_vl_8b_instruct": run_qwen3_vl_8b_instruct,
    "qwen3_vl_8b_thinking": run_qwen3_vl_8b_thinking,
    "keye_vl1_5": run_keye_vl1_5,
    "llava-next-video": run_llava_next_video,
    "llava-onevision": run_llava_onevision,
    "ovis2_5": run_ovis2_5,
    "qwen2_5_vl": run_qwen2_5_vl,
    "qwen3_vl": run_qwen3_vl,
}


MODELS_NEED_VIDEO_METADATA = [
    "glm4_1v",
    "qwen3_vl",
]


def get_multi_modal_input(args):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if args.modality == "image":
        # Input image and question
        image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
        img_questions = [
            "What is the content of this image?",
            "Describe the content of this image in detail.",
            "What's in the image?",
            "Where is this image taken?",
        ]

        return {
            "data": image,
            "questions": img_questions,
        }

    if args.modality == "video":
        # Input video and question
        needs_metadata = args.model_type in MODELS_NEED_VIDEO_METADATA
        video = VideoAsset(name="baby_reading", num_frames=args.num_frames).np_ndarrays
        metadata = VideoAsset(name="baby_reading", num_frames=args.num_frames).metadata
        vid_questions = ["Why is this video funny?"]

        return {
            "data": ([(video, metadata)] if needs_metadata else video),
            "questions": vid_questions,
        }

    msg = f"Modality {args.modality} is not supported."
    raise ValueError(msg)


def apply_image_repeat(
    image_repeat_prob, num_prompts, data, prompts: list[str], modality
):
    """Repeats images with provided probability of "image_repeat_prob".
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert image_repeat_prob <= 1.0 and image_repeat_prob >= 0
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    inputs_with_empty_media = []
    cur_image = data
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        uuid = "uuid_{}".format(i)

        inputs.append(
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {modality: cur_image},
                "multi_modal_uuids": {modality: uuid},
            }
        )

        inputs_with_empty_media.append(
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {modality: None},
                "multi_modal_uuids": {modality: uuid},
            }
        )

    return inputs, inputs_with_empty_media


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


def prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM from messages format."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def prepare_internvl_video(video_path, prompt, tokenizer, fps):
    """Sample RGB frames at the requested rate for vLLM's InternVL backend."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames = []
    next_time = 0.0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or fps)
        for index, frame in enumerate(container.decode(video=0)):
            timestamp = float(frame.time) if frame.time is not None else index / rate
            if timestamp >= next_time:
                frames.append(frame.to_ndarray(format="rgb24"))
                next_time = timestamp + 1.0 / fps
    if not frames:
        raise ValueError(f"No video frames decoded: {video_path}")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<video>\n{prompt}"}],
        tokenize=False, add_generation_prompt=True,
    )
    return {"prompt": text, "multi_modal_data": {"video": np.stack(frames)}}


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="llava-next-video",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--num-prompts", type=int, default=4, help="Number of prompts to run."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "video"],
        help="Modality of the input.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from the video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    parser.add_argument(
        "--image-repeat-prob",
        type=float,
        default=None,
        help="Simulates the hit-ratio for multi-modal preprocessor cache (if enabled)",
    )

    parser.add_argument(
        "--disable-mm-processor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal processor.",
    )

    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time",
    )

    parser.add_argument(
        "--use-different-prompt-per-request",
        action="store_true",
        help="If True, then use different prompt (with the same multi-modal "
        "data) for each request.",
    )

    parser.add_argument(
        "--verify-mm-cache-hit-with-uuids",
        action="store_true",
        help="If True, will send all requests in a second batch with empty mm "
        "data to verify cache hits with UUIDs.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=None,
        help="Tensor parallel size to override the model's default setting. ",
    )
    
    # Dataset inference arguments
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to input CSV file for dataset inference.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file for saving results.",
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="basic",
        choices=["with_description", "basic", "mcq", "basic_rl"],
        help="Type of prompt to use for dataset inference.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second for video processing.",
    )
    return parser.parse_args()


def run_dataset_inference(args):
    """Run inference on a dataset from CSV file."""
    # Load data
    df = pd.read_csv(args.input_csv)
    
    # Get model name from the model example function
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    
    # Get model configuration from example function
    dummy_questions = ["dummy"]
    req_data = model_example_map[model](dummy_questions, args.modality)
    model_name = req_data.engine_args.model
    
    # Initialize processor and vLLM model
    processor = None
    tokenizer = None
    if model in {"internvl_chat", "internvl3_5_8b"}:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "mm_processor_cache_gb": 0 if args.disable_mm_processor_cache else 4,
        "dtype": "bfloat16",
    }
    if args.tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = args.tensor_parallel_size
    else:
        engine_args["tensor_parallel_size"] = torch.cuda.device_count()
    
    llm = LLM(**engine_args)
    
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4048,
        top_k=-1,
        stop_token_ids=req_data.stop_token_ids,
    )
    
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            video_path = row["file_path"]
            
            # Construct prompt based on type
            if args.prompt_type == "with_description":
                prompt = PROMPTS[args.prompt_type].format(row['cleaned_text'])
            elif args.prompt_type == "mcq":
                prompt = PROMPTS[args.prompt_type].format(row['question'])
            else:
                prompt = PROMPTS[args.prompt_type]
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 360 * 420,
                            "fps": args.fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Prepare inputs for vLLM
            vllm_input = (
                prepare_internvl_video(video_path, prompt, tokenizer, args.fps)
                if tokenizer is not None
                else prepare_inputs_for_vllm(messages, processor)
            )
            
            # Generate with vLLM
            outputs = llm.generate([vllm_input], sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text
            
            print(output_text)
            results.append(output_text)
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results.append("Error")
    
    # Save results
    df_results = df.copy()
    df_results['model_output'] = results
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


def main(args):
    # If input CSV is provided, run dataset inference
    if args.input_csv is not None:
        if args.output_csv is None:
            raise ValueError("--output-csv is required when --input-csv is provided.")
        run_dataset_inference(args)
        return
    
    # Original demo inference logic
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be a positive integer, "
            f"got {args.tensor_parallel_size}"
        )

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    questions = mm_input["questions"]

    req_data = model_example_map[model](questions, modality)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "mm_processor_cache_gb": 0 if args.disable_mm_processor_cache else 4,
        "dtype": "bfloat16",
    }
    if args.tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = args.tensor_parallel_size
    llm = LLM(**engine_args)

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = (
        req_data.prompts
        if args.use_different_prompt_per_request
        else [req_data.prompts[0]]
    )

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = (
        SamplingParams(
            temperature=0.2, max_tokens=64, stop_token_ids=req_data.stop_token_ids
        )
        if req_data.sampling_params is None
        else req_data.sampling_params
    )

    assert args.num_prompts > 0
    if args.num_prompts == 1:
        # Single inference
        uuid = "uuid_0"
        inputs = {
            "prompt": prompts[0],
            "multi_modal_data": {modality: data},
            "multi_modal_uuids": {modality: uuid},
        }
        inputs_with_empty_media = {
            "prompt": prompts[0],
            "multi_modal_data": {modality: None},
            "multi_modal_uuids": {modality: uuid},
        }
    else:
        # Batch inference
        if args.image_repeat_prob is not None:
            # Repeat images with specified probability of "image_repeat_prob"
            inputs, inputs_with_empty_media = apply_image_repeat(
                args.image_repeat_prob,
                args.num_prompts,
                data,
                prompts,
                modality,
            )
        else:
            # Use the same image for all prompts
            inputs = []
            inputs_with_empty_media = []
            for i in range(args.num_prompts):
                uuid = "uuid_{}".format(i)
                inputs.append(
                    {
                        "prompt": prompts[i % len(prompts)],
                        "multi_modal_data": {modality: data},
                        "multi_modal_uuids": {modality: uuid},
                    }
                )
                inputs_with_empty_media.append(
                    {
                        "prompt": prompts[i % len(prompts)],
                        "multi_modal_data": {modality: None},
                        "multi_modal_uuids": {modality: uuid},
                    }
                )

    # Add LoRA request if applicable
    lora_request = (
        req_data.lora_requests * args.num_prompts if req_data.lora_requests else None
    )

    with time_counter(args.time_generate):
        outputs = llm.generate(
            inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

    print("-" * 50)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
        print("-" * 50)

    if args.verify_mm_cache_hit_with_uuids:
        try:
            # Verify cache hits with UUIDs
            print(
                "Sending a second batch of requests with empty media"
                " and matching UUIDs."
            )
            outputs = llm.generate(
                inputs_with_empty_media,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            print("-" * 50)
            for o in outputs:
                generated_text = o.outputs[0].text
                print(generated_text)
                print("-" * 50)
        except Exception as e:
            print(f"Failed to verify cache hits with UUIDs. Error: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)