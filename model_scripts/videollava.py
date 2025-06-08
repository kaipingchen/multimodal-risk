#!/usr/bin/env python
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with LLaVA-Video model")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--prompt_type", type=str, default="with_description", 
                        choices=["with_description", "basic", "mcq"],
                        help="Type of prompt to use")
    return parser.parse_args()


def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time


def main():
    args = parse_args()
    
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.to("cuda").eval()

    df = pd.read_csv(args.input_csv)

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row["file_path"]
        max_frames_num = 64
        try:
            video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(torch.bfloat16).cuda()
            video = [video]
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."

            # Define prompt based on prompt_type
            if args.prompt_type == "basic":
                prompt = "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]"
            elif args.prompt_type == "with_description":
                prompt = f"{row['cleaned_text']}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]"
            elif args.prompt_type == "mcq":
                prompt = f"Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{row['question']}"

            question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{prompt}"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            cont = model.generate(
                input_ids,
                images=video,
                modalities= ["video"],
                do_sample=False,
                temperature=0.2,
                max_new_tokens=1024,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            results.append(text_outputs)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results.append("Error")

    df_results = df
    df_results['model_output'] = results
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()