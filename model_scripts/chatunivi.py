import torch
import os
import argparse
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from decord import VideoReader, cpu
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with ChatUniVi model")
    
    # Input/output file paths
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    
    # Prompt options
    parser.add_argument("--prompt_type", type=str, default="basic", 
                        choices=["with_description", "basic", "mcq"],
                        help="Type of prompt to use")
    
    return parser.parse_args()

# Function to get video frames (no change needed here)
def _get_rawvideo_dec(video_path, image_processor, max_frames=MAX_IMAGE_LENGTH, image_resolution=224, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))

def main():
    args = parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    
    # Model Parameter
    model_path = "Chat-UniVi/Chat-UniVi-7B-v1.5"
    max_frames = 100
    video_framerate = 1
    conv_mode = "simple"
    temperature = 0.2
    top_p = None
    num_beams = 1

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = "ChatUniVi"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    if model.config.config["use_cluster"]:
        for n, m in model.named_modules():
            m = m.to(dtype=torch.bfloat16)

    # Define prompts
    prompts = {
        "basic": "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",
        "with_description": "{}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",
        "mcq": "Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{}"
    }

    results = []

    # Iterate over each video in the dataset
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        try:
            video_path = row['file_path']

            # Construct prompt based on type
            if args.prompt_type == "with_description":
                qs = prompts[args.prompt_type].format(row['cleaned_text'])
            elif args.prompt_type == "mcq":
                qs = prompts[args.prompt_type].format(row['question'])
            else:
                qs = prompts[args.prompt_type]

            # Check if the video exists
            if video_path is not None and os.path.exists(video_path):
                video_frames, slice_len = _get_rawvideo_dec(video_path, image_processor, max_frames=max_frames, video_framerate=video_framerate)

                cur_prompt = qs
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                    0).cuda()

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=video_frames.half().cuda(),
                        do_sample=False,
                        # temperature=temperature,
                        # top_p=top_p,
                        # num_beams=num_beams,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_new_tokens=1024,
                        min_new_tokens=128,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])

                output_ids = output_ids.sequences
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
                # print(outputs)
                results.append(outputs)

            else:
                print(f'Video file {video_path} does not exist.')
                results.append("Error")
        except Exception as e:
            print(f'Error processing video {video_path}: {e}')
            results.append("Error")

    # Save results
    df_results = df.copy()
    df_results['model_output'] = results
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

# Run inference for each video in the dataset
if __name__ == '__main__':
    main()