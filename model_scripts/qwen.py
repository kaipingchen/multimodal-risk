import argparse
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import av
import numpy as np
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with Qwen2.5-VL model")
    
    # Input/output file paths
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    
    # Model configuration
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model ID from HuggingFace")
    parser.add_argument("--min_pixels", type=int, default=256*28*28, help="Minimum pixel value for processor")
    parser.add_argument("--max_pixels", type=int, default=1024*28*28, help="Maximum pixel value for processor")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second for video processing")
    
    # Prompt options
    parser.add_argument("--prompt_type", type=str, default="with_description", 
                        choices=["with_description", "basic", "mcq"],
                        help="Type of prompt to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    
    # Initialize model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(
        args.model_id, 
        min_pixels=args.min_pixels, 
        max_pixels=args.max_pixels, 
        use_fast=True
    )
    
    # Define prompts
    prompts = {
        "basic": "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",
        "with_description": "{}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",
        "mcq": "Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{}"
    }
    
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            video_path = row["file_path"]
            
            # Construct prompt based on type
            if args.prompt_type == "with_description":
                prompt = prompts[args.prompt_type].format(row['cleaned_text'])
            elif args.prompt_type == "mcq":
                prompt = prompts[args.prompt_type].format(row['question'])
            else:
                prompt = prompts[args.prompt_type]
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": args.fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to("cuda")
            
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            results.append(output_text[0])
            
            # Clean up GPU memory
            del generated_ids, inputs, generated_ids_trimmed
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results.append("Error")
    
    # Save results
    df_results = df.copy()
    df_results['model_output'] = results
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()