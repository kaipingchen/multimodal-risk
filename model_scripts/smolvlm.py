import argparse
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import av
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with SmolVLM model")
    
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--prompt_type", type=str, default="basic", 
                        choices=["basic", "with_description", "mcq"],
                        help="Type of prompt to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    df = pd.read_csv(args.input_csv)
    
    # Initialize the model and processor
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2"
    ).to("cuda")
    
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            video_path = row["file_path"]
            
            # Select prompt based on prompt_type
            if args.prompt_type == "basic":
                prompt = "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]"
            elif args.prompt_type == "with_description":
                prompt = f"{row['cleaned_text']}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]"
            elif args.prompt_type == "mcq":
                prompt = f"Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{row['question']}"
        
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_path},
                        {"type": "text", "text": prompt}
                    ]
                },
            ]
        
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
        
            generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1024, temperature=0.2)
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
        
            results.append(generated_texts[0])
        
            del generated_ids, inputs
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results.append("Error")
    
    df_results = df
    df_results['model_output'] = results
    df_results['model_output'] = df_results['model_output'].mask(
        df_results['model_output'] != 'Error',
        df_results['model_output'].str.split("Assistant:", expand=True)[1]
    )
    
    df_results.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()