import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with VideoLLaMA3 model")
    
    # Input/output file paths
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    
    # Prompt options
    parser.add_argument("--prompt_type", type=str, default="with_description", 
                        choices=["with_description", "basic", "mcq"],
                        help="Type of prompt to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    df = pd.read_csv(args.input_csv)

    device = "cuda"
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running inference"):
        modal_path = row['file_path']
        
        # Define prompts based on prompt_type
        if args.prompt_type == "basic":
            instruct = "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]"
        elif args.prompt_type == "with_description":
            instruct = f"{row['cleaned_text']}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]"
        elif args.prompt_type == "mcq":
            instruct = f"Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{row['question']}"
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": modal_path, "fps": 1, "max_frames": 180}},
                    {"type": "text", "text": instruct},
                ]
            },
        ]
        
        try:
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            output_ids = model.generate(**inputs, max_new_tokens=1024)
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            results.append(response)
        except Exception as e:
            print(f"Error processing {modal_path}: {str(e)}")
            results.append("Error")
    
    df_results = df
    df_results['model_output'] = results
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()