import argparse
import torch
from transformers import AutoModelForCausalLM
from Apollo.apollo.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_mm_token,
    ApolloMMLoader
)
from Apollo.apollo.conversation import conv_templates, SeparatorStyle
from huggingface_hub import snapshot_download
from tqdm import tqdm
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with Apollo model")
    
    # Input/output file paths
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model_id", type=str, default="GoodiesHere/Apollo-LMMs-Apollo-7B-t32", help="Model ID from HuggingFace")
    # Prompt options
    parser.add_argument("--prompt_type", type=str, default="with_description", 
                        choices=["with_description", "basic", "mcq"],
                        help="Type of prompt to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    df = pd.read_csv(args.input_csv)

    model_url = args.model_id
    model_path = snapshot_download(model_url, repo_type="model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device=device, dtype=torch.bfloat16)

    tokenizer = model.tokenizer
    vision_processors = model.vision_tower.vision_processor
    config = model.config
    num_repeat_token = config.mm_connector_cfg['num_output_tokens']
    mm_processor = ApolloMMLoader(
        vision_processors,
        config.clip_duration,
        frames_per_clip=4,
        clip_sampling_ratio=0.65,
        model_max_length=config.model_max_length,
        device=device,
        num_repeat_token=num_repeat_token
    )

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running inference"):
        video_path = row['file_path']
        
        # Define prompts based on prompt_type
        if args.prompt_type == "basic":
            question = "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]"
        elif args.prompt_type == "with_description":
            question = f"{row['cleaned_text']}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]"
        elif args.prompt_type == "mcq":
            question = f"Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{row['question']}"

#         add_prompt = """Risk: arrest, fire, heavy smoke,  tear gas, breaking things, pepper balls, throwing bottles, breaching iron barrier/fence, water gun, blocking highway, climbing onto cars, cars running into protestors, physical aggression (police hitting protestors with batons, fights, riding on fallen people, kicking).
# Nonrisk: singing with drum, marching, speech, waving flags, holding signs, shaking hands/talking with police, allow police to herd them, protesting while not disturbing traffic/business operations.
# Given the above information, answer the following question:"""
#         question = add_prompt + "\n" + question
        
        try:
            mm_data, replace_string = mm_processor.load_video(video_path)

            conv = conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], replace_string + "\n\n" + question)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            input_ids = tokenizer_mm_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    vision_input=[mm_data],
                    data_types=['video'],
                    do_sample=True,
                    temperature=0.4,
                    max_new_tokens=256,
                    top_p=0.7,
                    use_cache=True,
                    num_beams=1,
                    stopping_criteria=[stopping_criteria]
                )

            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            results.append(pred)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            results.append("Error")
    
    df_results = df
    df_results['model_output'] = results
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()