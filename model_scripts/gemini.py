import argparse
from google import genai
from tqdm import tqdm
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with Gemini API")
    
    # Required arguments
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    
    # Prompt options
    parser.add_argument("--prompt_type", type=str, default="basic", 
                        choices=["basic", "with_description", "mcq"],
                        help="Type of prompt to use")
    
    # Processing options
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for video processing")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    
    # Initialize Gemini client
    client = genai.Client(api_key="AIzaSyAP8xsXE7TCDk4tnO0E9_1Q6eRmdPD0qQg")
    MODEL_NAME = "gemini-2.0-flash-lite"
    
    # Define prompts
    prompts = {
        "basic": "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",
        "with_description": "{}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",
        "mcq": "Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{}"
    }
    
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_path = row['file_path']

        try:
            video_file = client.files.upload(file=video_path)
        except Exception as e:
            print(f"Error uploading {video_path}: {e}")
            results.append("Error")
            continue
        
        try:
            while video_file.state.name == "PROCESSING":
                # print("Video is processing...", end="", flush=True)
                video_file = client.files.get(name=video_file.name)
        except Exception as e:
            print(f"Error getting {video_path}: {e}")
            results.append("Error")
            try:
                dr = client.files.delete(name=video_file.name)
            except Exception as e:
                print(f"Error deleting video file: {e}")
            continue

        if video_file.state.name == "FAILED":
            print(f"\nProcessing failed for video: {video_path}")
            results.append("Error")
            continue

        if args.prompt_type == "with_description":
            base_prompt = prompts[args.prompt_type].format(row['cleaned_text'])
        elif args.prompt_type == "mcq":
            base_prompt = prompts[args.prompt_type].format(row['question'])
        else:
            base_prompt = prompts[args.prompt_type]

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[video_file, base_prompt]
            )
            output = response.text

        except Exception as e:
            err_str = str(e)
            print(f"Error: {err_str}")
            results.append("Error")
            continue

        try:
            dr = client.files.delete(name=video_file.name)
        except Exception as e:
            print(f"Error deleting video file: {e}")

        results.append(output)
    
    # Save results
    df_results = df.copy()
    df_results['model_output'] = results
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()