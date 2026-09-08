import argparse
import time
from google import genai
from tqdm import tqdm
import pandas as pd
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Process videos with Gemini API")
    
    # Required arguments
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Gemini model name to use for inference.",
    )
    
    # Prompt options
    parser.add_argument("--prompt_type", type=str, default="basic", 
                        choices=["basic", "with_description", "mcq", "basic_rl"],
                        help="Type of prompt to use")
    
    # Processing options
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for video processing")
    
    return parser.parse_args()


def is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    retry_markers = [
        "503",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "429",
        "500",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
    ]
    return any(marker in msg for marker in retry_markers)


def with_retries(fn, *, max_attempts: int = 5, base_delay: float = 2.0):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts or not is_retryable_error(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"Retryable Gemini error on attempt {attempt}/{max_attempts}: {exc}. "
                f"Sleeping {delay:.1f}s before retry."
            )
            time.sleep(delay)
    raise last_error


def main():
    args = parse_args()
    
    # Load data
    df = pd.read_csv(args.input_csv)
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    MODEL_NAME = args.model_name
    
    # Define prompts
    prompts = {
        "basic": "Predict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",
        "with_description": "{}\nPredict whether a risk event is going to occur in the video by answering \"Yes\" or \"No\", and then provide a clear, step-by-step explanation of your reasoning for your answer. Format your response as follows:\nAnswer: [Yes/No]\nReasoning: [Explanation]",\
        "basic_rl": "Predict whether a risk event is going to occur in the video by answering in one word \"Yes\" or \"No\"",
        "mcq": "Answer the following multiple-choice question with ONLY the letter of the correct answer A, B, C, or D.\n{}"
    }
    
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_path = row['file_path']

        try:
            video_file = with_retries(lambda: client.files.upload(file=video_path))
        except Exception as e:
            print(f"Error uploading {video_path}: {e}")
            results.append("Error")
            continue
        
        try:
            start_time = time.time()
            while video_file.state.name == "PROCESSING":
                if time.time() - start_time > args.timeout:
                    raise TimeoutError(
                        f"Timed out after {args.timeout}s while processing {video_path}"
                    )
                time.sleep(2)
                video_file = with_retries(
                    lambda: client.files.get(name=video_file.name)
                )
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

        add_prompt = """Risk: arrest, fire, heavy smoke,  tear gas, breaking things, pepper balls, throwing bottles, breaching iron barrier/fence, water gun, blocking highway, climbing onto cars, cars running into protestors, physical aggression (police hitting protestors with batons, fights, riding on fallen people, kicking).
Nonrisk: singing with drum, marching, speech, waving flags, holding signs, shaking hands/talking with police, allow police to herd them, protesting while not disturbing traffic/business operations.
Given the above information, answer the following question:"""
        # base_prompt = add_prompt + "\n" + base_prompt

        try:
            response = with_retries(
                lambda: client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[video_file, base_prompt]
                )
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
