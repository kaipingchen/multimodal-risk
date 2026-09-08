import argparse
from google import genai
from tqdm import tqdm
import pandas as pd
import os
from google.genai import types
import json
from json_repair import repair_json

def get_video_analysis_with_retry(client, video_id, base_prompt, model_name, max_retries=8):
    """
    Get video analysis with JSON validation and retry logic
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(
                    parts=[
                        types.Part(file_data=types.FileData(file_uri=f'https://www.youtube.com/watch?v={video_id}')),
                        types.Part(text=base_prompt)
                    ]
                ),
                config={
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category_number": {"type": "integer", "minimum": 1, "maximum": 12},
                                "category_name": {"type": "string"},
                                "score": {"type": "integer", "minimum": 0, "maximum": 2}
                            },
                            "required": ["category_number", "category_name", "score"]
                        },
                        "minItems": 12,
                        "maxItems": 12
                    }
                }
            )
            
            # Try to parse the JSON
            result = json.loads(response.text)
            
            # Validate it has all 12 categories
            if len(result) == 12:
                return result
            else:
                print(f"Attempt {attempt + 1}: Got {len(result)} categories instead of 12")
                
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON decode error, trying to repair...")
            try:
                repaired = repair_json(response.text)
                result = json.loads(repaired)
                if len(result) == 12:
                    return result
                else:
                    print(f"Attempt {attempt + 1}: Repaired JSON has {len(result)} categories instead of 12")
            except Exception as repair_error:
                print(f"Attempt {attempt + 1}: Could not repair JSON - {repair_error}")
        
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")
    
    raise Exception(f"Failed to get valid response after {max_retries} attempts for video {video_id}")

def main():
    parser = argparse.ArgumentParser(description="Gemini 12-criterion video quality assessment")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-name", default="gemini-3.1-flash-lite")
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()
    if args.max_retries < 1:
        parser.error("--max-retries must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    input_path = args.input_csv
    output_path = args.output_csv

    # Load data
    df = pd.read_csv(input_path)
    
    # Initialize Gemini client
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    MODEL_NAME = args.model_name
    
    
    results = [None] * len(df)

    if os.path.exists(output_path):
        checkpoint_df = pd.read_csv(output_path)
        if len(checkpoint_df) == len(df) and "model_output" in checkpoint_df.columns:
            results = checkpoint_df["model_output"].tolist()
            completed = int(pd.Series(results).notna().sum())
            print(f"Resuming from checkpoint: {completed}/{len(df)} videos already processed")
        else:
            print("Checkpoint found but shape/columns do not match input. Starting fresh.")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        if pd.notna(results[idx]):
            continue

        video_id = row['video_id']

        base_prompt = """
        You are a video analysis expert tasked with evaluating YouTube videos based on 12 specific criteria. Analyze the provided video and assign scores for each category based on the detailed scoring rubric below.

        ## SCORING SYSTEM
        For each category, assign exactly one score:
        - Score 0 = Fails to meet criterion
        - Score 1 = Partially meets or unclear
        - Score 2 = Fully meets criterion

        ## EVALUATION CRITERIA

        ### 1. Logo Assessment
        **Criterion**: Video should have no news organization branding or features
        - **Score 0**: Obvious news marks, logos, or clear media branding visible
        - **Score 1**: Ambiguous news marks/logos or possible reposts with unclear branding
        - **Score 2**: No branding visible, appears to be bystander-style or CCTV-style capture

        ### 2. Location Information
        **Criterion**: Video should have no location mentioned in text or audio
        - **Score 0**: Obvious location references in text or audio (city, state, country)
        - **Score 1**: Ambiguous location references (hypothetical/unrealistic places)
        - **Score 2**: No location information present in text or audio

        ### 3. Time/Date Information
        **Criterion**: Video should have no time/date mentioned in text or audio
        - **Score 0**: Obvious time/date references (e.g., "May 25th, 2025", "13:00pm")
        - **Score 1**: Ambiguous temporal references (e.g., "a thousand years ago/later")
        - **Score 2**: No time/date information present in text or audio

        ### 4. Reporter Presence
        **Criterion**: Video should have no reporter, anchor, or journalist present
        - **Score 0**: Reporter, journalist, or anchor clearly present and identifiable
        - **Score 1**: Possibly media-affiliated speaker or unclear professional presence
        - **Score 2**: No formal speaker present, only participants/bystanders

        ### 5. SNS Engagement Overlays
        **Criterion**: Video should have no social media overlays or engagement icons
        - **Score 0**: Social media overlays/pop-ups clearly visible (likes, shares, emojis)
        - **Score 1**: Minimal or transient overlays present
        - **Score 2**: No overlays or engagement elements visible

        ### 6. Natural Image Quality
        **Criterion**: Video should show no signs of professional editing
        - **Score 0**: Highly edited with stylized transitions, cuts, or montage effects
        - **Score 1**: Some editing artifacts present or unclear transitions
        - **Score 2**: Continuous, shaky, or natural camera movement with no professional editing

        ### 7. Natural Temporal Continuity
        **Criterion**: Video should be continuous recording without interruptions
        - **Score 0**: Obvious jump cuts, stitched scenes, or time gaps
        - **Score 1**: Possibly continuous but breaks/transitions unclear
        - **Score 2**: One uninterrupted shot with natural temporal flow

        ### 8. Consequence Text
        **Criterion**: Video should have no embedded text about protest consequences
        - **Score 0**: Text describing consequences/severity (arrests, crackdowns, deaths, violent clashes)
        - **Score 1**: Unclear or minor commentary text present
        - **Score 2**: No consequence text; only basic metadata like location/time allowed

        ### 9. Title/Description/Banner Text
        **Criterion**: Video should have no inflammatory title/description banners
        - **Score 0**: Text mentioning violence, specific protest names, arrests, or inflammatory content
        - **Score 1**: Title/description is unclear or vague
        - **Score 2**: Only factual metadata like place/date present

        ### 10. Subtitle Text
        **Criterion**: Video should have no speech transcription subtitles
        - **Score 0**: Subtitles clearly present showing speech transcriptions
        - **Score 1**: Some subtitle presence but hard to distinguish
        - **Score 2**: No subtitles visible at all

        ### 11. Camera Perspective
        **Criterion**: Video should appear to be from participant, bystander, or CCTV perspective
        - **Score 0**: Clearly taken from media zone or professional media position
        - **Score 1**: Drone footage or distant zoomed footage
        - **Score 2**: Ground-level, hand-held, immersed in crowd, or surveillance camera perspective

        ### 12. Post-Production Effects
        **Criterion**: Video should have no post-processing effects applied
        - **Score 0**: Clear post-processing (filters, slow motion, stabilization effects)
        - **Score 1**: Some smoothing effects or unknown processing present
        - **Score 2**: Completely raw, unfiltered footage

        ## OUTPUT FORMAT
        Provide your analysis as a JSON array containing exactly 12 objects, one for each category. Each object must include:
        - category_number: integer (1-12)
        - category_name: string (exact name from criteria above)
        - score: integer (0, 1, or 2)

        ## EXAMPLE OUTPUT FORMAT
        [
        {
            "category_number": 1,
            "category_name": "Logo Assessment",
            "score": 2
        },
        {
            "category_number": 2,
            "category_name": "Location Information",
            "score": 0
        }
        // ... continue for all 12 categories
        ]

        ## ANALYSIS INSTRUCTIONS
        1. Watch the entire video carefully
        2. Listen to all audio content
        3. Examine all visible text and overlays
        4. Assess video quality and editing characteristics
        5. Score each category independently
        6. If uncertain between two scores, choose the lower score
        7. Ensure all 12 categories are evaluated and included in response

        Begin your analysis now.
        """

        try:
            # Use the retry function to get valid JSON response
            output = get_video_analysis_with_retry(client, video_id, base_prompt, MODEL_NAME, args.max_retries)
            # print(output)
            results[idx] = output
        
        except Exception as e:
            err_str = str(e)
            print(f"Error processing video {video_id}: {err_str}")
            results[idx] = "Error"
            continue

        # Save checkpoint every 100 videos
        if (idx + 1) % 100 == 0:
            df_results = df.copy()
            df_results['model_output'] = results
            df_results.to_csv(output_path, index=False)
            print(f"Checkpoint saved at {idx + 1} videos")

    # Final save
    df_results = df.copy()
    df_results['model_output'] = results
    df_results.to_csv(output_path, index=False)
    print(f"Final results saved to {output_path}")


if __name__ == "__main__":
    main()
