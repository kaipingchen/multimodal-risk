#!/usr/bin/env python3
"""Extract confusion counts and decision items from VLM reasoning with Gemini."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "confusion_count": {
            "type": "integer",
            "minimum": 0,
            "description": "Number of expressions of confusion, uncertainty, or self-correction",
        },
        "decision_items": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete items or objects mentioned as contributing to the decision",
        },
    },
    "required": ["confusion_count", "decision_items"],
}


def validate_analysis(result):
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    count = result.get("confusion_count")
    items = result.get("decision_items")
    if type(count) is not int or count < 0:
        raise ValueError("confusion_count must be a nonnegative integer")
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise ValueError("decision_items must be a list of strings")
    return {"confusion_count": count, "decision_items": items}


def get_reasoning_analysis_with_retry(client, base_prompt, model_name, max_retries=8):
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=base_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": RESPONSE_SCHEMA,
                },
            )
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                from json_repair import repair_json

                result = json.loads(repair_json(response.text))
            return validate_analysis(result)
        except Exception as exc:
            last_error = exc
            print(f"Attempt {attempt + 1}/{max_retries}: {exc}")
    raise RuntimeError(f"Failed to get valid analysis after {max_retries} attempts") from last_error


def build_prompt(reasoning, ground_truth, model_pred):
    return f"""
You are analyzing a reasoning chain from Video-Language Models to evaluate its quality and characteristics. Please analyze the following reasoning chain and provide the requested metrics.

GROUND TRUTH: {ground_truth}
MODEL PREDICTION: {model_pred}

REASONING CHAIN:
{reasoning}

Please analyze this reasoning chain and provide:

1. **Confusion Count**: Count the number of times the reasoning shows confusion, uncertainty, or self-correction. Look for phrases like:
   - "wait..."
   - "no let me think again"
   - "actually..."
   - "hold on..."
   - "let me reconsider"
   - "I'm confused"
   - "that doesn't seem right"
   - Similar expressions of uncertainty or backtracking

2. **Decision Items**: Extract ALL items, objects, or things that the model specifically mentions from the video as part of its reasoning process for reaching its conclusion. This includes:
   - Physical objects mentioned in the reasoning
   - Items that influenced the decision (whether risky or safe)
   - Specific things the model identified or considered
   - Objects that were part of the analysis
   - Any concrete items/things mentioned that contributed to the final decision
   - Both safe items (for non-risk scenarios) and dangerous items (for risk scenarios)

Provide your analysis in the requested JSON format.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-name", default="gemini-3.1-flash-lite")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--reasoning-column", default="thinking")
    parser.add_argument("--prediction-column", default="pred")
    args = parser.parse_args()
    if args.max_retries < 1:
        parser.error("--max-retries must be positive")
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("Use a separate output CSV to preserve the source predictions")

    df = pd.read_csv(args.input_csv, dtype={"video_id": str})
    required = {"video_id", "riskLabel", args.reasoning_column, args.prediction_column}
    missing = required.difference(df.columns)
    if missing:
        parser.error(f"Missing input columns: {', '.join(sorted(missing))}")
    if "model_output" in df.columns and "original_model_output" in df.columns:
        parser.error("Input already contains original_model_output; choose an unannotated input")

    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    analyses = []
    counts = []
    items = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing reasoning"):
        try:
            reasoning = row[args.reasoning_column]
            if pd.isna(reasoning) or not str(reasoning).strip():
                raise ValueError("Missing reasoning text")
            prompt = build_prompt(reasoning, row["riskLabel"], row[args.prediction_column])
            analysis = get_reasoning_analysis_with_retry(
                client, prompt, args.model_name, args.max_retries
            )
            analyses.append(json.dumps(analysis, ensure_ascii=False))
            counts.append(analysis["confusion_count"])
            items.append(json.dumps(analysis["decision_items"], ensure_ascii=False))
        except Exception as exc:
            print(f"Error processing video {row['video_id']}: {exc}")
            analyses.append("Error")
            counts.append(None)
            items.append(None)

    output = df.copy()
    if "model_output" in output.columns:
        output = output.rename(columns={"model_output": "original_model_output"})
    output["model_output"] = analyses
    output["confusion_count"] = pd.array(counts, dtype="Int64")
    output["decision_items"] = items
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
