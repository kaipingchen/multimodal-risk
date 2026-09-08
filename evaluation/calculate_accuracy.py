#!/usr/bin/env python3
"""
Script to calculate model accuracy from results.csv
"""

import argparse
import pandas as pd
import re
from collections import Counter


def extract_prediction(model_output: str) -> str | None:
    """
    Extract the Yes/No prediction from model output.
    Handles various formats including <answer> tags, \boxed{}, and plain text.
    """
    if pd.isna(model_output) or str(model_output).strip().lower() == "error":
        return None
    
    output = str(model_output).strip()
    
    # Try to extract from <answer> tags first
    answer_match = re.search(r"<answer>\s*(yes|no)\s*", output, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).lower()
    
    # Try to extract from \boxed{}
    boxed_match = re.search(r"\\boxed\{(yes|no)\}", output, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).lower()
    
    # Try to find standalone Yes/No at the end after </analysis> or </think>
    # Remove all tags and look for the final Yes/No
    clean_output = re.sub(r"<[^>]+>", " ", output)
    clean_output = re.sub(r"\\boxed\{[^}]*\}", " ", clean_output)
    
    # Look for Yes/No as whole words
    matches = re.findall(r"\b(yes|no)\b", clean_output, re.IGNORECASE)
    if matches:
        # Return the last match (usually the final answer)
        return matches[-1].lower()
    
    return None


def main():
    # Read the CSV file
    parser = argparse.ArgumentParser(description="Risk prediction accuracy and positive-yes F1")
    parser.add_argument("--input", required=True, help="CSV with riskLabel and model_output")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    required = {"riskLabel", "model_output"}
    if not required.issubset(df.columns):
        raise ValueError("Input must contain riskLabel and model_output")
    labels = df["riskLabel"].astype("string").str.strip().str.lower()
    if not labels.isin(["yes", "no"]).all():
        raise ValueError("riskLabel must be yes/no for every row; resolve missing labels before scoring")
    
    print(f"Total rows in dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 50)
    
    # Extract predictions
    df["predicted"] = df["model_output"].apply(extract_prediction)
    
    # Normalize ground truth labels
    df["ground_truth"] = df["riskLabel"].str.lower().str.strip()
    
    # Count parsing results
    valid_predictions = df["predicted"].notna()
    print(f"Valid predictions: {valid_predictions.sum()}")
    print(f"Failed to parse: {(~valid_predictions).sum()}")
    
    # Filter to valid predictions only
    valid_df = df[valid_predictions].copy()
    
    if len(valid_df) == 0:
        print("No valid predictions to evaluate!")
        return
    
    # Calculate accuracy
    correct = (valid_df["predicted"] == valid_df["ground_truth"]).sum()
    total = len(valid_df)
    accuracy = correct / total * 100
    
    print("-" * 50)
    print(f"Correct predictions: {correct}")
    print(f"Total valid predictions: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Breakdown by class
    print("-" * 50)
    print("Breakdown by ground truth class:")
    
    for label in ["yes", "no"]:
        subset = valid_df[valid_df["ground_truth"] == label]
        if len(subset) > 0:
            correct_in_class = (subset["predicted"] == subset["ground_truth"]).sum()
            class_accuracy = correct_in_class / len(subset) * 100
            print(f"  {label.upper()}: {correct_in_class}/{len(subset)} correct ({class_accuracy:.2f}%)")
    
    # Confusion matrix style breakdown
    print("-" * 50)
    print("Prediction distribution:")
    pred_counts = Counter(valid_df["predicted"])
    gt_counts = Counter(valid_df["ground_truth"])
    print(f"  Ground truth - YES: {gt_counts.get('yes', 0)}, NO: {gt_counts.get('no', 0)}")
    print(f"  Predicted    - YES: {pred_counts.get('yes', 0)}, NO: {pred_counts.get('no', 0)}")
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = ((valid_df["ground_truth"] == "yes") & (valid_df["predicted"] == "yes")).sum()
    fp = ((valid_df["ground_truth"] == "no") & (valid_df["predicted"] == "yes")).sum()
    tn = ((valid_df["ground_truth"] == "no") & (valid_df["predicted"] == "no")).sum()
    fn = ((valid_df["ground_truth"] == "yes") & (valid_df["predicted"] == "no")).sum()
    
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Negatives (FN): {fn}")
    
    # Precision, Recall, F1 for "yes" class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 50)
    print("Metrics for 'yes' (risk) class:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Show some examples of failed parses if any
    failed_df = df[~valid_predictions]
    if len(failed_df) > 0:
        print("-" * 50)
        print(f"Examples of failed parses ({min(5, len(failed_df))} shown):")
        for idx, row in failed_df.head(5).iterrows():
            output_preview = str(row["model_output"])[:100]
            print(f"  Row {idx}: {output_preview}...")


if __name__ == "__main__":
    main()

