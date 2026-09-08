#!/usr/bin/env python3
"""Rank candidates by sample variance of model votes (from data.ipynb, cell 88).

Unlike the exploratory notebook, align by explicit clip ID and reject malformed
answers rather than interpreting every answer without a trailing 'yes' as 'no'.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from calculate_accuracy import extract_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--id-column", default="video_id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("At least two model result CSVs are required")
    result = None
    vote_columns = []
    for i, path in enumerate(args.inputs):
        df = pd.read_csv(path, dtype={args.id_column: str})
        if not {args.id_column, "model_output"}.issubset(df.columns):
            raise ValueError(f"{path}: missing ID or model_output column")
        if df[args.id_column].isna().any() or df[args.id_column].duplicated().any():
            raise ValueError(f"{path}: IDs must be nonempty and unique per clip")
        df = df.set_index(args.id_column)
        column = f"model_{i + 1}_vote"
        votes = df["model_output"].apply(extract_prediction).map({"yes": 1, "no": 0})
        if result is None:
            result = df.drop(columns="model_output").copy()
        elif set(result.index) != set(df.index):
            raise ValueError(f"{path}: every model must score the same clip IDs")
        if column in result.columns:
            raise ValueError(f"Input already contains reserved column {column}")
        result[column] = votes
        vote_columns.append(column)
        print(f"{column}: {path}")
    valid = result[vote_columns].notna().all(axis=1)
    print(f"Excluded {int((~valid).sum())} clips with an unparseable model answer")
    result = result.loc[valid].copy()
    if result.empty:
        raise ValueError("No clips with valid predictions from every model")
    result["model_disagreement_score"] = result[vote_columns].var(axis=1, ddof=1)
    result.sort_values("model_disagreement_score", ascending=False, kind="stable").to_csv(args.output)


if __name__ == "__main__":
    main()
