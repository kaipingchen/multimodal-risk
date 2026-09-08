#!/usr/bin/env python3
"""Calculate TRD and SCD using RiskCueBench paper equations (4) and (5).

Scores use positive-yes F1 on a 0–1 scale. TRD aligns all four runs by clip ID
and excludes a clip from every run if any prediction cannot be parsed.
SCD expects a judge-extracted confusion_count alongside the original prediction.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from calculate_accuracy import extract_prediction


def read_predictions(path, id_column):
    df = pd.read_csv(path, dtype={id_column: str})
    required = {id_column, "riskLabel", "model_output"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: required columns are {sorted(required)}")
    if df[id_column].isna().any() or df[id_column].duplicated().any():
        raise ValueError(f"{path}: clip IDs must be nonempty and unique")
    df = df.set_index(id_column)
    df["riskLabel"] = df["riskLabel"].astype("string").str.strip().str.lower()
    if not df["riskLabel"].isin(["yes", "no"]).all():
        raise ValueError(f"{path}: every riskLabel must be yes/no")
    df["prediction"] = df["model_output"].apply(extract_prediction)
    return df


def f1(df):
    return float(f1_score(df["riskLabel"].tolist(), df["prediction"].tolist(),
                          pos_label="yes", zero_division=0)) if len(df) else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-column", default="video_id")
    subs = parser.add_subparsers(dest="metric", required=True)
    trd = subs.add_parser("trd")
    for name in ["original", "shuffled", "swapped", "reversed"]:
        trd.add_argument(f"--{name}", required=True)
    scd = subs.add_parser("scd")
    scd.add_argument("--input", required=True)
    args = parser.parse_args()
    if args.metric == "trd":
        names = ["original", "shuffled", "swapped", "reversed"]
        runs = {name: read_predictions(getattr(args, name), args.id_column) for name in names}
        base = runs["original"]
        valid = base["prediction"].notna()
        for name in names[1:]:
            df = runs[name]
            if set(df.index) != set(base.index):
                raise ValueError(f"{name}: clip IDs differ from original")
            df = df.reindex(base.index)
            if not df["riskLabel"].equals(base["riskLabel"]):
                raise ValueError(f"{name}: ground-truth labels differ from original")
            runs[name] = df
            valid &= df["prediction"].notna()
        if not valid.any():
            raise ValueError("No clips with valid predictions in every run")
        scores = {name: f1(df.loc[valid]) for name, df in runs.items()}
        report = {"f1": scores, "TRD": sum(abs(scores["original"] - scores[n]) for n in names[1:]) / 3}
    else:
        base = read_predictions(args.input, args.id_column)
        counts = pd.to_numeric(base["confusion_count"], errors="raise")
        if not (np.isfinite(counts) & (counts >= 0) & (counts % 1 == 0)).all():
            raise ValueError("confusion_count must be a nonnegative integer for every clip")
        valid = base["prediction"].notna()
        if not valid.any():
            raise ValueError("No valid predictions")
        df = base.loc[valid]
        has_correction = counts.loc[valid] > 0
        plus, minus = df.loc[has_correction], df.loc[~has_correction]
        report = {"with_correction": len(plus), "without_correction": len(minus),
                  "f1_with_correction": f1(plus), "f1_without_correction": f1(minus),
                  "SCD": (len(plus) * f1(plus) - len(minus) * f1(minus)) / len(df)}
    report.update(evaluated=int(valid.sum()), excluded=int((~valid).sum()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
