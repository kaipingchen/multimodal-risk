#!/usr/bin/env python3
"""Calculate per-sample Reasoning Grounding Accuracy (RGA).

The implementation follows the definition used in plots.ipynb and the RGA
equations supplied with this project:

1. Embed each model decision item and each annotated visual indicator with
   all-MiniLM-L6-v2.
2. Give every decision item its maximum cosine similarity to any indicator.
3. Fit a threshold that maximizes F1 between each sample's mean maximum
   similarity and its binary ``eval`` label (unless --threshold is supplied).
4. Set a sample's RGA to the percentage of its decision items whose maximum
   similarity is at least that threshold.

By default, the input CSV is updated in place with an ``RGA`` column and the
top/bottom 10% of rows are written to separate CSV files.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve


DEFAULT_INPUT = Path("data/gemini_reasoning_eval.csv")
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _patch_optional_kernels_mismatch() -> None:
    """Work around a transformers/kernels optional-dependency mismatch.

    Some environments have a transformers release that constructs kernel
    repositories without a revision while the installed kernels release
    requires one. MiniLM does not use those optional kernels, but the mismatch
    otherwise prevents sentence-transformers from importing at all. The patch
    is process-local and only supplies ``main`` to those registry objects.
    """

    try:
        import kernels
    except ImportError:
        return

    for class_name in ("LayerRepository", "FuncRepository"):
        base = getattr(kernels, class_name, None)
        if base is None:
            continue

        class CompatibleRepository(base):  # type: ignore[misc, valid-type]
            def __init__(
                self,
                repo_id: str,
                *,
                revision: str | None = None,
                version: int | None = None,
                **kwargs: Any,
            ) -> None:
                if revision is None and version is None:
                    revision = "main"
                super().__init__(
                    repo_id,
                    revision=revision,
                    version=version,
                    **kwargs,
                )

        CompatibleRepository.__name__ = class_name
        setattr(kernels, class_name, CompatibleRepository)


def load_sentence_transformer(model_name: str, device: str | None):
    _patch_optional_kernels_mismatch()
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "Could not import sentence-transformers. Install compatible versions "
            "of sentence-transformers, transformers, torch, and kernels."
        ) from exc

    return SentenceTransformer(model_name, device=device)


def parse_model_output(value: Any, row_number: int) -> list[str]:
    """Extract a clean decision_items list from a CSV model_output value."""

    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Row {row_number}: model_output is not valid JSON/Python data"
                ) from exc
    else:
        raise ValueError(f"Row {row_number}: model_output must contain an object")

    if not isinstance(parsed, dict):
        raise ValueError(f"Row {row_number}: model_output must contain an object")

    items = parsed.get("decision_items", [])
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"Row {row_number}: decision_items must be a list")
    return [str(item).strip() for item in items if str(item).strip()]


def parse_visual_indicators(value: Any) -> list[str]:
    """Parse an indicator list, falling back to the CSV's comma delimiter."""

    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (json.JSONDecodeError, SyntaxError, ValueError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]

    return [part.strip() for part in text.split(",") if part.strip()]


def encode_texts(model: Any, texts: list[str], kind: str, batch_size: int) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    method_name = "encode_query" if kind == "query" else "encode_document"
    encoder = getattr(model, method_name, model.encode)
    return np.asarray(
        encoder(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )


def calculate_max_similarities(
    model: Any,
    decisions_by_row: list[list[str]],
    indicators_by_row: list[list[str]],
    batch_size: int,
) -> list[np.ndarray]:
    """Return one maximum-cosine-similarity vector per input row."""

    all_decisions = [item for row in decisions_by_row for item in row]
    all_indicators = [item for row in indicators_by_row for item in row]

    decision_embeddings = encode_texts(model, all_decisions, "query", batch_size)
    indicator_embeddings = encode_texts(model, all_indicators, "document", batch_size)

    results: list[np.ndarray] = []
    decision_offset = 0
    indicator_offset = 0

    for decisions, indicators in zip(decisions_by_row, indicators_by_row):
        n_decisions = len(decisions)
        n_indicators = len(indicators)

        if n_decisions == 0:
            results.append(np.empty(0, dtype=np.float32))
        elif n_indicators == 0:
            results.append(np.zeros(n_decisions, dtype=np.float32))
        else:
            decision_block = decision_embeddings[
                decision_offset : decision_offset + n_decisions
            ]
            indicator_block = indicator_embeddings[
                indicator_offset : indicator_offset + n_indicators
            ]
            # The embeddings are normalized, so matrix multiplication is cosine
            # similarity. Each decision item keeps its best indicator match.
            similarities = decision_block @ indicator_block.T
            results.append(np.max(similarities, axis=1))

        decision_offset += n_decisions
        indicator_offset += n_indicators

    return results


def fit_f1_threshold(sample_scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(sample_scores) & np.isfinite(labels) & np.isin(labels, [0, 1])
    y_score = sample_scores[valid]
    y_true = labels[valid].astype(int)

    if y_score.size == 0:
        raise ValueError("No rows have both a valid relatedness score and binary eval label")
    if np.unique(y_true).size != 2:
        raise ValueError("F1 threshold fitting requires both eval=0 and eval=1 rows")

    _, _, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        raise ValueError("Could not derive candidate thresholds from the eval labels")

    f1_values = np.asarray(
        [
            f1_score(y_true, y_score >= threshold, zero_division=0)
            for threshold in thresholds
        ]
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def derived_subset_path(output_path: Path, side: str, percentile: float) -> Path:
    percentile_label = f"{percentile:g}".replace(".", "p")
    return output_path.with_name(
        f"{output_path.stem}_{side}_{percentile_label}pct_rga{output_path.suffix}"
    )


def write_csv_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        dataframe.to_csv(handle, index=False)
    os.replace(temporary_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV receiving the RGA column (default: update --input in place)",
    )
    parser.add_argument("--top-output", type=Path)
    parser.add_argument("--bottom-output", type=Path)
    parser.add_argument(
        "--percentile",
        type=float,
        default=10.0,
        help="Percentage of rows in each top/bottom export (default: 10)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", help="For example: cpu, cuda, or mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Use a fixed cosine threshold instead of fitting one from eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.percentile <= 50:
        raise ValueError("--percentile must be greater than 0 and at most 50")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    input_path = args.input.resolve()
    output_path = (args.output or args.input).resolve()
    top_path = (args.top_output or derived_subset_path(output_path, "top", args.percentile)).resolve()
    bottom_path = (
        args.bottom_output
        or derived_subset_path(output_path, "bottom", args.percentile)
    ).resolve()

    dataframe = pd.read_csv(input_path)
    required_columns = {"riskVisualIndicator", "model_output"}
    if args.threshold is None:
        required_columns.add("eval")
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(sorted(missing))}")

    decisions_by_row = [
        parse_model_output(value, row_number=index + 2)
        for index, value in enumerate(dataframe["model_output"])
    ]
    indicators_by_row = [
        parse_visual_indicators(value) for value in dataframe["riskVisualIndicator"]
    ]

    model = load_sentence_transformer(args.model, args.device)
    max_similarities = calculate_max_similarities(
        model,
        decisions_by_row,
        indicators_by_row,
        args.batch_size,
    )

    # s_i in the accompanying definition: mean of decision-item maxima.
    sample_scores = np.asarray(
        [float(np.mean(scores)) if scores.size else 0.0 for scores in max_similarities],
        dtype=float,
    )

    if args.threshold is None:
        labels = pd.to_numeric(dataframe["eval"], errors="coerce").to_numpy(dtype=float)
        threshold, best_f1 = fit_f1_threshold(sample_scores, labels)
        threshold_source = f"F1-optimized (F1={best_f1:.6f})"
    else:
        threshold = float(args.threshold)
        best_f1 = float("nan")
        threshold_source = "fixed from --threshold"

    rga_scores = np.asarray(
        [
            float(np.mean(scores >= threshold) * 100.0) if scores.size else 0.0
            for scores in max_similarities
        ]
    )
    dataframe["RGA"] = np.round(rga_scores, 6)

    # Select an exact number of rows per tail. Stable sorting makes tied RGA
    # values deterministic while keeping their original CSV order.
    subset_size = max(1, math.ceil(len(dataframe) * args.percentile / 100.0))
    bottom = dataframe.sort_values("RGA", ascending=True, kind="stable").head(subset_size)
    top = dataframe.sort_values("RGA", ascending=False, kind="stable").head(subset_size)

    write_csv_atomic(dataframe, output_path)
    write_csv_atomic(top, top_path)
    write_csv_atomic(bottom, bottom_path)

    print(f"Rows scored: {len(dataframe)}")
    print(f"Cosine threshold: {threshold:.9f} ({threshold_source})")
    print(f"Dataset mean RGA: {rga_scores.mean():.6f}%")
    print(f"RGA CSV: {output_path}")
    print(f"Top {args.percentile:g}% ({len(top)} rows): {top_path}")
    print(f"Bottom {args.percentile:g}% ({len(bottom)} rows): {bottom_path}")


if __name__ == "__main__":
    main()
