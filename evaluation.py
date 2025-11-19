"""
Evaluation utilities for the fingerprint authentication system.

This module measures system performance (accuracy, FAR, FRR, etc.) using
the enrolled templates and a query dataset (validation/test).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from preprocessing import load_image, preprocess_pipeline
from feature_extraction import extract_features, minutiae_to_features
from matching import ensure_templates_loaded, identify
from utils import parse_filename

DEFAULT_TEMPLATE_DIR = "database/templates"
DEFAULT_VALIDATE_DIR = "project-data/Project-Data/validate"
DEFAULT_TEST_DIR = "project-data/Project-Data/test"
EVALUATION_OUTPUT_DIR = "evaluation_output"


@dataclass
class EvaluationResult:
    filename: str
    actual_id: Optional[str]
    predicted_id: Optional[str]
    score: float


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def process_image_to_features(
    image_path: str,
    preprocess_kwargs: Optional[Dict] = None,
    feature_kwargs: Optional[Dict] = None,
):
    preprocess_kwargs = preprocess_kwargs or {}
    feature_kwargs = feature_kwargs or {}

    image = load_image(image_path)
    processed = preprocess_pipeline(image, **preprocess_kwargs)
    features = extract_features(processed, **feature_kwargs)
    return minutiae_to_features(features["minutiae"])


def evaluate_directory(
    template_dir: str = DEFAULT_TEMPLATE_DIR,
    query_dir: str = DEFAULT_VALIDATE_DIR,
    threshold: float = 0.4,
    max_queries: Optional[int] = None,
    output_dir: str = EVALUATION_OUTPUT_DIR,
    preprocess_kwargs: Optional[Dict] = None,
    feature_kwargs: Optional[Dict] = None,
    match_kwargs: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Evaluate system performance against the specified query directory.

    Returns:
        Dictionary containing evaluation metrics.
    """
    ensure_directory(output_dir)
    templates = ensure_templates_loaded(template_dir)
    if not templates:
        raise RuntimeError("No templates available. Run enrollment first.")

    enrolled_ids = set(templates.keys())
    preprocess_kwargs = preprocess_kwargs or {}
    feature_kwargs = feature_kwargs or {}
    match_kwargs = match_kwargs or {}

    results: List[EvaluationResult] = []
    filenames = sorted(f for f in os.listdir(query_dir) if f.lower().endswith(".bmp"))

    if max_queries is not None:
        filenames = filenames[:max_queries]

    for filename in filenames:
        image_path = os.path.join(query_dir, filename)
        person_id, _ = parse_filename(filename)
        query_features = process_image_to_features(
            image_path,
            preprocess_kwargs=preprocess_kwargs,
            feature_kwargs=feature_kwargs,
        )
        predicted_id, score = identify(
            query_features, templates, threshold=threshold, **match_kwargs
        )
        results.append(
            EvaluationResult(
                filename=filename,
                actual_id=person_id,
                predicted_id=predicted_id,
                score=score,
            )
        )

    metrics = compute_metrics(results, enrolled_ids)
    save_evaluation_results(results, metrics, output_dir, os.path.basename(query_dir))
    plot_metrics(metrics, output_dir, os.path.basename(query_dir))
    return metrics


def compute_metrics(
    results: List[EvaluationResult], enrolled_ids: set[str]
) -> Dict[str, float]:
    """Calculate accuracy, FAR, and FRR from evaluation results."""
    genuine_attempts = 0
    impostor_attempts = 0
    true_accepts = 0
    false_rejects = 0
    true_rejects = 0
    false_accepts = 0

    for result in results:
        actual_id = result.actual_id
        predicted_id = result.predicted_id
        if actual_id in enrolled_ids:
            genuine_attempts += 1
            if predicted_id == actual_id:
                true_accepts += 1
            else:
                false_rejects += 1
        else:
            impostor_attempts += 1
            if predicted_id is None:
                true_rejects += 1
            else:
                false_accepts += 1

    total_attempts = genuine_attempts + impostor_attempts
    accuracy = (
        (true_accepts + true_rejects) / total_attempts if total_attempts else 0.0
    )
    far = false_accepts / impostor_attempts if impostor_attempts else 0.0
    frr = false_rejects / genuine_attempts if genuine_attempts else 0.0

    metrics = {
        "total_attempts": total_attempts,
        "genuine_attempts": genuine_attempts,
        "impostor_attempts": impostor_attempts,
        "true_accepts": true_accepts,
        "false_rejects": false_rejects,
        "true_rejects": true_rejects,
        "false_accepts": false_accepts,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
    }
    return metrics


def save_evaluation_results(
    results: List[EvaluationResult],
    metrics: Dict[str, float],
    output_dir: str,
    run_name: str,
) -> None:
    """Persist raw evaluation results and metrics for later analysis."""
    ensure_directory(output_dir)
    summary = {
        "run_name": run_name,
        "metrics": metrics,
        "results": [result.__dict__ for result in results],
    }
    output_path = os.path.join(output_dir, f"{run_name}_results.json")
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    metrics_path = os.path.join(output_dir, f"{run_name}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        for key, value in metrics.items():
            fp.write(f"{key}: {value}\n")
    print(f"[INFO] Saved evaluation results to {output_path}")


def plot_metrics(metrics: Dict[str, float], output_dir: str, run_name: str) -> None:
    """Create a simple bar chart for accuracy, FAR, and FRR."""
    ensure_directory(output_dir)
    labels = ["Accuracy", "FAR", "FRR"]
    values = [
        metrics.get("accuracy", 0.0),
        metrics.get("far", 0.0),
        metrics.get("frr", 0.0),
    ]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#2E86AB", "#C0392B", "#F1C40F"])
    plt.ylim(0, 1)
    plt.title(f"Evaluation Metrics - {run_name}")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel("Value")
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{run_name}_metrics.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved metric visualization to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fingerprint system.")
    parser.add_argument(
        "--template-dir",
        default=DEFAULT_TEMPLATE_DIR,
        help="Directory containing enrollment templates.",
    )
    parser.add_argument(
        "--query-dir",
        default=DEFAULT_VALIDATE_DIR,
        help="Directory containing query fingerprint images.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Similarity threshold for positive match.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit on number of query images to evaluate.",
    )
    args = parser.parse_args()

    evaluate_directory(
        template_dir=args.template_dir,
        query_dir=args.query_dir,
        threshold=args.threshold,
        max_queries=args.max_queries,
    )

