"""
Matching utilities for the fingerprint authentication system.

This module loads enrolled templates and provides helper functions to
compare query fingerprints against the enrolled data.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from preprocessing import load_image, preprocess_pipeline
from feature_extraction import extract_features, minutiae_to_features

DEFAULT_TEMPLATE_DIR = "database/templates"


@dataclass
class TemplateData:
    """In-memory representation of a fingerprint template."""

    person_id: str
    minutiae: np.ndarray  # shape: (N, 4) -> x, y, orientation, type
    metadata: Dict


def ensure_templates_loaded(
    template_dir: str = DEFAULT_TEMPLATE_DIR,
) -> Dict[str, TemplateData]:
    """Load all template files from disk."""
    templates: Dict[str, TemplateData] = {}
    if not os.path.exists(template_dir):
        return templates

    for filename in os.listdir(template_dir):
        if not filename.lower().endswith(".json"):
            continue
        path = os.path.join(template_dir, filename)
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        records = payload.get("minutiae", [])
        minutiae = records_to_array(records)
        person_id = payload.get("person_id") or os.path.splitext(filename)[0]
        templates[person_id] = TemplateData(
            person_id=person_id,
            minutiae=minutiae,
            metadata=payload.get("metadata", {}),
        )
    return templates


def records_to_array(records: List[Dict[str, float]]) -> np.ndarray:
    """Convert stored minutiae records to numpy array form."""
    if not records:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(
        [[rec["x"], rec["y"], rec["orientation"], rec["type"]] for rec in records],
        dtype=np.float32,
    )


def center_minutiae(minutiae: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center minutiae around their centroid for translation invariance."""
    if len(minutiae) == 0:
        return minutiae, np.zeros(2, dtype=np.float32)
    centroid = minutiae[:, :2].mean(axis=0)
    centered = minutiae.copy()
    centered[:, :2] -= centroid
    return centered, centroid


def orientation_difference(angle_a: float, angle_b: float) -> float:
    """Return smallest absolute difference between two orientations."""
    diff = abs(angle_a - angle_b) % math.pi
    return min(diff, math.pi - diff)


def match_minutiae(
    template_minutiae: np.ndarray,
    query_minutiae: np.ndarray,
    distance_threshold: float = 15.0,
    orientation_threshold: float = 0.5,
    require_type_match: bool = True,
) -> Tuple[float, int]:
    """
    Compute similarity score between template and query minutiae sets.

    Returns:
        Tuple of (score, matched_pairs)
    """
    if len(template_minutiae) == 0 or len(query_minutiae) == 0:
        return 0.0, 0

    template_centered, _ = center_minutiae(template_minutiae)
    query_centered, _ = center_minutiae(query_minutiae)

    matched_template_indices: set[int] = set()
    matches = 0

    for q in query_centered:
        best_idx = -1
        best_distance = distance_threshold
        for idx, t in enumerate(template_centered):
            if idx in matched_template_indices:
                continue
            if require_type_match and q[3] != t[3]:
                continue

            distance = np.linalg.norm(q[:2] - t[:2])
            if distance > distance_threshold:
                continue

            orient_diff = orientation_difference(q[2], t[2])
            if orient_diff > orientation_threshold:
                continue

            if distance < best_distance:
                best_distance = distance
                best_idx = idx

        if best_idx >= 0:
            matched_template_indices.add(best_idx)
            matches += 1

    score = matches / max(len(template_centered), len(query_centered))
    return score, matches


def identify(
    query_minutiae: np.ndarray,
    templates: Dict[str, TemplateData],
    threshold: float = 0.4,
    **match_kwargs,
) -> Tuple[Optional[str], float]:
    """
    Identify the most likely person for the given query minutiae.

    Returns:
        (best_person_id or None, best_score)
    """
    best_score = 0.0
    best_person: Optional[str] = None

    for person_id, template in templates.items():
        score, _ = match_minutiae(template.minutiae, query_minutiae, **match_kwargs)
        if score > best_score:
            best_score = score
            best_person = person_id

    if best_score < threshold:
        return None, best_score
    return best_person, best_score


def process_query_image(
    image_path: str,
    preprocess_kwargs: Optional[Dict] = None,
    feature_kwargs: Optional[Dict] = None,
) -> np.ndarray:
    """Load, preprocess, and extract minutiae array from an image."""
    preprocess_kwargs = preprocess_kwargs or {}
    feature_kwargs = feature_kwargs or {}

    image = load_image(image_path)
    processed = preprocess_pipeline(image, **preprocess_kwargs)
    features = extract_features(processed, **feature_kwargs)
    feature_array = minutiae_to_features(features["minutiae"])
    return feature_array


def match_image(
    image_path: str,
    templates: Dict[str, TemplateData],
    threshold: float = 0.4,
    preprocess_kwargs: Optional[Dict] = None,
    feature_kwargs: Optional[Dict] = None,
    **match_kwargs,
) -> Tuple[Optional[str], float]:
    """Convenience helper that processes an image and performs identification."""
    query_minutiae = process_query_image(
        image_path,
        preprocess_kwargs=preprocess_kwargs,
        feature_kwargs=feature_kwargs,
    )
    return identify(query_minutiae, templates, threshold=threshold, **match_kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Match a fingerprint image.")
    parser.add_argument("image_path", help="Path to the fingerprint image to match.")
    parser.add_argument(
        "--template-dir",
        default=DEFAULT_TEMPLATE_DIR,
        help="Directory containing enrollment templates.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Similarity threshold required for a positive match.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(args.image_path)

    templates = ensure_templates_loaded(args.template_dir)
    if not templates:
        raise RuntimeError(
            f"No templates found in {args.template_dir}. "
            "Run enrollment.py before matching."
        )

    person_id, score = match_image(args.image_path, templates, threshold=args.threshold)
    if person_id:
        print(f"[MATCH] Identified person {person_id} with score {score:.3f}")
    else:
        print(f"[NO MATCH] Best score {score:.3f} below threshold {args.threshold}")

