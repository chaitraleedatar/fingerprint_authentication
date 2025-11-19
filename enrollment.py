"""
Enrollment utilities for the fingerprint authentication system.

This module processes the training fingerprints, extracts minutiae-based
features, and stores them as reusable templates for matching.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from preprocessing import load_image, preprocess_pipeline
from feature_extraction import extract_features, minutiae_to_features
from utils import get_images_by_person

TEMPLATE_VERSION = "1.0"
DEFAULT_TRAIN_DIR = "project-data/Project-Data/train"
DEFAULT_TEMPLATE_DIR = "database/templates"


def ensure_directory(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def features_to_records(feature_array: np.ndarray) -> List[Dict[str, float]]:
    """Convert minutiae feature array into JSON-serializable records."""
    records: List[Dict[str, float]] = []
    for row in feature_array:
        records.append(
            {
                "x": float(row[0]),
                "y": float(row[1]),
                "orientation": float(row[2]),
                "type": int(row[3]),
            }
        )
    return records


def records_to_array(records: List[Dict[str, float]]) -> np.ndarray:
    """Convert stored records back to numpy array form."""
    if not records:
        return np.empty((0, 4), dtype=np.float32)
    array = np.array(
        [[rec["x"], rec["y"], rec["orientation"], rec["type"]] for rec in records],
        dtype=np.float32,
    )
    return array


def enroll_image(
    image_path: str,
    preprocess_kwargs: Optional[Dict] = None,
    feature_kwargs: Optional[Dict] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Enroll a single fingerprint image by extracting minutiae records.

    Returns:
        Tuple of (minutiae_records, stats_dict)
    """
    preprocess_kwargs = preprocess_kwargs or {}
    feature_kwargs = feature_kwargs or {}

    image = load_image(image_path)
    processed = preprocess_pipeline(image, **preprocess_kwargs)
    features = extract_features(processed, **feature_kwargs)
    feature_array = minutiae_to_features(features["minutiae"])

    records = features_to_records(feature_array)
    centroid = (
        feature_array[:, :2].mean(axis=0).tolist() if len(feature_array) else [0.0, 0.0]
    )

    stats = {
        "image_path": image_path,
        "minutiae_count": len(records),
        "centroid": centroid,
    }
    return records, stats


def build_template(
    person_id: str,
    minutiae_records: List[Dict[str, float]],
    samples: List[Dict[str, float]],
) -> Dict:
    """Assemble the template payload for persistence."""
    template = {
        "version": TEMPLATE_VERSION,
        "person_id": person_id,
        "minutiae": minutiae_records,
        "metadata": {
            "created_at": time.time(),
            "image_count": len(samples),
            "samples": samples,
            "total_minutiae": len(minutiae_records),
        },
    }
    return template


def save_template(person_id: str, template: Dict, template_dir: str) -> str:
    """Persist template to disk and return the file path."""
    ensure_directory(template_dir)
    template_path = os.path.join(template_dir, f"{person_id}.json")
    with open(template_path, "w", encoding="utf-8") as fp:
        json.dump(template, fp, indent=2)
    return template_path


def enroll_person(
    person_id: str,
    image_paths: Sequence[str],
    template_dir: str = DEFAULT_TEMPLATE_DIR,
    preprocess_kwargs: Optional[Dict] = None,
    feature_kwargs: Optional[Dict] = None,
) -> Optional[str]:
    """
    Enroll a single person and save their template.

    Returns:
        Path to the saved template or None if enrollment failed.
    """
    minutiae_records: List[Dict[str, float]] = []
    sample_stats: List[Dict[str, float]] = []

    for image_path in image_paths:
        try:
            records, stats = enroll_image(
                image_path,
                preprocess_kwargs=preprocess_kwargs,
                feature_kwargs=feature_kwargs,
            )
            if not records:
                continue
            minutiae_records.extend(records)
            sample_stats.append(stats)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to enroll {image_path}: {exc}")

    if not minutiae_records:
        print(f"[WARN] No minutiae recorded for person {person_id}; skipping template.")
        return None

    template = build_template(person_id, minutiae_records, sample_stats)
    return save_template(person_id, template, template_dir)


def enroll_all(
    train_dir: str = DEFAULT_TRAIN_DIR,
    template_dir: str = DEFAULT_TEMPLATE_DIR,
    limit: Optional[int] = None,
    preprocess_kwargs: Optional[Dict] = None,
    feature_kwargs: Optional[Dict] = None,
) -> List[str]:
    """
    Enroll all persons present in the training directory.

    Args:
        train_dir: Directory containing training fingerprint images.
        template_dir: Directory where templates will be stored.
        limit: Optional limit on number of persons to enroll.

    Returns:
        List of template file paths that were created.
    """
    ensure_directory(template_dir)
    images_by_person = get_images_by_person(train_dir)
    enrolled_paths: List[str] = []

    for idx, (person_id, filenames) in enumerate(sorted(images_by_person.items())):
        if limit is not None and idx >= limit:
            break

        image_paths = [os.path.join(train_dir, name) for name in filenames]
        print(f"[INFO] Enrolling person {person_id} ({len(image_paths)} images)")
        template_path = enroll_person(
            person_id,
            image_paths,
            template_dir=template_dir,
            preprocess_kwargs=preprocess_kwargs,
            feature_kwargs=feature_kwargs,
        )
        if template_path:
            enrolled_paths.append(template_path)

    print(f"[INFO] Enrollment complete. Templates created: {len(enrolled_paths)}")
    return enrolled_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enroll training fingerprints.")
    parser.add_argument(
        "--train-dir",
        default=DEFAULT_TRAIN_DIR,
        help="Directory containing training fingerprint images.",
    )
    parser.add_argument(
        "--template-dir",
        default=DEFAULT_TEMPLATE_DIR,
        help="Directory where templates will be stored.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of persons to enroll.",
    )
    args = parser.parse_args()

    enroll_all(
        train_dir=args.train_dir,
        template_dir=args.template_dir,
        limit=args.limit,
    )

