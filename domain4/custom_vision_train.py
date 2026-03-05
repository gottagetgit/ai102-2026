"""
custom_vision_train.py
======================
Demonstrates the full Custom Vision model training workflow using the
Azure Cognitive Services Custom Vision Training SDK.

Covers the complete pipeline:
    1. Create a Custom Vision Training client
    2. Create a new project (classification or object detection)
    3. Create tags (classification labels)
    4. Upload and tag training images
    5. Trigger a training iteration
    6. Poll until training completes
    7. Evaluate model metrics: precision, recall, and average precision (AP)

Project types supported:
    - Classification (Multiclass or Multilabel)
    - Object Detection (with bounding boxes)

Exam Skill Mapping:
    - "Choose between image classification and object detection models"
    - "Label images"
    - "Train a custom image model"
    - "Evaluate custom vision model metrics"

Required Environment Variables (.env):
    CUSTOM_VISION_TRAINING_ENDPOINT  - e.g. https://<resource>.cognitiveservices.azure.com/
    CUSTOM_VISION_TRAINING_KEY       - Training resource key
    CUSTOM_VISION_PROJECT_NAME       - Name for the project (will be created if not exists)

Install:
    pip install azure-cognitiveservices-vision-customvision python-dotenv requests
"""

import os
import io
import time
import uuid
import requests
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageUrlCreateEntry,
    ImageCreateSummary,
    Region,
)

load_dotenv()

TRAINING_ENDPOINT  = os.environ.get("CUSTOM_VISION_TRAINING_ENDPOINT")
TRAINING_KEY       = os.environ.get("CUSTOM_VISION_TRAINING_KEY")
PROJECT_NAME       = os.environ.get("CUSTOM_VISION_PROJECT_NAME", "AI102-Demo-Project")

# ---------------------------------------------------------------------------
# Sample image URLs with their labels
# Using publicly available images grouped by category.
# In a real project, use your own domain-specific images.
# ---------------------------------------------------------------------------

# Image Classification samples (2 classes: cat, dog)
CLASSIFICATION_SAMPLES = {
    "cat": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/320px-Kittyply_edit1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/320px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
    ],
    "dog": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/320px-Dog_Breeds.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Tibetan_Mastiff2.jpg/320px-Tibetan_Mastiff2.jpg",
    ],
}

# Minimum images per tag for a successful training run
MINIMUM_IMAGES_PER_TAG = 5
# NOTE: For a real project, Azure requires at least 5 images per tag.
# The sample URLs above only have 3 each; supplement with your own images.


def get_training_client() -> CustomVisionTrainingClient:
    """Create and return a CustomVisionTrainingClient."""
    if not TRAINING_ENDPOINT or not TRAINING_KEY:
        raise ValueError(
            "CUSTOM_VISION_TRAINING_ENDPOINT and CUSTOM_VISION_TRAINING_KEY must be set in .env"
        )
    credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    return CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)


def create_or_get_project(trainer: CustomVisionTrainingClient, name: str, project_type: str = "Classification"):
    """Find an existing project by name or create a new one.

    Args:
        trainer:      CustomVisionTrainingClient
        name:         Project name
        project_type: "Classification" or "ObjectDetection"

    Returns:
        Project object
    """
    # Check for existing projects
    existing = trainer.get_projects()
    for proj in existing:
        if proj.name == name:
            print(f"  Found existing project: '{name}' (id={proj.id})")
            return proj

    # Determine the classification type domain
    # "General" is a good starting point; domain-specific options available
    domains = trainer.get_domains()

    if project_type == "Classification":
        # Compact domains support export; standard domains have higher accuracy
        target_domain = next(
            (d for d in domains if d.name == "General" and not d.exportable),
            domains[0],
        )
        project = trainer.create_project(
            name=name,
            domain_id=target_domain.id,
            classification_type="Multiclass",  # or "Multilabel" for multiple tags per image
        )
    else:
        # Object detection domain
        target_domain = next(
            (d for d in domains if "Object" in d.name and not d.exportable),
            domains[0],
        )
        project = trainer.create_project(
            name=name,
            domain_id=target_domain.id,
        )

    print(f"  Created project: '{name}' (id={project.id})")
    print(f"  Domain: {target_domain.name} (exportable={target_domain.exportable})")
    return project


def create_tags(trainer: CustomVisionTrainingClient, project_id: str, tag_names: list) -> dict:
    """Create classification tags in the project.

    Checks for existing tags to avoid duplicates.

    Args:
        trainer:    CustomVisionTrainingClient
        project_id: UUID of the project
        tag_names:  List of tag name strings

    Returns:
        Dict mapping tag_name → Tag object
    """
    existing_tags = {t.name: t for t in trainer.get_tags(project_id)}
    tag_map = {}

    for name in tag_names:
        if name in existing_tags:
            tag_map[name] = existing_tags[name]
            print(f"  Tag already exists: '{name}' (id={existing_tags[name].id})")
        else:
            tag = trainer.create_tag(project_id, name)
            tag_map[name] = tag
            print(f"  Created tag: '{name}' (id={tag.id})")

    return tag_map


def upload_images_from_urls(
    trainer: CustomVisionTrainingClient,
    project_id: str,
    tag_to_urls: dict,
    tag_map: dict,
) -> int:
    """Upload images from URLs and assign tags.

    Processes in batches of 64 (API limit per call).

    Args:
        trainer:      CustomVisionTrainingClient
        project_id:   UUID of the project
        tag_to_urls:  Dict mapping tag_name → list of image URLs
        tag_map:      Dict mapping tag_name → Tag object

    Returns:
        Total number of successfully uploaded images
    """
    all_entries = []
    for tag_name, urls in tag_to_urls.items():
        tag_id = tag_map[tag_name].id
        for url in urls:
            all_entries.append(
                ImageUrlCreateEntry(url=url, tag_ids=[tag_id])
            )

    # Upload in batches of 64
    batch_size = 64
    total_ok = 0
    total_fail = 0

    for i in range(0, len(all_entries), batch_size):
        batch = all_entries[i : i + batch_size]
        print(f"  Uploading batch of {len(batch)} images...")
        summary: ImageCreateSummary = trainer.create_images_from_urls(
            project_id=project_id,
            images=batch,
        )
        ok   = sum(1 for r in summary.images if r.status in ("OK", "OKDuplicate"))
        fail = sum(1 for r in summary.images if r.status not in ("OK", "OKDuplicate"))
        total_ok   += ok
        total_fail += fail

        for img_result in summary.images:
            if img_result.status not in ("OK", "OKDuplicate"):
                print(f"    Failed: {img_result.source_url} — {img_result.status}")

    print(f"  Upload complete: {total_ok} succeeded, {total_fail} failed")
    return total_ok


def upload_images_from_bytes(
    trainer: CustomVisionTrainingClient,
    project_id: str,
    tag_to_paths: dict,
    tag_map: dict,
) -> int:
    """Upload images from local file paths (alternative to URL upload).

    Args:
        trainer:        CustomVisionTrainingClient
        project_id:     UUID of the project
        tag_to_paths:   Dict mapping tag_name → list of local file paths
        tag_map:        Dict mapping tag_name → Tag object

    Returns:
        Total number of successfully uploaded images
    """
    total_ok = 0
    for tag_name, paths in tag_to_paths.items():
        tag_id = tag_map[tag_name].id
        for path in paths:
            with open(path, "rb") as f:
                image_bytes = f.read()
            summary = trainer.create_images_from_data(
                project_id=project_id,
                image_data=image_bytes,
                tag_ids=[tag_id],
            )
            if summary.is_batch_successful:
                total_ok += 1
                print(f"  Uploaded: {os.path.basename(path)}")
            else:
                print(f"  Failed: {path}")
    return total_ok


def train_model(trainer: CustomVisionTrainingClient, project_id: str, poll_interval: int = 10):
    """Trigger training and poll until the iteration is complete.

    Training is asynchronous. We poll every `poll_interval` seconds until
    the iteration status becomes "Completed" or "Failed".

    Args:
        trainer:       CustomVisionTrainingClient
        project_id:    UUID of the project
        poll_interval: Seconds between status checks

    Returns:
        Completed Iteration object
    """
    print("\n  Starting training iteration...")
    iteration = trainer.train_project(project_id)
    print(f"  Iteration id: {iteration.id}")

    while iteration.status not in ("Completed", "Failed"):
        print(f"  Training status: {iteration.status} — waiting {poll_interval}s...")
        time.sleep(poll_interval)
        iteration = trainer.get_iteration(project_id, iteration.id)

    if iteration.status == "Failed":
        raise RuntimeError(f"Training failed for iteration {iteration.id}")

    print(f"  Training completed! Iteration: {iteration.id}, status: {iteration.status}")
    return iteration


def evaluate_model(
    trainer: CustomVisionTrainingClient, project_id: str, iteration_id: str
) -> None:
    """Display model performance metrics from the completed training iteration.

    Key metrics:
        Precision:  Of images predicted as this tag, how many were correct?
                    High precision = few false positives.
        Recall:     Of images actually this tag, how many were found?
                    High recall = few false negatives.
        AP:         Area under the Precision-Recall curve (0–1).
                    AP > 0.9 indicates an excellent model.

    Args:
        trainer:      CustomVisionTrainingClient
        project_id:   UUID of the project
        iteration_id: UUID of the completed iteration
    """
    performance = trainer.get_iteration_performance(
        project_id=project_id,
        iteration_id=iteration_id,
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)

    # Overall (macro-averaged) metrics
    print(f"\nOverall Metrics:")
    print(f"  Precision:  {performance.precision:.4f}  ({performance.precision:.2%})")
    print(f"  Recall:     {performance.recall:.4f}    ({performance.recall:.2%})")
    print(f"  AP:         {performance.average_precision:.4f}")

    # Threshold used for these metrics
    print(f"  Threshold:  {performance.precision_threshold:.2f} (probability cutoff)")

    # Per-tag metrics
    if performance.per_tag_performance:
        print(f"\nPer-Tag Performance:")
        print(f"  {'Tag':<20} {'Precision':>10} {'Recall':>10} {'AP':>10} {'Images':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for tag_perf in performance.per_tag_performance:
            print(
                f"  {tag_perf.name:<20} "
                f"{tag_perf.precision:>10.4f} "
                f"{tag_perf.recall:>10.4f} "
                f"{tag_perf.average_precision:>10.4f} "
                f"{tag_perf.image_count:>8}"
            )

    # Quality guidance
    print("\nMetrics Interpretation:")
    p = performance.precision
    r = performance.recall
    ap = performance.average_precision
    if ap >= 0.9:
        print("  AP >= 0.90: Excellent model — ready for production consideration")
    elif ap >= 0.75:
        print("  AP >= 0.75: Good model — add more diverse training images to improve")
    elif ap >= 0.5:
        print("  AP >= 0.50: Fair model — needs more training data and diversity")
    else:
        print("  AP < 0.50:  Poor model — review data quality, labelling, and balance")

    if abs(p - r) > 0.2:
        print("  Large precision-recall gap: adjust probability threshold or balance data")

    print("=" * 60)


def run_classification_training_demo():
    """Run the full classification training pipeline."""
    print("[1/5] Connecting to Custom Vision...")
    trainer = get_training_client()
    print(f"  Endpoint: {TRAINING_ENDPOINT}")

    print("\n[2/5] Creating/finding project...")
    project = create_or_get_project(trainer, PROJECT_NAME, "Classification")

    print("\n[3/5] Creating tags...")
    tag_names = list(CLASSIFICATION_SAMPLES.keys())
    tag_map = create_tags(trainer, project.id, tag_names)

    print("\n[4/5] Uploading training images...")
    total = upload_images_from_urls(trainer, project.id, CLASSIFICATION_SAMPLES, tag_map)
    print(f"  Total images uploaded: {total}")

    if total < len(tag_names) * MINIMUM_IMAGES_PER_TAG:
        print(
            f"\n  WARNING: Fewer than {MINIMUM_IMAGES_PER_TAG} images per tag. "
            f"Training requires at least {MINIMUM_IMAGES_PER_TAG} images per tag.\n"
            f"  Add more images before calling train_project."
        )
        # In a demo context we still attempt to train
        # to show the full workflow structure.

    print("\n[5/5] Training model...")
    iteration = train_model(trainer, project.id)

    evaluate_model(trainer, project.id, str(iteration.id))

    print(f"\nProject ID (save this for prediction): {project.id}")
    print(f"Iteration ID: {iteration.id}")
    return project.id, str(iteration.id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure Custom Vision — Training Demo ===\n")
    try:
        project_id, iteration_id = run_classification_training_demo()
        print(f"\nSave these values for custom_vision_predict.py:")
        print(f"  CUSTOM_VISION_PROJECT_ID={project_id}")
        print(f"  CUSTOM_VISION_ITERATION_NAME=<name after publishing>")
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
