"""
custom_vision_train.py
======================
Full lifecycle demo for Azure Custom Vision:
  1. Create a project (classification or object detection)
  2. Create tags
  3. Upload images with tags (classification) or regions (object detection)
  4. Train an iteration
  5. Evaluate performance (precision, recall, AP)
  6. Publish the iteration to a prediction resource
  7. Export the model (ONNX, TensorFlow, CoreML)

Exam Skill Mapping:
    - "Train an image classification model by using Azure Custom Vision"
    - "Train an object detection model by using Azure Custom Vision"
    - "Evaluate model performance metrics (precision, recall, mAP)"
    - "Publish a Custom Vision model iteration"
    - "Export a Custom Vision model for use at the edge"

Required Environment Variables (.env):
    CUSTOM_VISION_TRAINING_ENDPOINT    - e.g. https://<resource>.cognitiveservices.azure.com/
    CUSTOM_VISION_TRAINING_KEY         - training key
    CUSTOM_VISION_PREDICTION_RESOURCE_ID - full ARM resource ID of prediction resource

Install:
    pip install azure-cognitiveservices-vision-customvision python-dotenv requests
"""

import os
import time
import requests
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageUrlCreateEntry,
    ImageUrlCreateBatch,
    Region,
    ImageFileCreateEntry,
    ImageFileCreateBatch,
    Export,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAINING_ENDPOINT    = os.environ["CUSTOM_VISION_TRAINING_ENDPOINT"]
TRAINING_KEY         = os.environ["CUSTOM_VISION_TRAINING_KEY"]
PREDICTION_RESOURCE_ID = os.environ["CUSTOM_VISION_PREDICTION_RESOURCE_ID"]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_training_client() -> CustomVisionTrainingClient:
    creds = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    return CustomVisionTrainingClient(TRAINING_ENDPOINT, creds)


# ---------------------------------------------------------------------------
# 1. Image Classification Project
# ---------------------------------------------------------------------------

def demo_classification_project(trainer: CustomVisionTrainingClient) -> None:
    """
    Creates a multi-class classification project, uploads tagged images,
    trains, evaluates, and publishes an iteration.
    """
    print("\n" + "="*60)
    print("Classification Project Demo")
    print("="*60)

    # --- Create project ---
    project = trainer.create_project(
        name="AI102-Classification-Demo",
        description="Exam demo: classify cats vs dogs",
        classification_type="Multiclass",  # or "Multilabel"
        domain_type="General",  # "General", "General [A1]", "Food", etc.
    )
    print(f"Created project: {project.name} (id={project.id})")

    # --- Create tags ---
    tag_cat = trainer.create_tag(project.id, "cat")
    tag_dog = trainer.create_tag(project.id, "dog")
    print(f"Created tags: {tag_cat.name}, {tag_dog.name}")

    # --- Upload images from URLs ---
    # In a real scenario you'd have many labelled images per tag
    cat_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/320px-Kittyply_edit1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
    ]
    dog_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
    ]

    cat_entries = [ImageUrlCreateEntry(url=u, tag_ids=[tag_cat.id]) for u in cat_urls]
    dog_entries = [ImageUrlCreateEntry(url=u, tag_ids=[tag_dog.id]) for u in dog_urls]

    upload_result = trainer.create_images_from_urls(
        project.id, ImageUrlCreateBatch(images=cat_entries + dog_entries)
    )
    print(f"Uploaded {len(upload_result.images)} images; "
          f"is_batch_successful={upload_result.is_batch_successful}")

    # --- Train ---
    print("Training... (this may take a minute)")
    iteration = trainer.train_project(project.id)
    while iteration.status != "Completed":
        time.sleep(5)
        iteration = trainer.get_iteration(project.id, iteration.id)
        print(f"  Training status: {iteration.status}")
    print("Training complete!")

    # --- Evaluate ---
    perf = trainer.get_iteration_performance(project.id, iteration.id, threshold=0.5)
    print(f"\nModel performance (threshold=0.5):")
    print(f"  Precision : {perf.precision:.4f}")
    print(f"  Recall    : {perf.recall:.4f}")
    print(f"  AP (mAP)  : {perf.average_precision:.4f}")
    for tag_perf in perf.per_tag_performance:
        print(f"  Tag '{tag_perf.name}': precision={tag_perf.precision:.4f} recall={tag_perf.recall:.4f}")

    # --- Publish iteration ---
    trainer.publish_iteration(
        project.id,
        iteration.id,
        publish_name="Iteration1",
        prediction_id=PREDICTION_RESOURCE_ID,
    )
    print("\nIteration published as 'Iteration1'")

    # --- Export model (optional) ---
    _export_iteration(trainer, project.id, iteration.id, platform="ONNX")


# ---------------------------------------------------------------------------
# 2. Object Detection Project
# ---------------------------------------------------------------------------

def demo_object_detection_project(trainer: CustomVisionTrainingClient) -> None:
    """
    Creates an object detection project, uploads images with bounding-box
    regions, trains, evaluates, and publishes.
    """
    print("\n" + "="*60)
    print("Object Detection Project Demo")
    print("="*60)

    # Object detection domain
    domains = trainer.get_domains()
    od_domain = next(d for d in domains if d.type == "ObjectDetection" and d.name == "General")

    project = trainer.create_project(
        name="AI102-ObjectDetection-Demo",
        description="Exam demo: detect apples and bananas",
        domain_id=od_domain.id,
    )
    print(f"Created OD project: {project.name} (id={project.id})")

    # Tags
    tag_apple  = trainer.create_tag(project.id, "apple")
    tag_banana = trainer.create_tag(project.id, "banana")

    # Upload images with normalised bounding box regions
    # Region(tag_id, left, top, width, height)  -- all values 0.0–1.0
    tagged_images = [
        ImageUrlCreateEntry(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/320px-Red_Apple.jpg",
            regions=[
                Region(tag_id=tag_apple.id, left=0.1, top=0.05, width=0.8, height=0.85)
            ],
        ),
        ImageUrlCreateEntry(
            url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Fruit-Bundle.jpg/320px-Banana-Fruit-Bundle.jpg",
            regions=[
                Region(tag_id=tag_banana.id, left=0.05, top=0.1, width=0.9, height=0.75)
            ],
        ),
    ]

    upload_result = trainer.create_images_from_urls(
        project.id, ImageUrlCreateBatch(images=tagged_images)
    )
    print(f"Uploaded {len(upload_result.images)} images")

    # Train
    print("Training object detection model...")
    iteration = trainer.train_project(project.id)
    while iteration.status != "Completed":
        time.sleep(5)
        iteration = trainer.get_iteration(project.id, iteration.id)
        print(f"  Status: {iteration.status}")

    # Evaluate
    perf = trainer.get_iteration_performance(project.id, iteration.id, threshold=0.5, overlap_threshold=0.3)
    print(f"\nOD Model performance:")
    print(f"  Precision : {perf.precision:.4f}")
    print(f"  Recall    : {perf.recall:.4f}")
    print(f"  mAP       : {perf.average_precision:.4f}")

    # Publish
    trainer.publish_iteration(
        project.id,
        iteration.id,
        publish_name="ODIteration1",
        prediction_id=PREDICTION_RESOURCE_ID,
    )
    print("Iteration published as 'ODIteration1'")

    # Export
    _export_iteration(trainer, project.id, iteration.id, platform="TensorFlow")


# ---------------------------------------------------------------------------
# 3. Export helper
# ---------------------------------------------------------------------------

def _export_iteration(
    trainer: CustomVisionTrainingClient,
    project_id: str,
    iteration_id: str,
    platform: str = "ONNX",  # ONNX | TensorFlow | CoreML | DockerFile
    flavor: str = None,       # e.g. "Windows", "Linux", "ARM" for TF
) -> None:
    """
    Request an export of the iteration and poll until ready.
    """
    print(f"\nRequesting {platform} export...")
    kwargs = {"flavor": flavor} if flavor else {}
    export: Export = trainer.export_iteration(project_id, iteration_id, platform, **kwargs)

    while export.status == "Exporting":
        time.sleep(3)
        exports = trainer.get_exports(project_id, iteration_id)
        export = next((e for e in exports if e.platform == platform), export)
        print(f"  Export status: {export.status}")

    if export.status == "Done":
        print(f"Export ready: {export.download_uri}")
    else:
        print(f"Export status: {export.status}")


# ---------------------------------------------------------------------------
# 4. List projects and iterations (utility)
# ---------------------------------------------------------------------------

def list_projects_and_iterations() -> None:
    """List all projects and their iterations – useful for audit."""
    trainer = get_training_client()
    projects = trainer.get_projects()
    print(f"\nFound {len(projects)} Custom Vision project(s):")
    for proj in projects:
        print(f"  Project: {proj.name} (id={proj.id})")
        iterations = trainer.get_iterations(proj.id)
        for it in iterations:
            pub = f"→ published as '{it.publish_name}'" if it.publish_name else "(not published)"
            print(f"    Iteration: {it.name}  status={it.status}  {pub}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = get_training_client()

    # Demo classification
    demo_classification_project(trainer)

    # Demo object detection
    demo_object_detection_project(trainer)

    # Utility: list everything
    list_projects_and_iterations()
