"""
custom_vision_predict.py
========================
Demonstrates how to consume Azure Custom Vision prediction endpoints for:
  1. Image Classification  – classifies an image into one or more tags
  2. Object Detection      – detects objects and returns bounding boxes

This file covers the PREDICTION side. For training, see custom_vision_train.py.

Exam Skill Mapping:
    - "Consume a Custom Vision image classification solution"
    - "Consume a Custom Vision object detection solution"
    - "Train, evaluate, publish, and consume Custom Vision models"

Prerequisites:
    A trained and PUBLISHED Custom Vision project (iteration must be published
    to a prediction resource before the prediction endpoint works).

Required Environment Variables (.env):
    CUSTOM_VISION_PREDICTION_ENDPOINT  - e.g. https://<resource>.cognitiveservices.azure.com/
    CUSTOM_VISION_PREDICTION_KEY       - prediction key
    CUSTOM_VISION_PREDICTION_RESOURCE_ID - full ARM resource ID of the prediction resource
    CUSTOM_VISION_PROJECT_ID           - GUID of your project
    CUSTOM_VISION_ITERATION_NAME       - published iteration name (e.g. "Iteration1")

Install:
    pip install azure-cognitiveservices-vision-customvision python-dotenv requests
"""

import os
import io
import requests
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.prediction.models import (
    ImagePrediction,
    Prediction,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREDICTION_ENDPOINT = os.environ["CUSTOM_VISION_PREDICTION_ENDPOINT"]
PREDICTION_KEY      = os.environ["CUSTOM_VISION_PREDICTION_KEY"]
PROJECT_ID          = os.environ["CUSTOM_VISION_PROJECT_ID"]
ITERATION_NAME      = os.environ["CUSTOM_VISION_ITERATION_NAME"]

# ---------------------------------------------------------------------------
# Build prediction client
# ---------------------------------------------------------------------------

def get_prediction_client() -> CustomVisionPredictionClient:
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
    return CustomVisionPredictionClient(PREDICTION_ENDPOINT, credentials)


# ---------------------------------------------------------------------------
# 1. Image Classification
# ---------------------------------------------------------------------------

def classify_image_from_url(image_url: str) -> None:
    """
    Classify a remote image using a published classification model.
    Returns the top predicted tags with confidence scores.
    """
    client = get_prediction_client()

    print(f"\nClassifying image: {image_url}")
    result: ImagePrediction = client.classify_image_url(
        project_id=PROJECT_ID,
        published_name=ITERATION_NAME,
        url=image_url,
    )

    print("Classification results (sorted by probability):")
    sorted_preds = sorted(result.predictions, key=lambda p: p.probability, reverse=True)
    for pred in sorted_preds:
        bar = "#" * int(pred.probability * 20)
        print(f"  {pred.tag_name:<20} {pred.probability:.4f}  [{bar:<20}]")


def classify_image_from_file(image_path: str) -> None:
    """
    Classify a local image file using a published classification model.
    """
    client = get_prediction_client()

    print(f"\nClassifying local image: {image_path}")
    with open(image_path, "rb") as f:
        image_data = f.read()

    result: ImagePrediction = client.classify_image(
        project_id=PROJECT_ID,
        published_name=ITERATION_NAME,
        image_data=image_data,
    )

    print("Classification results:")
    sorted_preds = sorted(result.predictions, key=lambda p: p.probability, reverse=True)
    for pred in sorted_preds:
        print(f"  {pred.tag_name:<20} {pred.probability:.4f}")


# ---------------------------------------------------------------------------
# 2. Object Detection
# ---------------------------------------------------------------------------

def detect_objects_from_url(image_url: str) -> None:
    """
    Run object detection on a remote image.
    Returns bounding boxes, tag names, and confidence scores.
    """
    client = get_prediction_client()

    print(f"\nDetecting objects in: {image_url}")
    result: ImagePrediction = client.detect_image_url(
        project_id=PROJECT_ID,
        published_name=ITERATION_NAME,
        url=image_url,
    )

    print("Object detection results:")
    for pred in result.predictions:
        if pred.probability > 0.5:   # filter low-confidence detections
            bb = pred.bounding_box
            print(
                f"  {pred.tag_name:<20} {pred.probability:.4f}  "
                f"BBox: left={bb.left:.3f} top={bb.top:.3f} "
                f"width={bb.width:.3f} height={bb.height:.3f}"
            )


def detect_objects_from_file(image_path: str) -> None:
    """
    Run object detection on a local image file.
    """
    client = get_prediction_client()

    with open(image_path, "rb") as f:
        image_data = f.read()

    print(f"\nDetecting objects in local image: {image_path}")
    result: ImagePrediction = client.detect_image(
        project_id=PROJECT_ID,
        published_name=ITERATION_NAME,
        image_data=image_data,
    )

    print("Object detection results:")
    for pred in result.predictions:
        if pred.probability > 0.5:
            bb = pred.bounding_box
            print(
                f"  {pred.tag_name:<20} {pred.probability:.4f}  "
                f"BBox: left={bb.left:.3f} top={bb.top:.3f} "
                f"width={bb.width:.3f} height={bb.height:.3f}"
            )


# ---------------------------------------------------------------------------
# 3. Batch prediction with direct REST (alternative to SDK)
# ---------------------------------------------------------------------------

def classify_image_rest(image_url: str) -> None:
    """
    Call the Custom Vision prediction endpoint directly via REST.
    Useful when the SDK is not available or for understanding the raw API.
    """
    # Construct the prediction URL
    # Format: {endpoint}/customvision/v3.0/Prediction/{projectId}/classify/iterations/{iterationName}/url
    url = (
        f"{PREDICTION_ENDPOINT.rstrip('/')}/customvision/v3.0/Prediction/"
        f"{PROJECT_ID}/classify/iterations/{ITERATION_NAME}/url"
    )

    headers = {
        "Prediction-Key": PREDICTION_KEY,
        "Content-Type": "application/json",
    }
    body = {"Url": image_url}

    print(f"\nREST prediction for: {image_url}")
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    data = response.json()

    print("REST Classification results:")
    for pred in sorted(data["predictions"], key=lambda p: p["probability"], reverse=True):
        print(f"  {pred['tagName']:<20} {pred['probability']:.4f}")


# ---------------------------------------------------------------------------
# 4. Interpret results and export model info
# ---------------------------------------------------------------------------

def get_project_iterations() -> None:
    """
    List all published iterations for the project (training client needed).
    Demonstrates how to check which iteration is published.
    """
    from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
    training_endpoint = os.environ.get("CUSTOM_VISION_TRAINING_ENDPOINT", PREDICTION_ENDPOINT)
    training_key      = os.environ.get("CUSTOM_VISION_TRAINING_KEY")

    if not training_key:
        print("CUSTOM_VISION_TRAINING_KEY not set; skipping iteration listing.")
        return

    creds    = ApiKeyCredentials(in_headers={"Training-key": training_key})
    trainer  = CustomVisionTrainingClient(training_endpoint, creds)
    project_id = PROJECT_ID

    iterations = trainer.get_iterations(project_id)
    print("\nPublished iterations:")
    for it in iterations:
        status = f"published as '{it.publish_name}'" if it.publish_name else "not published"
        print(f"  Iteration {it.name} (id={it.id}): {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example image URLs for testing
    SAMPLE_CLASSIFICATION_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/"
        "Cute_dog.jpg/320px-Cute_dog.jpg"
    )
    SAMPLE_DETECTION_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/"
        "Shopping_Center_Magna_Plaza_Amsterdam_2012.jpg/"
        "320px-Shopping_Center_Magna_Plaza_Amsterdam_2012.jpg"
    )

    print("=" * 60)
    print("Azure Custom Vision Prediction Demo")
    print("=" * 60)

    # --- Classification ---
    classify_image_from_url(SAMPLE_CLASSIFICATION_URL)

    # --- Object Detection ---
    detect_objects_from_url(SAMPLE_DETECTION_URL)

    # --- REST API alternative ---
    classify_image_rest(SAMPLE_CLASSIFICATION_URL)

    # --- List iterations ---
    get_project_iterations()
