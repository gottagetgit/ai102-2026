"""
custom_vision_predict.py
========================
Demonstrates publishing a trained Custom Vision iteration and consuming
the prediction endpoint to classify new images or detect objects.

Workflow:
    1. Publish the trained iteration to a prediction resource
    2. Create a CustomVisionPredictionClient
    3. Send images to the prediction endpoint (URL or file bytes)
    4. Parse and rank predictions by probability
    5. Apply a confidence threshold to filter low-probability results
    6. Show how to delete a published iteration (to save prediction costs)

Note:
    Run custom_vision_train.py first to create a project and iteration,
    then copy the project ID and iteration ID into your .env file.

Exam Skill Mapping:
    - "Publish a custom vision model"
    - "Consume a custom vision model"

Required Environment Variables (.env):
    CUSTOM_VISION_TRAINING_ENDPOINT
    CUSTOM_VISION_TRAINING_KEY
    CUSTOM_VISION_PREDICTION_ENDPOINT - Prediction resource endpoint
    CUSTOM_VISION_PREDICTION_KEY      - Prediction resource key
    CUSTOM_VISION_PREDICTION_RESOURCE_ID - Full ARM resource ID of prediction resource
    CUSTOM_VISION_PROJECT_ID          - From custom_vision_train.py output
    CUSTOM_VISION_ITERATION_ID        - From custom_vision_train.py output
    CUSTOM_VISION_PUBLISH_NAME        - Friendly name for the published model

Install:
    pip install azure-cognitiveservices-vision-customvision python-dotenv
"""

import os
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.prediction.models import (
    ImagePrediction,
    Prediction,
)

load_dotenv()

TRAINING_ENDPOINT       = os.environ.get("CUSTOM_VISION_TRAINING_ENDPOINT")
TRAINING_KEY            = os.environ.get("CUSTOM_VISION_TRAINING_KEY")
PREDICTION_ENDPOINT     = os.environ.get("CUSTOM_VISION_PREDICTION_ENDPOINT")
PREDICTION_KEY          = os.environ.get("CUSTOM_VISION_PREDICTION_KEY")
PREDICTION_RESOURCE_ID  = os.environ.get("CUSTOM_VISION_PREDICTION_RESOURCE_ID")
PROJECT_ID              = os.environ.get("CUSTOM_VISION_PROJECT_ID")
ITERATION_ID            = os.environ.get("CUSTOM_VISION_ITERATION_ID")
PUBLISH_NAME            = os.environ.get("CUSTOM_VISION_PUBLISH_NAME", "ProductionModel")

# Minimum probability to report a prediction result
CONFIDENCE_THRESHOLD = 0.5

# Test images to classify (using the same classes from the training script)
TEST_IMAGES = {
    "expected_cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    "expected_dog": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Black_Labrador_Retriever_-_Male_IMG_3323.jpg/320px-Black_Labrador_Retriever_-_Male_IMG_3323.jpg",
    "ambiguous":    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SNice.svg/320px-SNice.svg.png",
}


def get_training_client() -> CustomVisionTrainingClient:
    """Create a training client (needed for publishing)."""
    if not TRAINING_ENDPOINT or not TRAINING_KEY:
        raise ValueError("CUSTOM_VISION_TRAINING_ENDPOINT and CUSTOM_VISION_TRAINING_KEY must be set.")
    credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    return CustomVisionTrainingClient(TRAINING_ENDPOINT, credentials)


def get_prediction_client() -> CustomVisionPredictionClient:
    """Create a prediction client."""
    if not PREDICTION_ENDPOINT or not PREDICTION_KEY:
        raise ValueError("CUSTOM_VISION_PREDICTION_ENDPOINT and CUSTOM_VISION_PREDICTION_KEY must be set.")
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
    return CustomVisionPredictionClient(PREDICTION_ENDPOINT, credentials)


def publish_iteration(
    trainer: CustomVisionTrainingClient,
    project_id: str,
    iteration_id: str,
    publish_name: str,
    prediction_resource_id: str,
) -> bool:
    """Publish a training iteration to the prediction endpoint.

    Once published, the iteration is available under the given `publish_name`
    for prediction calls. A project can have multiple published iterations,
    allowing A/B testing or gradual rollout.

    Args:
        trainer:                CustomVisionTrainingClient
        project_id:             UUID of the project
        iteration_id:           UUID of the iteration to publish
        publish_name:           Friendly name clients use to call this model
        prediction_resource_id: Full ARM resource ID of the prediction resource
                                Format: /subscriptions/<sub>/resourceGroups/<rg>/
                                        providers/Microsoft.CognitiveServices/
                                        accounts/<resource>

    Returns:
        True if published successfully.
    """
    # Check if already published
    iteration = trainer.get_iteration(project_id, iteration_id)
    if iteration.publish_name:
        print(f"  Iteration already published as: '{iteration.publish_name}'")
        return True

    trainer.publish_iteration(
        project_id=project_id,
        iteration_id=iteration_id,
        publish_name=publish_name,
        prediction_id=prediction_resource_id,
    )
    print(f"  Iteration published as: '{publish_name}'")
    return True


def classify_image_url(
    predictor: CustomVisionPredictionClient,
    project_id: str,
    publish_name: str,
    image_url: str,
) -> list[dict]:
    """Classify an image at a URL.

    Args:
        predictor:    CustomVisionPredictionClient
        project_id:   UUID of the Custom Vision project
        publish_name: Name under which the model was published
        image_url:    URL of the image to classify

    Returns:
        List of prediction dicts sorted by probability descending.
    """
    result: ImagePrediction = predictor.classify_image_url(
        project_id=project_id,
        published_name=publish_name,
        url=image_url,
    )
    return parse_predictions(result.predictions)


def classify_image_file(
    predictor: CustomVisionPredictionClient,
    project_id: str,
    publish_name: str,
    file_path: str,
) -> list[dict]:
    """Classify a local image file.

    Args:
        predictor:    CustomVisionPredictionClient
        project_id:   UUID of the project
        publish_name: Published model name
        file_path:    Path to the local image file

    Returns:
        List of prediction dicts sorted by probability descending.
    """
    with open(file_path, "rb") as f:
        result: ImagePrediction = predictor.classify_image(
            project_id=project_id,
            published_name=publish_name,
            image_data=f,
        )
    return parse_predictions(result.predictions)


def detect_objects_url(
    predictor: CustomVisionPredictionClient,
    project_id: str,
    publish_name: str,
    image_url: str,
) -> list[dict]:
    """Run object detection on an image URL.

    Only use with Object Detection projects (not classification).
    Bounding boxes are normalised (0.0–1.0) relative to image dimensions.

    Args:
        predictor:    CustomVisionPredictionClient
        project_id:   UUID of the object detection project
        publish_name: Published model name
        image_url:    URL of the image

    Returns:
        List of detection dicts with tag, probability, and bounding box.
    """
    result: ImagePrediction = predictor.detect_image_url(
        project_id=project_id,
        published_name=publish_name,
        url=image_url,
    )
    detections = []
    for p in result.predictions:
        if p.probability >= CONFIDENCE_THRESHOLD:
            bb = p.bounding_box
            detections.append({
                "tag":         p.tag_name,
                "probability": round(p.probability, 4),
                "bounding_box": {
                    "left":   round(bb.left, 4),
                    "top":    round(bb.top, 4),
                    "width":  round(bb.width, 4),
                    "height": round(bb.height, 4),
                },
            })
    return sorted(detections, key=lambda d: d["probability"], reverse=True)


def parse_predictions(predictions: list[Prediction]) -> list[dict]:
    """Convert SDK Prediction objects to plain dicts and sort by probability.

    Args:
        predictions: List of Prediction objects from the SDK response.

    Returns:
        List of dicts with tag_name and probability, sorted descending.
    """
    parsed = [
        {
            "tag":         p.tag_name,
            "probability": round(p.probability, 4),
        }
        for p in predictions
    ]
    return sorted(parsed, key=lambda x: x["probability"], reverse=True)


def print_classification_results(image_label: str, predictions: list[dict]) -> None:
    """Display classification results with a visual confidence bar.

    Args:
        image_label: Descriptive label for the image being classified.
        predictions: List of prediction dicts from parse_predictions().
    """
    print(f"\n  Image: {image_label}")
    print(f"  {'Tag':<25} {'Probability':>12} {'Bar'}")
    print(f"  {'-'*25} {'-'*12} {'-'*20}")

    if not predictions:
        print("  (no predictions returned)")
        return

    for pred in predictions:
        bar   = "█" * int(pred["probability"] * 20)
        flag  = " ← TOP" if pred == predictions[0] else ""
        below = " (below threshold)" if pred["probability"] < CONFIDENCE_THRESHOLD else ""
        print(
            f"  {pred['tag']:<25} {pred['probability']:>12.4f}  {bar}{flag}{below}"
        )

    # Best prediction (highest probability above threshold)
    best = next(
        (p for p in predictions if p["probability"] >= CONFIDENCE_THRESHOLD), None
    )
    if best:
        print(
            f"\n  Result: '{best['tag']}' "
            f"({best['probability']:.2%} confidence)"
        )
    else:
        print(f"\n  Result: No prediction met the {CONFIDENCE_THRESHOLD:.0%} threshold.")


def unpublish_iteration(
    trainer: CustomVisionTrainingClient, project_id: str, iteration_id: str
) -> None:
    """Unpublish an iteration to stop prediction billing.

    Unpublished iterations still exist in the project for retraining or
    re-publishing. Prediction API calls against an unpublished model will fail.
    """
    trainer.unpublish_iteration(project_id=project_id, iteration_id=iteration_id)
    print(f"  Iteration {iteration_id} unpublished.")


def run_prediction_demo():
    """Full prediction pipeline demonstration."""
    # Validate required env vars
    missing = []
    if not PROJECT_ID:        missing.append("CUSTOM_VISION_PROJECT_ID")
    if not ITERATION_ID:      missing.append("CUSTOM_VISION_ITERATION_ID")
    if not PREDICTION_RESOURCE_ID: missing.append("CUSTOM_VISION_PREDICTION_RESOURCE_ID")
    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}\n"
            "Run custom_vision_train.py first and copy the output values to .env."
        )

    trainer  = get_training_client()
    predictor = get_prediction_client()

    # ------------------------------------------------------------------
    # 1. Publish the iteration
    # ------------------------------------------------------------------
    print("[1/3] Publishing iteration to prediction endpoint...")
    publish_iteration(trainer, PROJECT_ID, ITERATION_ID, PUBLISH_NAME, PREDICTION_RESOURCE_ID)

    # ------------------------------------------------------------------
    # 2. Classify test images
    # ------------------------------------------------------------------
    print(f"\n[2/3] Classifying test images (threshold={CONFIDENCE_THRESHOLD:.0%})...")
    print("=" * 60)

    for label, url in TEST_IMAGES.items():
        try:
            predictions = classify_image_url(predictor, PROJECT_ID, PUBLISH_NAME, url)
            print_classification_results(label, predictions)
        except Exception as exc:
            print(f"  Error classifying '{label}': {exc}")

    # ------------------------------------------------------------------
    # 3. (Optional) Unpublish to stop billing
    # ------------------------------------------------------------------
    print(f"\n[3/3] Model consumed successfully.")
    print(f"\nTip: To stop prediction billing, call unpublish_iteration():")
    print(f"     unpublish_iteration(trainer, PROJECT_ID, ITERATION_ID)")
    print(f"     The iteration remains in the project and can be re-published later.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure Custom Vision — Prediction Demo ===\n")
    try:
        run_prediction_demo()
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
