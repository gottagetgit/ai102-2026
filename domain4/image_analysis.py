"""
image_analysis.py
=================
Demonstrates Azure AI Vision v4 image analysis using ImageAnalysisClient.
Covers all major visual features tested in the AI-102 exam:

  - CAPTION        : single-sentence image description
  - DENSE_CAPTIONS : region-level captions
  - TAGS           : list of applicable tags with confidence
  - OBJECTS        : detected objects with bounding boxes
  - PEOPLE         : detected people with bounding boxes
  - SMART_CROPS    : aspect-ratio-aware thumbnail regions
  - READ           : embedded text / OCR

Also demonstrates background segmentation (segment API).

Exam Skill Mapping:
    - "Analyze images by using Azure AI Vision"
    - "Generate thumbnail images (smart cropping)"
    - "Detect and identify objects in images"
    - "Detect people in images"
    - "Extract text from images (OCR)"
    - "Remove image backgrounds"

Required Environment Variables (.env):
    AZURE_VISION_ENDPOINT  - e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_VISION_KEY       - Azure AI Vision key

Install:
    pip install azure-ai-vision-imageanalysis python-dotenv
"""

import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures, ImageAnalysisResult
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENDPOINT = os.environ["AZURE_VISION_ENDPOINT"]
KEY      = os.environ["AZURE_VISION_KEY"]

# Sample image for demo – a well-known public photo
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
    "Cat03.jpg/320px-Cat03.jpg"
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_client() -> ImageAnalysisClient:
    return ImageAnalysisClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


# ---------------------------------------------------------------------------
# 1. Full analysis – all features in one call
# ---------------------------------------------------------------------------

def analyze_image_full(image_url: str) -> ImageAnalysisResult:
    """
    Call analyze_from_url with every VisualFeature.
    Returns the raw result object for further processing.
    """
    client = get_client()

    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.PEOPLE,
            VisualFeatures.SMART_CROPS,
            VisualFeatures.READ,
        ],
        smart_crops_aspect_ratios=[0.9, 1.33],   # 4:3 landscape + ~1:1 square
        gender_neutral_caption=True,              # avoid gendered language
        language="en",
    )
    return result


# ---------------------------------------------------------------------------
# 2. Display helpers
# ---------------------------------------------------------------------------

def print_caption(result: ImageAnalysisResult) -> None:
    if result.caption:
        print(f"\nCaption:")
        print(f"  {result.caption.text!r}  (confidence={result.caption.confidence:.4f})")


def print_dense_captions(result: ImageAnalysisResult) -> None:
    if result.dense_captions:
        print(f"\nDense Captions ({len(result.dense_captions.list)} regions):")
        for cap in result.dense_captions.list:
            bb = cap.bounding_box
            print(
                f"  {cap.text!r:40s}  confidence={cap.confidence:.4f}  "
                f"bbox=[x={bb.x},y={bb.y},w={bb.width},h={bb.height}]"
            )


def print_tags(result: ImageAnalysisResult) -> None:
    if result.tags:
        print(f"\nTags ({len(result.tags.list)}):")
        for tag in result.tags.list:
            print(f"  {tag.name:<20s}  confidence={tag.confidence:.4f}")


def print_objects(result: ImageAnalysisResult) -> None:
    if result.objects:
        print(f"\nObjects ({len(result.objects.list)}):")
        for obj in result.objects.list:
            bb = obj.bounding_box
            # Each object can have multiple tags (hierarchical)
            labels = ", ".join(f"{t.name}({t.confidence:.2f})" for t in obj.tags)
            print(
                f"  {labels}  "
                f"bbox=[x={bb.x},y={bb.y},w={bb.width},h={bb.height}]"
            )


def print_people(result: ImageAnalysisResult) -> None:
    if result.people:
        print(f"\nPeople ({len(result.people.list)}):")
        for person in result.people.list:
            bb = person.bounding_box
            print(
                f"  confidence={person.confidence:.4f}  "
                f"bbox=[x={bb.x},y={bb.y},w={bb.width},h={bb.height}]"
            )


def print_smart_crops(result: ImageAnalysisResult) -> None:
    if result.smart_crops:
        print(f"\nSmart Crops ({len(result.smart_crops.list)}):")
        for crop in result.smart_crops.list:
            bb = crop.bounding_box
            print(
                f"  aspect_ratio={crop.aspect_ratio:.2f}  "
                f"bbox=[x={bb.x},y={bb.y},w={bb.width},h={bb.height}]"
            )


def print_read_text(result: ImageAnalysisResult) -> None:
    if result.read and result.read.blocks:
        print(f"\nRead Text:")
        for block in result.read.blocks:
            for line in block.lines:
                print(f"  Line: {line.text}")
                for word in line.words:
                    print(f"    Word: {word.text!r:20s}  confidence={word.confidence:.3f}")
    else:
        print("\nRead Text: none detected")


# ---------------------------------------------------------------------------
# 3. Targeted single-feature analysis
# ---------------------------------------------------------------------------

def analyze_tags_only(image_url: str) -> None:
    """Lightweight call: fetch only TAGS to minimise latency."""
    client = get_client()
    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.TAGS],
    )
    print_tags(result)


def analyze_ocr_only(image_url: str) -> None:
    """Lightweight call: fetch only READ (OCR text)."""
    client = get_client()
    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.READ],
    )
    print_read_text(result)


# ---------------------------------------------------------------------------
# 4. Local image (stream) variant
# ---------------------------------------------------------------------------

def analyze_local_image(image_path: str) -> None:
    """
    Analyse a local image by reading it as bytes and calling client.analyze().
    Useful when the image is not publicly accessible via URL.
    """
    client = get_client()
    with open(image_path, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS],
        content_type="image/jpeg",
    )
    print_caption(result)
    print_tags(result)


# ---------------------------------------------------------------------------
# 5. Background removal (segmentation)
# ---------------------------------------------------------------------------

def remove_background(image_url: str, output_path: str = "foreground.png") -> None:
    """
    Remove the background from an image using the Azure AI Vision segment API.
    Saves the result as a PNG with transparency.
    """
    client = get_client()
    print(f"\nRemoving background from: {image_url}")
    # segment() returns a BinaryData (bytes-like object)
    result = client.segment(
        segmentation_mode="backgroundRemoval",  # or "foregroundMatting"
        image_url=image_url,
    )
    with open(output_path, "wb") as f:
        f.write(result)
    print(f"Saved foreground image to: {output_path}")


# ---------------------------------------------------------------------------
# Main: run all demos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Azure AI Vision – Full Image Analysis Demo")
    print("=" * 60)
    print(f"Image: {SAMPLE_IMAGE_URL}")

    result = analyze_image_full(SAMPLE_IMAGE_URL)

    print_caption(result)
    print_dense_captions(result)
    print_tags(result)
    print_objects(result)
    print_people(result)
    print_smart_crops(result)
    print_read_text(result)

    # Targeted calls
    print("\n" + "-"*40)
    print("Targeted: Tags only")
    analyze_tags_only(SAMPLE_IMAGE_URL)

    print("\n" + "-"*40)
    print("Targeted: OCR only")
    analyze_ocr_only(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/"
        "Atomist_quote_from_Democritus.png/320px-Atomist_quote_from_Democritus.png"
    )

    # Background removal (uncomment to run – writes a file)
    # remove_background(SAMPLE_IMAGE_URL)
