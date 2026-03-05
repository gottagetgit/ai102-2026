"""
image_analysis.py
=================
Demonstrates using the Azure AI Vision Image Analysis API (azure-ai-vision-imageanalysis)
to extract a comprehensive range of visual features from an image in a single API call.

Visual features demonstrated:
    - CAPTION:        A natural-language description of the whole image
    - DENSE_CAPTIONS: Captions for individual regions within the image
    - TAGS:           Taxonomy tags with confidence scores
    - OBJECTS:        Detected objects with bounding boxes
    - PEOPLE:         Detected people with bounding boxes
    - SMART_CROPS:    Recommended crop regions for different aspect ratios
    - READ:           OCR text (combined with image analysis)

Workflow:
    1. Load the Vision client with endpoint + key
    2. Select the visual features you want
    3. Call analyze() with a URL or a local image stream
    4. Parse and display each feature's results
    5. Demonstrate with both a URL-based and file-based image

Exam Skill Mapping:
    - "Select visual features to meet image processing requirements"
    - "Detect objects in images and generate image tags"
    - "Include image analysis features in an image processing request"
    - "Interpret image processing responses"

Required Environment Variables (.env):
    AZURE_AI_SERVICES_ENDPOINT  - e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_AI_SERVICES_KEY       - Cognitive Services / AI Services API key

Install:
    pip install azure-ai-vision-imageanalysis python-dotenv
"""

import os
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

ENDPOINT = os.environ.get("AZURE_AI_SERVICES_ENDPOINT")
KEY      = os.environ.get("AZURE_AI_SERVICES_KEY")

# Public sample image URL (Microsoft docs sample)
SAMPLE_IMAGE_URL = (
    "https://learn.microsoft.com/azure/ai-services/computer-vision/media/"
    "quickstarts/image-url.png"
)


def get_client() -> ImageAnalysisClient:
    """Create and return a Vision ImageAnalysisClient."""
    if not ENDPOINT or not KEY:
        raise ValueError(
            "AZURE_AI_SERVICES_ENDPOINT and AZURE_AI_SERVICES_KEY must be set in .env"
        )
    return ImageAnalysisClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def analyze_image_from_url(image_url: str) -> None:
    """Analyze an image at a public URL and print all visual feature results.

    Args:
        image_url: Publicly accessible URL to the image.
    """
    client = get_client()

    print(f"Analyzing image from URL:\n  {image_url}\n")

    try:
        # Request ALL visual features in one API call.
        # In production, request only the features you need to minimise cost.
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
            smart_crops_aspect_ratios=[1.0, 1.78, 0.75],  # Square, 16:9, 3:4
            gender_neutral_caption=True,  # Use gender-neutral pronouns in captions
            language="en",
        )

        print_analysis_results(result)

    except HttpResponseError as e:
        print(f"HTTP error during image analysis: {e.status_code} — {e.message}")
        raise


def analyze_image_from_file(file_path: str) -> None:
    """Analyze a local image file.

    Args:
        file_path: Path to a local image (JPEG, PNG, BMP, GIF, TIFF, or WebP).
    """
    client = get_client()

    print(f"Analyzing local image:\n  {file_path}\n")

    with open(file_path, "rb") as f:
        image_data = f.read()

    try:
        result = client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE,
            ],
            gender_neutral_caption=True,
        )
        print_analysis_results(result)

    except HttpResponseError as e:
        print(f"HTTP error during image analysis: {e.status_code} — {e.message}")
        raise


def print_analysis_results(result) -> None:
    """Pretty-print all visual analysis results.

    Args:
        result: ImageAnalysisResult object from the SDK.
    """
    print("=" * 60)
    print("IMAGE ANALYSIS RESULTS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Caption: single natural-language description of the image
    # ------------------------------------------------------------------
    if result.caption is not None:
        print(f"\n[CAPTION]")
        print(f"  Text:       {result.caption.text}")
        print(f"  Confidence: {result.caption.confidence:.4f}")

    # ------------------------------------------------------------------
    # Dense Captions: captions for individual regions
    # The first item is always the whole-image caption.
    # ------------------------------------------------------------------
    if result.dense_captions is not None:
        print(f"\n[DENSE CAPTIONS] ({len(result.dense_captions.list)} regions)")
        for i, dc in enumerate(result.dense_captions.list):
            bbox = dc.bounding_box
            print(f"  Region {i+1}: '{dc.text}'")
            print(f"    Confidence: {dc.confidence:.4f}")
            print(
                f"    Bounding box: x={bbox.x}, y={bbox.y}, "
                f"w={bbox.width}, h={bbox.height}"
            )

    # ------------------------------------------------------------------
    # Tags: keyword tags with confidence scores
    # ------------------------------------------------------------------
    if result.tags is not None:
        print(f"\n[TAGS] ({len(result.tags.list)} tags)")
        # Sort by confidence descending for easier reading
        sorted_tags = sorted(result.tags.list, key=lambda t: t.confidence, reverse=True)
        for tag in sorted_tags:
            bar = "█" * int(tag.confidence * 20)
            print(f"  {tag.name:<25} {tag.confidence:.4f}  {bar}")

    # ------------------------------------------------------------------
    # Objects: detected objects with bounding boxes
    # Objects are more specific than tags and include spatial location.
    # ------------------------------------------------------------------
    if result.objects is not None:
        print(f"\n[OBJECTS] ({len(result.objects.list)} objects)")
        for obj in result.objects.list:
            bbox = obj.bounding_box
            tags_str = ", ".join(
                f"{t.name}({t.confidence:.2f})"
                for t in (obj.tags or [])
            )
            print(f"  Object: {tags_str}")
            print(
                f"    Bounding box: x={bbox.x}, y={bbox.y}, "
                f"w={bbox.width}, h={bbox.height}"
            )

    # ------------------------------------------------------------------
    # People: detected people with bounding boxes and confidence
    # ------------------------------------------------------------------
    if result.people is not None:
        print(f"\n[PEOPLE] ({len(result.people.list)} people detected)")
        for i, person in enumerate(result.people.list):
            bbox = person.bounding_box
            print(f"  Person {i+1}: confidence={person.confidence:.4f}")
            print(
                f"    Bounding box: x={bbox.x}, y={bbox.y}, "
                f"w={bbox.width}, h={bbox.height}"
            )

    # ------------------------------------------------------------------
    # Smart Crops: recommended crop regions for given aspect ratios
    # Useful for generating thumbnails that keep the main subject in frame
    # ------------------------------------------------------------------
    if result.smart_crops is not None:
        print(f"\n[SMART CROPS]")
        for crop in result.smart_crops.list:
            bbox = crop.bounding_box
            print(f"  Aspect ratio {crop.aspect_ratio:.2f}:")
            print(
                f"    Crop: x={bbox.x}, y={bbox.y}, "
                f"w={bbox.width}, h={bbox.height}"
            )

    # ------------------------------------------------------------------
    # Read (OCR): any text found in the image
    # ------------------------------------------------------------------
    if result.read is not None:
        blocks = result.read.blocks
        if blocks:
            print(f"\n[OCR / READ TEXT] ({len(blocks)} text block(s))")
            for block in blocks:
                for line in block.lines:
                    print(f"  Line: '{line.text}'")
        else:
            print("\n[OCR / READ TEXT] No text detected.")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    if result.metadata is not None:
        print(f"\n[IMAGE METADATA]")
        print(f"  Dimensions: {result.metadata.width} x {result.metadata.height} px")

    if result.model_version:
        print(f"\n[MODEL VERSION] {result.model_version}")

    print("\n" + "=" * 60)


def demonstrate_selective_features() -> None:
    """Show how to request only specific features (more efficient)."""
    client = get_client()
    print("\n--- Selective Features Demo (Tags + Caption only) ---")

    result = client.analyze_from_url(
        image_url=SAMPLE_IMAGE_URL,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.TAGS,
        ],
    )

    if result.caption:
        print(f"Caption: {result.caption.text} ({result.caption.confidence:.2%} confidence)")
    if result.tags:
        top5 = sorted(result.tags.list, key=lambda t: t.confidence, reverse=True)[:5]
        print("Top 5 tags:", ", ".join(f"{t.name}({t.confidence:.2f})" for t in top5))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Vision — Image Analysis Demo ===\n")

    try:
        # Full analysis from URL
        analyze_image_from_url(SAMPLE_IMAGE_URL)

        # Selective features only (more efficient in production)
        demonstrate_selective_features()

    except Exception as exc:
        print(f"\nError: {exc}")
        raise
