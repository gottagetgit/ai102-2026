"""
ocr_read_text.py
================
Demonstrates extracting printed text from images using the Azure AI Vision
Read OCR API (azure-ai-vision-imageanalysis). The READ feature uses a deep
learning model optimised for printed (typed) text and supports 164 languages.

Results are structured in a hierarchy:
    Page → Block → Line → Word
    Each element includes a bounding polygon (list of {x,y} points).

Workflow:
    1. Call analyze() with VisualFeatures.READ
    2. Iterate over blocks → lines → words in result.read.blocks
    3. Display text content and spatial information
    4. Show how to reconstruct the full document text
    5. Demonstrate language hint usage

Exam Skill Mapping:
    - "Extract text from images using Azure Vision in Foundry Tools"

Required Environment Variables (.env):
    AZURE_AI_SERVICES_ENDPOINT
    AZURE_AI_SERVICES_KEY

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

# A printed text sample image from Microsoft docs
PRINTED_TEXT_IMAGE_URL = (
    "https://learn.microsoft.com/azure/ai-services/computer-vision/media/"
    "quickstarts/read-printed-text.png"
)

# Image with mixed content (printed + other elements)
MIXED_CONTENT_IMAGE_URL = (
    "https://learn.microsoft.com/azure/ai-services/computer-vision/media/"
    "concept-ocr/sample-ocr.png"
)


def get_client() -> ImageAnalysisClient:
    """Create and return an ImageAnalysisClient."""
    if not ENDPOINT or not KEY:
        raise ValueError(
            "AZURE_AI_SERVICES_ENDPOINT and AZURE_AI_SERVICES_KEY must be set in .env"
        )
    return ImageAnalysisClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def extract_printed_text(image_url: str, language: str = "en") -> dict:
    """Extract all printed text from an image URL and return structured results.

    Args:
        image_url: Publicly accessible URL to the image.
        language:  BCP-47 language hint (e.g. "en", "fr", "de"). Use "auto"
                   to let the service detect the language automatically.

    Returns:
        Dict with keys: full_text, blocks, word_count, line_count.
    """
    client = get_client()

    try:
        result = client.analyze_from_url(
            image_url=image_url,
            visual_features=[VisualFeatures.READ],
            # language hint improves accuracy when you know the document language
            language=language if language != "auto" else None,
        )
    except HttpResponseError as e:
        print(f"HTTP error: {e.status_code} — {e.message}")
        raise

    if result.read is None or not result.read.blocks:
        return {"full_text": "", "blocks": [], "word_count": 0, "line_count": 0}

    # -----------------------------------------------------------------------
    # Parse the structured OCR output
    # -----------------------------------------------------------------------
    structured_blocks = []
    all_lines = []

    for block_idx, block in enumerate(result.read.blocks):
        block_data = {
            "block_index": block_idx,
            "lines": [],
        }

        for line in block.lines:
            # Each Line has .text (full line) and .bounding_polygon
            words_data = []
            for word in line.words:
                words_data.append({
                    "text":       word.text,
                    "confidence": round(word.confidence, 4),
                    "polygon":    [{"x": p.x, "y": p.y} for p in word.bounding_polygon],
                })

            line_data = {
                "text":    line.text,
                "polygon": [{"x": p.x, "y": p.y} for p in line.bounding_polygon],
                "words":   words_data,
            }
            block_data["lines"].append(line_data)
            all_lines.append(line.text)

        structured_blocks.append(block_data)

    full_text = "\n".join(all_lines)
    word_count = sum(len(line.split()) for line in all_lines)

    return {
        "full_text":  full_text,
        "blocks":     structured_blocks,
        "word_count": word_count,
        "line_count": len(all_lines),
    }


def print_ocr_results(ocr_data: dict, show_polygons: bool = False) -> None:
    """Display OCR extraction results in a readable format.

    Args:
        ocr_data:      Output from extract_printed_text().
        show_polygons: If True, print bounding polygon coordinates.
    """
    print("=" * 60)
    print("OCR EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Lines extracted:  {ocr_data['line_count']}")
    print(f"Words extracted:  {ocr_data['word_count']}")
    print(f"Text blocks:      {len(ocr_data['blocks'])}")
    print()

    # Show the reconstructed plain text
    print("--- Reconstructed Plain Text ---")
    print(ocr_data["full_text"])
    print()

    # Show the structured hierarchy
    print("--- Structured Block/Line/Word Hierarchy ---")
    for block in ocr_data["blocks"]:
        print(f"\nBlock {block['block_index']}:")
        for line_idx, line in enumerate(block["lines"]):
            print(f"  Line {line_idx}: '{line['text']}'")
            if show_polygons:
                pts = " | ".join(f"({p['x']},{p['y']})" for p in line["polygon"])
                print(f"    Polygon: {pts}")
            # Show individual words with confidence
            for word in line["words"]:
                conf_bar = "█" * int(word["confidence"] * 10)
                print(
                    f"    Word: '{word['text']:<20}' "
                    f"conf={word['confidence']:.4f} {conf_bar}"
                )


def extract_text_from_file(file_path: str) -> dict:
    """Extract OCR text from a local image file.

    Args:
        file_path: Path to local image file (JPEG, PNG, BMP, TIFF, or PDF).

    Returns:
        Same structure as extract_printed_text().
    """
    client = get_client()

    with open(file_path, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
    )

    if result.read is None or not result.read.blocks:
        return {"full_text": "", "blocks": [], "word_count": 0, "line_count": 0}

    all_lines = []
    structured_blocks = []

    for block_idx, block in enumerate(result.read.blocks):
        block_data = {"block_index": block_idx, "lines": []}
        for line in block.lines:
            words_data = [
                {
                    "text": w.text,
                    "confidence": round(w.confidence, 4),
                }
                for w in line.words
            ]
            block_data["lines"].append({
                "text":  line.text,
                "words": words_data,
            })
            all_lines.append(line.text)
        structured_blocks.append(block_data)

    return {
        "full_text":  "\n".join(all_lines),
        "blocks":     structured_blocks,
        "word_count": sum(len(l.split()) for l in all_lines),
        "line_count": len(all_lines),
    }


def demonstrate_low_confidence_filtering(ocr_data: dict, threshold: float = 0.9) -> None:
    """Show words that fall below a confidence threshold.

    Low-confidence words may indicate poor image quality, unusual fonts,
    or partially obscured text that may need human review.

    Args:
        ocr_data:   Output from extract_printed_text().
        threshold:  Confidence below which a word is flagged.
    """
    print(f"\n--- Words below confidence threshold ({threshold}) ---")
    flagged = []
    for block in ocr_data["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                if word["confidence"] < threshold:
                    flagged.append(word)

    if flagged:
        for w in flagged:
            print(f"  '{w['text']}' — confidence: {w['confidence']:.4f}")
    else:
        print(f"  All words meet the confidence threshold of {threshold}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Vision — OCR Read Text Demo ===\n")

    # Example 1: Standard printed text
    print("--- Example 1: Printed text image ---")
    try:
        ocr_data = extract_printed_text(PRINTED_TEXT_IMAGE_URL)
        print_ocr_results(ocr_data, show_polygons=True)
        demonstrate_low_confidence_filtering(ocr_data, threshold=0.95)
    except Exception as exc:
        print(f"Example 1 error: {exc}")

    # Example 2: Mixed content image (demonstrate URL flexibility)
    print("\n--- Example 2: Mixed content image ---")
    try:
        ocr_data2 = extract_printed_text(MIXED_CONTENT_IMAGE_URL, language="auto")
        print_ocr_results(ocr_data2)
    except Exception as exc:
        print(f"Example 2 error: {exc}")
