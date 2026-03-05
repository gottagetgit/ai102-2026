"""
handwriting_recognition.py
===========================
Demonstrates extracting handwritten text from images using the Azure AI Vision
Read OCR API. The same READ feature handles both printed and handwritten text —
the model automatically distinguishes between them.

This script focuses on aspects specific to handwriting:
    - Confidence scores (handwriting typically has lower confidence than print)
    - Appearance style metadata (handwritten vs print classification per line)
    - Handling mixed documents (handwritten annotations on printed forms)
    - Tips for improving accuracy with handwritten content

Workflow:
    1. Analyze an image with VisualFeatures.READ
    2. Check the appearance.style property on each line for handwriting detection
    3. Display per-word confidence scores (critical for handwriting quality checks)
    4. Aggregate statistics on handwriting vs printed text regions

Exam Skill Mapping:
    - "Convert handwritten text using Azure Vision in Foundry Tools"

Required Environment Variables (.env):
    AZURE_AI_SERVICES_ENDPOINT
    AZURE_AI_SERVICES_KEY

Install:
    pip install azure-ai-vision-imageanalysis python-dotenv
"""

import os
import statistics
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

ENDPOINT = os.environ.get("AZURE_AI_SERVICES_ENDPOINT")
KEY      = os.environ.get("AZURE_AI_SERVICES_KEY")

# Handwriting sample from Microsoft docs
HANDWRITING_IMAGE_URL = (
    "https://learn.microsoft.com/azure/ai-services/computer-vision/media/"
    "quickstarts/read-handwritten-text.png"
)

# A note/form image that contains both handwritten and printed elements
MIXED_HANDWRITING_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/"
    "Handwriting_of_Albert_Einstein.jpg/640px-Handwriting_of_Albert_Einstein.jpg"
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


def extract_handwritten_text(image_url: str) -> dict:
    """Extract text from a handwritten image and return detailed results.

    The READ API processes handwriting automatically — no separate mode is needed.
    The `appearance` property on each line indicates if it was identified as
    handwritten or printed text, along with confidence in that classification.

    Args:
        image_url: URL to a handwritten document image.

    Returns:
        Dict with:
            full_text:          All extracted text joined by newlines
            lines:              List of line dicts with text, style, and words
            handwritten_lines:  Count of lines classified as handwriting
            printed_lines:      Count of lines classified as print
            avg_confidence:     Average word-level confidence
            low_confidence_words: Words with confidence < 0.7
    """
    client = get_client()

    try:
        result = client.analyze_from_url(
            image_url=image_url,
            visual_features=[VisualFeatures.READ],
        )
    except HttpResponseError as e:
        print(f"HTTP error: {e.status_code} — {e.message}")
        raise

    if result.read is None or not result.read.blocks:
        return {
            "full_text":            "",
            "lines":                [],
            "handwritten_lines":    0,
            "printed_lines":        0,
            "avg_confidence":       0.0,
            "low_confidence_words": [],
        }

    all_lines = []
    all_words = []
    handwritten_count = 0
    printed_count = 0
    low_conf_words = []

    for block in result.read.blocks:
        for line in block.lines:
            # ----------------------------------------------------------
            # Appearance style: "handwriting" or "print"
            # The style property may be None for older model versions.
            # ----------------------------------------------------------
            style_name = None
            style_conf = None

            if hasattr(line, "appearance") and line.appearance is not None:
                if hasattr(line.appearance, "style"):
                    style_name = str(line.appearance.style.name).lower()
                    style_conf = line.appearance.style.confidence

            # Count by style
            if style_name == "handwriting":
                handwritten_count += 1
            elif style_name == "print":
                printed_count += 1

            # Collect word-level data
            words_data = []
            for word in line.words:
                all_words.append(word.confidence)
                word_entry = {
                    "text":       word.text,
                    "confidence": round(word.confidence, 4),
                    "polygon":    [{"x": p.x, "y": p.y} for p in word.bounding_polygon],
                }
                words_data.append(word_entry)
                if word.confidence < 0.70:
                    low_conf_words.append({
                        "text":       word.text,
                        "confidence": round(word.confidence, 4),
                        "line":       line.text,
                    })

            all_lines.append({
                "text":        line.text,
                "style":       style_name,
                "style_conf":  round(style_conf, 4) if style_conf else None,
                "bounding":    [{"x": p.x, "y": p.y} for p in line.bounding_polygon],
                "words":       words_data,
            })

    avg_conf = statistics.mean(all_words) if all_words else 0.0

    return {
        "full_text":            "\n".join(line["text"] for line in all_lines),
        "lines":                all_lines,
        "handwritten_lines":    handwritten_count,
        "printed_lines":        printed_count,
        "unknown_style_lines":  len(all_lines) - handwritten_count - printed_count,
        "total_words":          len(all_words),
        "avg_confidence":       round(avg_conf, 4),
        "low_confidence_words": low_conf_words,
    }


def print_handwriting_results(data: dict) -> None:
    """Display handwriting extraction results with quality analysis.

    Args:
        data: Output from extract_handwritten_text().
    """
    print("=" * 60)
    print("HANDWRITING RECOGNITION RESULTS")
    print("=" * 60)

    print(f"\nTotal lines:           {len(data['lines'])}")
    print(f"Handwritten lines:     {data['handwritten_lines']}")
    print(f"Printed lines:         {data['printed_lines']}")
    print(f"Style unknown:         {data.get('unknown_style_lines', 0)}")
    print(f"Total words:           {data['total_words']}")
    print(f"Average confidence:    {data['avg_confidence']:.4f}")
    print(f"Low-confidence words:  {len(data['low_confidence_words'])}")

    # Confidence quality rating
    avg = data["avg_confidence"]
    if avg >= 0.90:
        quality = "Excellent — clear handwriting, high accuracy expected"
    elif avg >= 0.75:
        quality = "Good — minor legibility issues, review recommended"
    elif avg >= 0.60:
        quality = "Fair — some words may be misread, human review suggested"
    else:
        quality = "Poor — unclear handwriting, significant errors likely"
    print(f"Quality assessment:    {quality}")

    # Extracted text
    print(f"\n--- Extracted Text ---")
    print(data["full_text"])

    # Line-by-line detail with style classification
    print(f"\n--- Line-by-Line Detail ---")
    for i, line in enumerate(data["lines"]):
        style_display = f"[{line['style']}]" if line["style"] else "[unknown style]"
        conf_display  = f"(style conf: {line['style_conf']})" if line["style_conf"] else ""
        print(f"\n  Line {i+1} {style_display} {conf_display}")
        print(f"  Text: '{line['text']}'")

        # Show word confidences as a mini table
        if line["words"]:
            print(f"  Words:")
            for word in line["words"]:
                bar = "█" * int(word["confidence"] * 10)
                flag = " ⚠ low confidence" if word["confidence"] < 0.70 else ""
                print(
                    f"    '{word['text']:<20}' "
                    f"conf={word['confidence']:.4f}  {bar}{flag}"
                )

    # Low confidence summary
    if data["low_confidence_words"]:
        print(f"\n--- Low Confidence Words (may need review) ---")
        for w in data["low_confidence_words"]:
            print(f"  '{w['text']}' (confidence: {w['confidence']:.4f})")
            print(f"    In line: '{w['line']}'")

    print("\n" + "=" * 60)


def handwriting_quality_tips() -> None:
    """Print guidance on improving handwriting recognition accuracy."""
    tips = [
        ("Image resolution",   "Use at least 200 DPI; 300 DPI recommended for small text."),
        ("Lighting",           "Even, diffuse lighting avoids shadows that obscure strokes."),
        ("Contrast",           "Dark ink on light background (or vice versa) performs best."),
        ("Angle",              "Scan/photograph as flat as possible; < 5° skew is ideal."),
        ("Language hints",     "Set the language parameter if the handwriting language is known."),
        ("Segment images",     "Crop densely packed writing into smaller regions for better results."),
        ("Pre-processing",     "Apply contrast enhancement and noise reduction before calling the API."),
        ("Post-processing",    "Apply spell-check with a domain dictionary to correct low-conf words."),
        ("Training data",      "For specialist handwriting (medical, legal), consider a custom model."),
    ]

    print("\n--- Tips for Improving Handwriting Recognition Accuracy ---")
    for tip, detail in tips:
        print(f"  {tip:<22} {detail}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Vision — Handwriting Recognition Demo ===\n")

    # Example 1: Handwritten notes image
    print("--- Example 1: Handwritten notes ---")
    try:
        data = extract_handwritten_text(HANDWRITING_IMAGE_URL)
        print_handwriting_results(data)
    except Exception as exc:
        print(f"Example 1 error: {exc}")

    # Example 2: Historical handwriting (more challenging)
    print("\n--- Example 2: Historical handwriting ---")
    try:
        data2 = extract_handwritten_text(MIXED_HANDWRITING_URL)
        print_handwriting_results(data2)
    except Exception as exc:
        print(f"Example 2 error: {exc}")

    # Print accuracy tips
    handwriting_quality_tips()
