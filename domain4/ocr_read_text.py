"""
ocr_read_text.py
================
Demonstrates text extraction from images using the Azure AI Vision
Read API (v4 ImageAnalysisClient) and optionally the Azure AI Document
Intelligence prebuilt-read model.

Exam Skill Mapping:
    - "Extract text from images by using Azure AI Vision"
    - "Use the Read API to extract text from images"
    - "Differentiate between synchronous and asynchronous OCR operations"
    - "Access OCR results including bounding polygons and confidence scores"

Required Environment Variables (.env):
    AZURE_VISION_ENDPOINT  - e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_VISION_KEY       - Azure AI Vision key

Optional (for Document Intelligence section):
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
    AZURE_DOCUMENT_INTELLIGENCE_KEY

Install:
    pip install azure-ai-vision-imageanalysis azure-ai-documentintelligence python-dotenv
"""

import os
from dotenv import load_dotenv

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VISION_ENDPOINT = os.environ["AZURE_VISION_ENDPOINT"]
VISION_KEY      = os.environ["AZURE_VISION_KEY"]

# Public images with visible text for demo purposes
PRINTED_TEXT_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/"
    "Atomist_quote_from_Democritus.png/320px-Atomist_quote_from_Democritus.png"
)
HANDWRITING_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/"
    "Handwriting_of_Frederic_Chopin.jpg/320px-Handwriting_of_Frederic_Chopin.jpg"
)


# ---------------------------------------------------------------------------
# 1. Basic OCR – extract all text from an image URL
# ---------------------------------------------------------------------------

def extract_text_from_url(image_url: str) -> None:
    """
    Synchronous OCR using Azure AI Vision v4 ImageAnalysisClient.
    Prints every line and word with bounding polygon and confidence.
    """
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY),
    )

    print(f"\nExtracting text from: {image_url}")
    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.READ],
    )

    if not result.read or not result.read.blocks:
        print("  No text detected.")
        return

    for block_idx, block in enumerate(result.read.blocks):
        print(f"\nBlock {block_idx}:")
        for line in block.lines:
            # Bounding polygon: list of (x, y) points
            poly = [(p.x, p.y) for p in line.bounding_polygon]
            print(f"  Line: {line.text!r:50s}  polygon={poly}")
            for word in line.words:
                w_poly = [(p.x, p.y) for p in word.bounding_polygon]
                print(f"    Word: {word.text!r:20s}  confidence={word.confidence:.3f}  polygon={w_poly}")


# ---------------------------------------------------------------------------
# 2. OCR from local file (binary stream)
# ---------------------------------------------------------------------------

def extract_text_from_file(image_path: str, content_type: str = "image/jpeg") -> None:
    """
    Reads a local image file and sends its bytes to the Vision Read API.
    content_type: 'image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/gif'
    """
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY),
    )

    print(f"\nExtracting text from local file: {image_path}")
    with open(image_path, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
        content_type=content_type,
    )

    if result.read:
        full_text = " ".join(
            word.text
            for block in result.read.blocks
            for line in block.lines
            for word in line.words
        )
        print(f"  Full text: {full_text}")


# ---------------------------------------------------------------------------
# 3. Language detection hint
# ---------------------------------------------------------------------------

def extract_text_with_language(image_url: str, language: str = "en") -> None:
    """
    Pass a language hint to improve OCR accuracy for non-English text.
    Supported language codes: https://aka.ms/cv-languages
    """
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY),
    )

    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.READ],
        language=language,
    )

    if result.read:
        for block in result.read.blocks:
            for line in block.lines:
                print(f"  [{language}] {line.text}")


# ---------------------------------------------------------------------------
# 4. Document Intelligence Read (async, multi-page)
# ---------------------------------------------------------------------------

def extract_text_document_intelligence(document_url: str) -> None:
    """
    Use Azure AI Document Intelligence prebuilt-read model for multi-page
    documents (PDFs, TIFF, Office files).

    Unlike Vision API (single image), Document Intelligence supports:
    - Multi-page PDFs and TIFFs
    - Office documents (Word, Excel, PowerPoint)
    - Structured output with paragraphs, tables, and selection marks
    """
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import AnalyzeResult
    except ImportError:
        print("azure-ai-documentintelligence not installed; skipping.")
        return

    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key      = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not endpoint or not key:
        print("Document Intelligence credentials not set; skipping.")
        return

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    print(f"\n[Document Intelligence] Extracting text from: {document_url}")
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body={"urlSource": document_url},
    )
    result: AnalyzeResult = poller.result()

    # Flat text content
    print(f"  Total pages: {len(result.pages)}")
    for page in result.pages:
        print(f"  Page {page.page_number}:")
        if page.lines:
            for line in page.lines:
                print(f"    {line.content}")


# ---------------------------------------------------------------------------
# 5. Utility: flatten all text to a single string
# ---------------------------------------------------------------------------

def get_all_text(image_url: str) -> str:
    """
    Returns all detected text from an image as a single newline-joined string.
    Useful for downstream NLP processing.
    """
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY),
    )

    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.READ],
    )

    lines = []
    if result.read:
        for block in result.read.blocks:
            for line in block.lines:
                lines.append(line.text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Azure AI Vision – OCR / Read API Demo")
    print("=" * 60)

    # Printed text
    print("\n--- Printed Text ---")
    extract_text_from_url(PRINTED_TEXT_URL)

    # Handwriting
    print("\n--- Handwriting ---")
    extract_text_from_url(HANDWRITING_URL)

    # Language hint (French)
    print("\n--- French language hint ---")
    extract_text_with_language(HANDWRITING_URL, language="fr")

    # Full text as string
    print("\n--- All text as string ---")
    text = get_all_text(PRINTED_TEXT_URL)
    print(text)

    # Document Intelligence (requires credentials)
    SAMPLE_PDF = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf"
    extract_text_document_intelligence(SAMPLE_PDF)
