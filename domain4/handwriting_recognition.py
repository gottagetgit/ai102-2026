"""
handwriting_recognition.py
==========================
Demonstrates handwriting recognition and document text extraction using:
  1. Azure AI Vision Read API (ImageAnalysisClient)  – single images
  2. Azure AI Document Intelligence (prebuilt-read)  – multi-page PDFs and images

Exam Skill Mapping:
    - "Extract text from images and documents by using Azure AI Vision Read API"
    - "Extract handwritten text by using Azure AI Vision"
    - "Use Azure AI Document Intelligence to extract text from forms and documents"
    - "Distinguish between printed and handwritten text"

Required Environment Variables (.env):
    AZURE_VISION_ENDPOINT              - e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_VISION_KEY                   - Azure AI Vision key
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT - e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_DOCUMENT_INTELLIGENCE_KEY    - Document Intelligence key

Install:
    pip install azure-ai-vision-imageanalysis azure-ai-documentintelligence python-dotenv
"""

import os
from dotenv import load_dotenv

# Azure AI Vision SDK
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Azure AI Document Intelligence SDK
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentPage

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VISION_ENDPOINT    = os.environ["AZURE_VISION_ENDPOINT"]
VISION_KEY         = os.environ["AZURE_VISION_KEY"]
DOC_INT_ENDPOINT   = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
DOC_INT_KEY        = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]


# ---------------------------------------------------------------------------
# 1. Handwriting extraction via Azure AI Vision (Read API)
# ---------------------------------------------------------------------------

def extract_handwriting_vision(image_url: str) -> None:
    """
    Use Azure AI Vision v4 ImageAnalysisClient to extract text (including
    handwriting) from a single image URL.

    The READ visual feature is used.  Results include:
    - Pages → Lines → Words with bounding polygons and confidence scores.
    - Each word has a 'kind' property: 'printed' or 'handwritten'.
    """
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY),
    )

    print(f"\n[Vision Read API] Extracting text from: {image_url}")
    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.READ],
    )

    if not result.read or not result.read.blocks:
        print("No text detected.")
        return

    for block in result.read.blocks:
        for line in block.lines:
            print(f"  Line: {line.text}")
            for word in line.words:
                kind = getattr(word, 'kind', 'unknown')  # 'printed' | 'handwritten'
                print(f"    Word: {word.text!r:20s}  confidence={word.confidence:.3f}  kind={kind}")


def extract_text_vision_stream(image_path: str) -> None:
    """
    Variant: pass a local image file as a binary stream to the Vision API.
    """
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY),
    )

    print(f"\n[Vision Read API] Extracting text from local file: {image_path}")
    with open(image_path, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
        content_type="image/jpeg",  # adjust if PNG: "image/png"
    )

    if result.read:
        for block in result.read.blocks:
            for line in block.lines:
                print(f"  {line.text}")


# ---------------------------------------------------------------------------
# 2. Multi-page document extraction via Document Intelligence (prebuilt-read)
# ---------------------------------------------------------------------------

def extract_document_text(document_url: str) -> None:
    """
    Use Azure AI Document Intelligence with the prebuilt-read model to extract
    text from a multi-page PDF, image, or Office document.

    Results include:
    - Pages → Lines → Words with bounding boxes and confidence
    - Paragraphs (semantic blocks)
    - Selection marks (checkboxes)
    - Distinction between printed and handwritten spans
    """
    client = DocumentIntelligenceClient(
        endpoint=DOC_INT_ENDPOINT,
        credential=AzureKeyCredential(DOC_INT_KEY),
    )

    print(f"\n[Document Intelligence] Analyzing: {document_url}")
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body={"urlSource": document_url},
    )
    result: AnalyzeResult = poller.result()

    print(f"Document has {len(result.pages)} page(s)")
    for page in result.pages:
        print(f"\n-- Page {page.page_number} (width={page.width}, height={page.height}, unit={page.unit}) --")
        if page.lines:
            for line in page.lines:
                print(f"  Line: {line.content}")
        if page.words:
            for word in page.words:
                print(f"  Word: {word.content!r:25s}  confidence={word.confidence:.3f}")

    # Paragraphs (higher-level semantic structure)
    if result.paragraphs:
        print(f"\nFound {len(result.paragraphs)} paragraph(s):")
        for para in result.paragraphs[:5]:   # show first 5
            print(f"  [{para.role}] {para.content[:80]}..." if len(para.content) > 80 else f"  [{para.role}] {para.content}")


def extract_document_from_file(file_path: str) -> None:
    """
    Variant: pass a local PDF/image file to Document Intelligence.
    """
    client = DocumentIntelligenceClient(
        endpoint=DOC_INT_ENDPOINT,
        credential=AzureKeyCredential(DOC_INT_KEY),
    )

    print(f"\n[Document Intelligence] Analyzing local file: {file_path}")
    with open(file_path, "rb") as f:
        file_content = f.read()

    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body=file_content,
        content_type="application/pdf",   # or image/jpeg, image/png, etc.
    )
    result: AnalyzeResult = poller.result()

    for page in result.pages:
        print(f"\nPage {page.page_number}:")
        if page.lines:
            for line in page.lines:
                print(f"  {line.content}")


# ---------------------------------------------------------------------------
# 3. Handwriting style detection (printed vs handwritten)
# ---------------------------------------------------------------------------

def detect_handwriting_style(document_url: str) -> None:
    """
    Document Intelligence returns style information that distinguishes
    handwritten text spans from printed text spans.
    """
    client = DocumentIntelligenceClient(
        endpoint=DOC_INT_ENDPOINT,
        credential=AzureKeyCredential(DOC_INT_KEY),
    )

    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body={"urlSource": document_url},
    )
    result: AnalyzeResult = poller.result()

    if result.styles:
        print("\nText styles detected:")
        for style in result.styles:
            is_hw = getattr(style, 'is_handwritten', False)
            confidence = getattr(style, 'confidence', 'N/A')
            spans = style.spans
            print(
                f"  Handwritten={is_hw}  confidence={confidence}  "
                f"spans={[(s.offset, s.length) for s in spans]}"
            )
    else:
        print("No style information available.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Handwriting via Vision API ---
    HANDWRITING_IMAGE_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/"
        "Handwriting_of_Frederic_Chopin.jpg/320px-Handwriting_of_Frederic_Chopin.jpg"
    )
    extract_handwriting_vision(HANDWRITING_IMAGE_URL)

    # --- Multi-page PDF via Document Intelligence ---
    SAMPLE_PDF_URL = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf"
    extract_document_text(SAMPLE_PDF_URL)

    # --- Handwriting style detection ---
    detect_handwriting_style(SAMPLE_PDF_URL)
