"""
content_understanding.py
=========================
Demonstrates Azure Content Understanding (formerly Azure AI Content Understanding)
via the Azure AI Foundry REST API. Content Understanding is a multi-modal
document analysis service that goes beyond OCR to understand, summarize,
classify, and extract structured data from documents, images, video, and audio.

This script covers:
  1. Create an Analyzer (pipeline definition) for:
       a. Document OCR + text extraction
       b. Document summarization and classification
       c. Entity and table extraction
       d. Image description and classification
  2. Submit documents/images/video/audio for analysis
  3. Poll for results and parse the structured output
  4. Multi-modal processing (document, image, video, audio)

Architecture:
  - Analyzers are reusable pipeline configurations (like a skillset in AI Search)
  - Each analyzer has a scenario (prebuilt template) + optional customizations
  - You create an analyzer ONCE, then call analyze() repeatedly

AI-102 Exam Skills Mapped:
  - Create an OCR pipeline to extract text from images and documents
  - Summarize, classify, and detect attributes of documents
  - Extract entities, tables, and images from documents
  - Process and ingest documents, images, videos, and audio with
    Azure Content Understanding in Foundry Tools

Required environment variables (see .env.sample):
  AZURE_AI_SERVICES_ENDPOINT   - https://<resource>.cognitiveservices.azure.com/
  AZURE_AI_SERVICES_KEY        - Multi-service or Content Understanding API key
  AZURE_AI_PROJECT_ENDPOINT    - https://<project>.api.azureml.ms/ (AI Foundry)
  AZURE_OPENAI_API_KEY         - For LLM-based summarization/extraction steps

Note: Azure Content Understanding is in preview (as of 2026). The REST API
paths shown here are based on the latest preview specification. Use
`api-version=2024-12-01-preview` or later.

Package: requests>=2.31.0, azure-identity>=1.15.0
"""

import json
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AI_SERVICES_ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"].rstrip("/")
AI_SERVICES_KEY = os.environ["AZURE_AI_SERVICES_KEY"]
API_VERSION = "2024-12-01-preview"

BASE_URL = f"{AI_SERVICES_ENDPOINT}/contentunderstanding"

HEADERS = {
    "Ocp-Apim-Subscription-Key": AI_SERVICES_KEY,
    "Content-Type": "application/json",
}

# Sample publicly accessible documents for testing
SAMPLE_PDF_URL = (
    "https://raw.githubusercontent.com/Azure-Samples/"
    "cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf"
)
SAMPLE_IMAGE_URL = (
    "https://learn.microsoft.com/azure/ai-services/computer-vision/media/"
    "quickstarts/presentation.png"
)


# ---------------------------------------------------------------------------
# Helper: REST API wrapper with error handling
# ---------------------------------------------------------------------------
def api_get(path: str, params: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=HEADERS, params={**(params or {}), "api-version": API_VERSION})
    resp.raise_for_status()
    return resp.json()


def api_put(path: str, body: dict) -> dict:
    url = f"{BASE_URL}{path}"
    resp = requests.put(
        url, headers=HEADERS, json=body, params={"api-version": API_VERSION}
    )
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, body: dict) -> requests.Response:
    url = f"{BASE_URL}{path}"
    resp = requests.post(
        url, headers=HEADERS, json=body, params={"api-version": API_VERSION}
    )
    resp.raise_for_status()
    return resp


def api_delete(path: str) -> bool:
    url = f"{BASE_URL}{path}"
    resp = requests.delete(url, headers=HEADERS, params={"api-version": API_VERSION})
    return resp.status_code in (200, 204)


# ---------------------------------------------------------------------------
# Helper: poll an async operation until complete
# ---------------------------------------------------------------------------
def poll_operation(operation_url: str, poll_interval: int = 3, max_attempts: int = 30) -> dict:
    """
    Poll a long-running operation URL (returned in the 'Operation-Location' header).
    Returns the final result dict.
    """
    for attempt in range(max_attempts):
        resp = requests.get(operation_url, headers=HEADERS)
        resp.raise_for_status()
        result = resp.json()

        status = result.get("status", "").lower()
        print(f"  Poll [{attempt+1}] status: {status}")

        if status == "succeeded":
            return result
        elif status in ("failed", "canceled"):
            error = result.get("error", {})
            raise RuntimeError(
                f"Operation {status}: {error.get('code')} — {error.get('message')}"
            )

        time.sleep(poll_interval)

    raise TimeoutError(f"Operation did not complete within {max_attempts * poll_interval} seconds")


# ---------------------------------------------------------------------------
# 1. Create an Analyzer (OCR + text extraction pipeline)
# ---------------------------------------------------------------------------
def create_document_ocr_analyzer(analyzer_id: str = "ai102-ocr-analyzer") -> dict:
    """
    Create a Content Understanding analyzer for OCR and text extraction.

    Scenario: 'documentIntelligence' — extracts text, layout, tables
    from documents using the Document Intelligence backend.

    An analyzer is a reusable pipeline; create it once and call analyze()
    many times. Analyzers persist until deleted.
    """
    print(f"\n{'='*60}")
    print(f" Creating OCR Analyzer: '{analyzer_id}'")
    print(f"{'='*60}")

    analyzer_definition = {
        "description": "AI-102 demo: OCR and text extraction analyzer",
        "scenario": "documentIntelligence",
        "fieldSchema": {
            "fields": {
                # Request the layout extractor to also return tables
                "tables": {
                    "type": "array",
                    "description": "Tables found in the document",
                },
                "paragraphs": {
                    "type": "array",
                    "description": "Paragraphs and their semantic roles",
                },
                "keyValuePairs": {
                    "type": "array",
                    "description": "Key-value pairs extracted from the document",
                },
            }
        },
        "config": {
            "returnDetails": True,
            "enableOcr": True,
            "enableLayout": True,
        },
    }

    try:
        result = api_put(f"/analyzers/{analyzer_id}", analyzer_definition)
        print(f"  Analyzer '{result.get('analyzerId', analyzer_id)}' created/updated.")
        return result
    except requests.HTTPError as e:
        print(f"  Error creating analyzer: {e.response.status_code} — {e.response.text[:200]}")
        return {}


# ---------------------------------------------------------------------------
# 2. Create a summarization + classification analyzer
# ---------------------------------------------------------------------------
def create_summarization_analyzer(analyzer_id: str = "ai102-summarization-analyzer") -> dict:
    """
    Create an analyzer that combines OCR with LLM-based summarization
    and document classification.

    This uses the 'contentExtraction' scenario with additional
    generative fields (summary, classification) powered by Azure OpenAI.
    """
    print(f"\n{'='*60}")
    print(f" Creating Summarization/Classification Analyzer: '{analyzer_id}'")
    print(f"{'='*60}")

    analyzer_definition = {
        "description": "Summarize and classify documents",
        "scenario": "contentExtraction",
        "fieldSchema": {
            "fields": {
                "summary": {
                    "type": "string",
                    "description": "A 2-3 sentence summary of the document's main content",
                    "method": "generate",   # LLM-generated field
                },
                "documentType": {
                    "type": "string",
                    "description": "The type of document",
                    "enum": ["invoice", "contract", "report", "email", "receipt", "other"],
                    "method": "classify",
                },
                "language": {
                    "type": "string",
                    "description": "Primary language of the document (ISO 639-1)",
                    "method": "extract",
                },
                "sentiment": {
                    "type": "string",
                    "description": "Overall sentiment/tone of the document",
                    "enum": ["positive", "neutral", "negative"],
                    "method": "classify",
                },
                "mainTopics": {
                    "type": "array",
                    "description": "List of main topics covered in the document",
                    "method": "generate",
                    "items": {"type": "string"},
                },
            }
        },
    }

    try:
        result = api_put(f"/analyzers/{analyzer_id}", analyzer_definition)
        print(f"  Analyzer '{result.get('analyzerId', analyzer_id)}' created.")
        return result
    except requests.HTTPError as e:
        print(f"  Error: {e.response.status_code} — {e.response.text[:200]}")
        return {}


# ---------------------------------------------------------------------------
# 3. Create an entity and table extraction analyzer
# ---------------------------------------------------------------------------
def create_entity_extraction_analyzer(analyzer_id: str = "ai102-entity-analyzer") -> dict:
    """
    Create an analyzer for extracting named entities and tables from documents.

    This is useful for:
      - Contract analysis (parties, dates, obligations)
      - Medical records (diagnoses, medications, dates)
      - Financial documents (amounts, parties, account numbers)
    """
    print(f"\n{'='*60}")
    print(f" Creating Entity Extraction Analyzer: '{analyzer_id}'")
    print(f"{'='*60}")

    analyzer_definition = {
        "description": "Extract entities and tables from business documents",
        "scenario": "contentExtraction",
        "fieldSchema": {
            "fields": {
                "people": {
                    "type": "array",
                    "description": "Names of people mentioned in the document",
                    "method": "extract",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string", "description": "Their role or title"},
                        },
                    },
                },
                "organizations": {
                    "type": "array",
                    "description": "Organizations and companies mentioned",
                    "method": "extract",
                    "items": {"type": "string"},
                },
                "dates": {
                    "type": "array",
                    "description": "Important dates mentioned (ISO 8601 format)",
                    "method": "extract",
                    "items": {"type": "string"},
                },
                "amounts": {
                    "type": "array",
                    "description": "Monetary amounts with currency",
                    "method": "extract",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                            "currency": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "tables": {
                    "type": "array",
                    "description": "Tabular data extracted from the document",
                    "method": "extract",
                },
            }
        },
    }

    try:
        result = api_put(f"/analyzers/{analyzer_id}", analyzer_definition)
        print(f"  Analyzer '{result.get('analyzerId', analyzer_id)}' created.")
        return result
    except requests.HTTPError as e:
        print(f"  Error: {e.response.status_code} — {e.response.text[:200]}")
        return {}


# ---------------------------------------------------------------------------
# 4. Analyze a document using a created analyzer
# ---------------------------------------------------------------------------
def analyze_document(
    analyzer_id: str,
    document_url: str,
    label: str = "Document Analysis",
) -> dict:
    """
    Submit a document for analysis using a named analyzer.

    The service returns an Operation-Location header for async polling.
    Supported inputs: URL, base64-encoded bytes, Azure Blob Storage URL
    Supported formats: PDF, DOCX, XLSX, PPTX, HTML, TXT, PNG, JPEG, TIFF, BMP
    """
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  Analyzer : {analyzer_id}")
    print(f"  Document : {document_url[:80]}")

    body = {
        "url": document_url,
    }

    try:
        resp = api_post(f"/analyzers/{analyzer_id}:analyze", body)

        # The response is 202 Accepted with an operation URL
        operation_url = resp.headers.get("Operation-Location")
        if not operation_url:
            print("  Warning: No Operation-Location header in response")
            print(f"  Response: {resp.text[:200]}")
            return {}

        print(f"  Analysis started. Polling: {operation_url[:80]}...")
        result = poll_operation(operation_url)

        # Pretty print extracted fields
        analyze_result = result.get("result", result)
        if "contents" in analyze_result:
            contents = analyze_result["contents"]
            print(f"\n  Content items returned: {len(contents)}")
            for content_item in contents[:2]:
                print(f"\n  Content type: {content_item.get('kind', 'document')}")
                fields = content_item.get("fields", {})
                if fields:
                    print(f"  Extracted fields ({len(fields)}):")
                    for field_name, field_val in fields.items():
                        val_str = json.dumps(field_val)[:120]
                        print(f"    {field_name}: {val_str}")

        return result

    except requests.HTTPError as e:
        print(f"  HTTP Error: {e.response.status_code} — {e.response.text[:300]}")
        return {}
    except (RuntimeError, TimeoutError) as e:
        print(f"  Analysis error: {e}")
        return {}


# ---------------------------------------------------------------------------
# 5. Image analysis using Content Understanding
# ---------------------------------------------------------------------------
def analyze_image(
    image_url: str,
    analyzer_id: str = "ai102-image-analyzer",
) -> dict:
    """
    Analyze an image with a Content Understanding analyzer configured
    for visual content (description, objects, text in image).

    First creates the analyzer if it doesn't exist.
    """
    print(f"\n{'='*60}")
    print(f" Image Analysis")
    print(f"{'='*60}")

    # Create a vision analyzer
    vision_analyzer = {
        "description": "AI-102 demo: image analysis",
        "scenario": "imageAnalysis",
        "fieldSchema": {
            "fields": {
                "caption": {
                    "type": "string",
                    "description": "A one-sentence description of the image",
                    "method": "generate",
                },
                "objects": {
                    "type": "array",
                    "description": "Objects and items visible in the image",
                    "method": "extract",
                    "items": {"type": "string"},
                },
                "text": {
                    "type": "string",
                    "description": "Any text visible in the image (OCR)",
                    "method": "extract",
                },
                "tags": {
                    "type": "array",
                    "description": "Descriptive tags for the image",
                    "method": "extract",
                    "items": {"type": "string"},
                },
            }
        },
    }

    try:
        api_put(f"/analyzers/{analyzer_id}", vision_analyzer)
        print(f"  Analyzer '{analyzer_id}' ready.")

        return analyze_document(
            analyzer_id=analyzer_id,
            document_url=image_url,
            label="Image Analysis",
        )
    except requests.HTTPError as e:
        print(f"  Error: {e.response.status_code} — {e.response.text[:200]}")
        return {}


# ---------------------------------------------------------------------------
# 6. Video analysis (Content Understanding supports video)
# ---------------------------------------------------------------------------
def analyze_video(video_url: str, analyzer_id: str = "ai102-video-analyzer") -> dict:
    """
    Analyze a video file using Azure Content Understanding.
    Extracts: transcript, scene descriptions, key frames, visual content.

    Note: Video analysis requires a video-capable endpoint and may take
    several minutes for longer videos.
    """
    print(f"\n{'='*60}")
    print(f" Video Analysis")
    print(f"{'='*60}")

    video_analyzer = {
        "description": "AI-102 demo: video content understanding",
        "scenario": "videoContentUnderstanding",
        "fieldSchema": {
            "fields": {
                "transcript": {
                    "type": "string",
                    "description": "Full spoken-word transcript of the video",
                    "method": "extract",
                },
                "summary": {
                    "type": "string",
                    "description": "A summary of the video content",
                    "method": "generate",
                },
                "chapters": {
                    "type": "array",
                    "description": "Main sections or chapters of the video",
                    "method": "generate",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "startTimeMs": {"type": "number"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "speakers": {
                    "type": "array",
                    "description": "Identified speakers in the video",
                    "method": "extract",
                    "items": {"type": "string"},
                },
            }
        },
    }

    try:
        api_put(f"/analyzers/{analyzer_id}", video_analyzer)
        print(f"  Video analyzer '{analyzer_id}' ready.")
        return analyze_document(
            analyzer_id=analyzer_id,
            document_url=video_url,
            label="Video Content Analysis",
        )
    except requests.HTTPError as e:
        print(f"  Error setting up video analyzer: {e.response.status_code}")
        return {}


# ---------------------------------------------------------------------------
# 7. Audio analysis (speech-to-text + understanding)
# ---------------------------------------------------------------------------
def analyze_audio(audio_url: str, analyzer_id: str = "ai102-audio-analyzer") -> dict:
    """
    Analyze an audio file — transcribe speech and extract structured information.
    Supports: MP3, WAV, MP4 (audio), FLAC, OGG
    """
    print(f"\n{'='*60}")
    print(f" Audio Analysis")
    print(f"{'='*60}")

    audio_analyzer = {
        "description": "AI-102 demo: audio analysis with transcription",
        "scenario": "audioContentUnderstanding",
        "fieldSchema": {
            "fields": {
                "transcript": {
                    "type": "string",
                    "description": "Full transcript of the audio",
                    "method": "extract",
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of what was discussed",
                    "method": "generate",
                },
                "actionItems": {
                    "type": "array",
                    "description": "Action items or tasks mentioned",
                    "method": "generate",
                    "items": {"type": "string"},
                },
                "keyTopics": {
                    "type": "array",
                    "description": "Main topics discussed",
                    "method": "generate",
                    "items": {"type": "string"},
                },
            }
        },
    }

    try:
        api_put(f"/analyzers/{analyzer_id}", audio_analyzer)
        return analyze_document(
            analyzer_id=analyzer_id,
            document_url=audio_url,
            label="Audio Content Analysis",
        )
    except requests.HTTPError as e:
        print(f"  Error: {e.response.status_code}")
        return {}


# ---------------------------------------------------------------------------
# 8. List and manage analyzers
# ---------------------------------------------------------------------------
def list_analyzers():
    """List all analyzers created in this resource."""
    print(f"\n{'='*60}")
    print(" Listing Content Understanding Analyzers")
    print(f"{'='*60}")

    try:
        result = api_get("/analyzers")
        analyzers = result.get("value", [])
        print(f"  Total analyzers: {len(analyzers)}")
        for analyzer in analyzers:
            print(
                f"  - {analyzer.get('analyzerId'):<40} "
                f"scenario={analyzer.get('scenario', 'N/A')}"
            )
        return analyzers
    except requests.HTTPError as e:
        print(f"  Error: {e.response.status_code} — {e.response.text[:200]}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Azure Content Understanding — Multi-modal Analysis Demo")
    print(f"Endpoint: {AI_SERVICES_ENDPOINT}")
    print(f"API Version: {API_VERSION}")

    try:
        # 1. List existing analyzers
        list_analyzers()

        # 2. Create analyzers
        create_document_ocr_analyzer()
        create_summarization_analyzer()
        create_entity_extraction_analyzer()

        # 3. Analyze a document with the OCR analyzer
        analyze_document(
            analyzer_id="ai102-ocr-analyzer",
            document_url=SAMPLE_PDF_URL,
            label="OCR: Extract Text from Invoice PDF",
        )

        # 4. Analyze the same document with the summarization analyzer
        analyze_document(
            analyzer_id="ai102-summarization-analyzer",
            document_url=SAMPLE_PDF_URL,
            label="Summarize and Classify Invoice",
        )

        # 5. Extract entities from the document
        analyze_document(
            analyzer_id="ai102-entity-analyzer",
            document_url=SAMPLE_PDF_URL,
            label="Extract Entities from Invoice",
        )

        # 6. Analyze an image
        analyze_image(image_url=SAMPLE_IMAGE_URL)

        # 7. List all created analyzers
        list_analyzers()

        print("\nContent Understanding demo complete!")
        print("\nExam Tips:")
        print("  - Content Understanding is the newer Foundry-based successor to")
        print("    Document Intelligence and Azure AI Vision combined")
        print("  - Analyzers persist and are reusable — create once, analyze many")
        print("  - Supports document, image, video, and audio in a unified API")
        print("  - 'generate' method uses LLM; 'extract' uses DI/Vision; 'classify' uses both")

    except requests.ConnectionError:
        print(f"Cannot connect to endpoint: {AI_SERVICES_ENDPOINT}")
        print("Check AZURE_AI_SERVICES_ENDPOINT in your .env file")
    except KeyError as e:
        print(f"Missing environment variable: {e}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
