"""
language_detection.py
=====================
Demonstrates detecting the language of text samples using Azure AI Language.

Exam skill: "Detect the language used in text" (AI-102 Domain 5)

Concepts covered:
- Passing documents without a language hint (auto-detection)
- Reading the detected language name and ISO 639-1 code
- Confidence scores for each detection
- Handling ambiguous / mixed-script input

Required env vars:
    AZURE_LANGUAGE_ENDPOINT  – e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_LANGUAGE_KEY       – 32-character key from Azure portal

Install:
    pip install azure-ai-textanalytics python-dotenv
"""

import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

ENDPOINT = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
KEY = os.environ.get("AZURE_LANGUAGE_KEY")

# ---------------------------------------------------------------------------
# Sample documents in various languages – no language hint provided so the
# service must detect entirely from the text content.
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {"id": "1",  "text": "Hello, how are you today? I hope everything is going well."},
    {"id": "2",  "text": "Hola, ¿cómo estás hoy? Espero que todo vaya bien."},
    {"id": "3",  "text": "Bonjour, comment allez-vous aujourd'hui ? J'espère que tout va bien."},
    {"id": "4",  "text": "Guten Tag! Wie geht es Ihnen? Ich hoffe, es geht Ihnen gut."},
    {"id": "5",  "text": "こんにちは。今日はどのようにお過ごしですか？"},
    {"id": "6",  "text": "مرحبًا، كيف حالك اليوم؟ أتمنى أن تكون بخير."},
    {"id": "7",  "text": "Привет! Как ты сегодня? Надеюсь, у тебя всё хорошо."},
    {"id": "8",  "text": "Olá, como vai você hoje? Espero que tudo esteja bem."},
    {"id": "9",  "text": "नमस्ते, आप कैसे हैं? उम्मीद है सब ठीक है।"},
    # Ambiguous: very short text – the service may show low confidence
    {"id": "10", "text": "OK"},
    # Mixed-language text – service picks the dominant language
    {"id": "11", "text": "Hello world. Hola mundo. Bonjour le monde."},
]


def get_client() -> TextAnalyticsClient:
    """Create and return an authenticated TextAnalyticsClient."""
    if not ENDPOINT or not KEY:
        raise EnvironmentError(
            "AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY must be set in your .env file."
        )
    return TextAnalyticsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def detect_languages(client: TextAnalyticsClient) -> None:
    """
    Call detect_language on all sample documents and print results.

    The response for each document includes:
        primary_language.name            – e.g. "English"
        primary_language.iso6391_name    – e.g. "en"
        primary_language.confidence_score – 0.0 to 1.0

    A confidence score < 1.0 can indicate:
        - Very short text (insufficient signal)
        - Mixed-language content
        - Ambiguous characters shared across languages

    Exam tip: If the service cannot determine the language it returns
    iso6391_name = "(Unknown)" with confidence_score = 0.0.
    """
    print("\n" + "=" * 60)
    print("LANGUAGE DETECTION")
    print("=" * 60)
    print(f"  {'ID':<4} {'Detected Language':<25} {'ISO Code':<10} {'Confidence'}")
    print("  " + "-" * 55)

    try:
        response = client.detect_language(DOCUMENTS)
    except Exception as exc:
        print(f"[ERROR] detect_language failed: {exc}")
        return

    for doc in response:
        if doc.is_error:
            print(f"  {doc.id:<4} ERROR: {doc.error.message}")
            continue

        lang = doc.primary_language
        confidence_str = f"{lang.confidence_score:.2f}"
        # Flag low-confidence results for attention
        if lang.confidence_score < 0.8:
            confidence_str += "  ⚠ low confidence"

        print(
            f"  {doc.id:<4} {lang.name:<25} {lang.iso6391_name:<10} {confidence_str}"
        )


def main() -> None:
    """Entry point."""
    print("Azure AI Language – Language Detection Demo")
    print("Endpoint:", ENDPOINT or "(not set)")

    client = get_client()
    detect_languages(client)
    print("\nDone.")


if __name__ == "__main__":
    main()
