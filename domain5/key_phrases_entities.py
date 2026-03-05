"""
key_phrases_entities.py
=======================
Demonstrates extracting key phrases and named entities from text using
Azure AI Language (azure-ai-textanalytics).

Exam skill: "Extract key phrases and entities" (AI-102 Domain 5)

Concepts covered:
- Batch processing multiple documents in a single API call
- Key phrase extraction
- Named Entity Recognition (NER) with categories and confidence scores
- Entity linking to well-known knowledge bases (Wikipedia)

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
# Sample documents – a mix of topics so we see varied entity categories
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "1",
        "language": "en",
        "text": (
            "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975, "
            "in Albuquerque, New Mexico. The company is headquartered in Redmond, Washington."
        ),
    },
    {
        "id": "2",
        "language": "en",
        "text": (
            "The Azure OpenAI Service provides REST API access to OpenAI's powerful language "
            "models including GPT-4. It is available in regions such as East US and West Europe."
        ),
    },
    {
        "id": "3",
        "language": "en",
        "text": (
            "Customer order #A1234 placed on 01/15/2025 for a Surface Laptop 5 costing $1,299.99. "
            "Ship to: Jane Doe, 123 Main St, Seattle WA 98101. Contact: jane.doe@example.com."
        ),
    },
    {
        "id": "4",
        "language": "es",
        "text": "El presidente de México visitó la Ciudad de México el lunes para hablar sobre la economía.",
    },
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


def extract_key_phrases(client: TextAnalyticsClient) -> None:
    """
    Call the extract_key_phrases API on all sample documents and print results.
    Key phrases represent the most salient topics in each piece of text.
    """
    print("\n" + "=" * 60)
    print("KEY PHRASE EXTRACTION")
    print("=" * 60)

    try:
        # The SDK accepts a list of dicts or TextDocumentInput objects
        response = client.extract_key_phrases(DOCUMENTS)
    except Exception as exc:
        print(f"[ERROR] extract_key_phrases failed: {exc}")
        return

    for doc in response:
        if doc.is_error:
            print(f"  Doc {doc.id} ERROR: {doc.error.message}")
            continue

        print(f"\n  Document {doc.id}:")
        if doc.key_phrases:
            for phrase in doc.key_phrases:
                print(f"    - {phrase}")
        else:
            print("    (no key phrases detected)")


def recognize_entities(client: TextAnalyticsClient) -> None:
    """
    Call the recognize_entities API and display entity category, sub-category,
    text, and confidence score for each recognised entity.

    Common entity categories returned by Azure:
        Person, PersonType, Location, Organization, Event,
        Product, Skill, Address, PhoneNumber, Email, URL,
        IP Address, DateTime, Quantity
    """
    print("\n" + "=" * 60)
    print("NAMED ENTITY RECOGNITION (NER)")
    print("=" * 60)

    try:
        response = client.recognize_entities(DOCUMENTS)
    except Exception as exc:
        print(f"[ERROR] recognize_entities failed: {exc}")
        return

    for doc in response:
        if doc.is_error:
            print(f"  Doc {doc.id} ERROR: {doc.error.message}")
            continue

        print(f"\n  Document {doc.id}:")
        if not doc.entities:
            print("    (no entities detected)")
            continue

        # Sort by confidence score descending for easier reading
        for entity in sorted(doc.entities, key=lambda e: e.confidence_score, reverse=True):
            sub = f" / {entity.subcategory}" if entity.subcategory else ""
            print(
                f"    [{entity.category}{sub}]  '{entity.text}'  "
                f"(confidence: {entity.confidence_score:.2f})"
            )


def recognize_linked_entities(client: TextAnalyticsClient) -> None:
    """
    Entity linking resolves recognised entities to entries in a knowledge base
    (Wikipedia by default).  Each linked entity has a URL to its Wikipedia page
    and a Bing Entity Search ID.
    """
    print("\n" + "=" * 60)
    print("ENTITY LINKING")
    print("=" * 60)

    # Only submit English documents for linking (the service supports several
    # languages but Spanish linking coverage is more limited)
    english_docs = [d for d in DOCUMENTS if d["language"] == "en"]

    try:
        response = client.recognize_linked_entities(english_docs)
    except Exception as exc:
        print(f"[ERROR] recognize_linked_entities failed: {exc}")
        return

    for doc in response:
        if doc.is_error:
            print(f"  Doc {doc.id} ERROR: {doc.error.message}")
            continue

        print(f"\n  Document {doc.id}:")
        if not doc.entities:
            print("    (no linked entities detected)")
            continue

        for entity in doc.entities:
            print(f"    Entity  : {entity.name}")
            print(f"    URL     : {entity.url}")
            print(f"    Score   : {entity.matches[0].confidence_score:.2f}")
            print()


def main() -> None:
    """Entry point – run all three demonstrations sequentially."""
    print("Azure AI Language – Key Phrases & Entity Recognition Demo")
    print("Endpoint:", ENDPOINT or "(not set)")

    client = get_client()

    extract_key_phrases(client)
    recognize_entities(client)
    recognize_linked_entities(client)

    print("\nDone.")


if __name__ == "__main__":
    main()
