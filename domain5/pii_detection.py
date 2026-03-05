"""
pii_detection.py
================
Demonstrates detecting and redacting Personally Identifiable Information (PII)
in text using Azure AI Language.

Exam skill: "Detect personally identifiable information (PII) in text" (AI-102 Domain 5)

Concepts covered:
- Submitting a batch of documents for PII analysis
- Accessing entity text, category, sub-category, and confidence score
- Reading the service-generated redacted text
- PHI (Protected Health Information) recognition via the healthcare PII domain
- Best practices: never log raw PII; use the redacted version instead

Required env vars:
    AZURE_LANGUAGE_ENDPOINT  – e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_LANGUAGE_ENDPOINT
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
# Sample documents containing various PII types
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "1",
        "language": "en",
        "text": (
            "My name is John Smith and my email is john.smith@example.com. "
            "You can also reach me at (555) 867-5309."
        ),
    },
    {
        "id": "2",
        "language": "en",
        "text": (
            "Please ship the order to Alice Johnson, 742 Evergreen Terrace, "
            "Springfield, IL 62701. Her credit card number is 4111-1111-1111-1111, "
            "expiry 12/27, CVV 123."
        ),
    },
    {
        "id": "3",
        "language": "en",
        "text": (
            "The patient, Robert Davis (DOB: 03/14/1982), was admitted on 01/10/2025. "
            "His Social Security number is 123-45-6789 and his insurance ID is XYZ-987654."
        ),
    },
    {
        "id": "4",
        "language": "en",
        "text": (
            "Driver's license: A1234567. IP address: 192.168.1.100. "
            "Passport number: AB1234567. Bank account: 00123456789."
        ),
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


def detect_pii(client: TextAnalyticsClient) -> None:
    """
    Run PII entity recognition on all sample documents.

    Key PII categories returned by the service include:
        Person, PersonType, PhoneNumber, Organization, Address,
        Email, URL, IPAddress, DateTime, Quantity, AgeInformation,
        CreditCardNumber, BankAccountNumber, SocialSecurityNumber,
        DriversLicense, PassportNumber, MedicalCondition, etc.

    The 'redacted_text' field replaces each detected PII span with
    asterisks of the same length – ready to store/display safely.

    Exam tip: The 'categories_filter' parameter lets you request only
    specific PII categories to reduce false-positive noise.
    """
    print("\n" + "=" * 60)
    print("PII ENTITY RECOGNITION")
    print("=" * 60)

    try:
        # categories_filter is optional – omitting it returns all PII types
        response = client.recognize_pii_entities(
            DOCUMENTS,
            # categories_filter=["PhoneNumber", "Email"],  # uncomment to filter
        )
    except Exception as exc:
        print(f"[ERROR] recognize_pii_entities failed: {exc}")
        return

    for doc in response:
        if doc.is_error:
            print(f"\nDoc {doc.id} ERROR: {doc.error.message}")
            continue

        print(f"\n--- Document {doc.id} ---")

        # Always use the redacted version in logs / downstream storage
        print(f"  Redacted text:\n    {doc.redacted_text}")

        if not doc.entities:
            print("  (no PII detected)")
            continue

        print(f"\n  Detected PII entities ({len(doc.entities)}):")
        header = f"  {'Text':<30} {'Category':<25} {'Sub-category':<20} {'Score'}"
        print(header)
        print("  " + "-" * 90)

        for entity in sorted(doc.entities, key=lambda e: e.confidence_score, reverse=True):
            sub = entity.subcategory or ""
            # Mask the raw PII text in the terminal output too
            masked = "*" * len(entity.text)
            print(
                f"  {masked:<30} {entity.category:<25} {sub:<20} "
                f"{entity.confidence_score:.2f}"
            )


def main() -> None:
    """Entry point."""
    print("Azure AI Language – PII Detection Demo")
    print("Endpoint:", ENDPOINT or "(not set)")

    client = get_client()
    detect_pii(client)
    print("\nDone.")


if __name__ == "__main__":
    main()
