"""
document_translation.py
========================
Demonstrates the asynchronous document translation workflow using the Azure
Document Translation service.

Exam skill: "Translate text and documents by using the Azure Translator in
            Foundry Tools service" (AI-102 Domain 5)

Concepts covered:
- The begin_translation() pattern (long-running operation / LRO)
- Translating all documents in a source blob container to a target container
- Monitoring job status and polling until completion
- Retrieving per-document results (status, character count, errors)
- Supported document formats (DOCX, PDF, HTML, XLSX, PPTX, TXT, …)

Architecture:
    ┌──────────────┐     begin_translation()     ┌──────────────────────┐
    │  Source Blob │ ─────────────────────────►  │  Document Translation │
    │  Container   │                             │  Service (async job)  │
    └──────────────┘                             └──────────────┬───────┘
                                                                │
    ┌──────────────┐         polling result       │
    │  Target Blob │ ◄──────────────────────────  │
    │  Container   │                             (translated files land here)
    └──────────────┘

Required env vars:
    AZURE_TRANSLATOR_ENDPOINT  – Document translation endpoint:
                                 https://<resource>.cognitiveservices.azure.com/
                                 (NOT the global api.cognitive.microsofttranslator.com)
    AZURE_TRANSLATOR_KEY       – Subscription key
    AZURE_SOURCE_SAS_URL       – Full SAS URL to the source blob container
                                 (read permission required)
    AZURE_TARGET_SAS_URL       – Full SAS URL to the target blob container
                                 (write permission required)

Install:
    pip install azure-ai-translation-document python-dotenv
"""

import os
import time
from dotenv import load_dotenv

from azure.ai.translation.document import DocumentTranslationClient, TranslationTarget
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

ENDPOINT = os.environ.get("AZURE_TRANSLATOR_ENDPOINT")
KEY = os.environ.get("AZURE_TRANSLATOR_KEY")
SOURCE_SAS_URL = os.environ.get("AZURE_SOURCE_SAS_URL")
TARGET_SAS_URL = os.environ.get("AZURE_TARGET_SAS_URL")


def get_client() -> DocumentTranslationClient:
    """Create and return an authenticated DocumentTranslationClient."""
    if not ENDPOINT or not KEY:
        raise EnvironmentError(
            "AZURE_TRANSLATOR_ENDPOINT and AZURE_TRANSLATOR_KEY must be set in your .env file."
        )
    return DocumentTranslationClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def translate_documents(
    client: DocumentTranslationClient,
    source_url: str,
    target_url: str,
    target_language: str = "fr",
) -> None:
    """
    Submit an async document translation job and poll until complete.

    Parameters
    ----------
    source_url      : SAS URL of the blob container holding source documents
    target_url      : SAS URL of the destination blob container
    target_language : BCP-47 language tag for the translation target

    The begin_translation() method returns a DocumentTranslationLROPoller.
    Calling .result() blocks until the job finishes (polling automatically).
    Alternatively, use .status() / .wait() for non-blocking patterns.

    Exam tip: A single job can specify multiple TranslationTarget objects,
    each with a different target language, producing translated copies of
    every document in the source container.
    """
    print("\n" + "=" * 60)
    print("ASYNC DOCUMENT TRANSLATION")
    print(f"  Target language: {target_language}")
    print("=" * 60)

    targets = [
        TranslationTarget(
            target_url=target_url,
            language=target_language,
        )
    ]

    try:
        print("  Submitting translation job…")
        # begin_translation starts the async job and returns a poller
        poller = client.begin_translation(
            source_url=source_url,
            target_url=target_url,
            target_language=target_language,
        )

        print(f"  Job created. ID: {poller.id}")
        print("  Polling for completion (this may take several minutes)…")

        # Poll status manually every 10 seconds for visibility
        while not poller.done():
            status = poller.status()
            print(f"    Status: {status}")
            time.sleep(10)

        # Retrieve the final result – this also raises on terminal failure
        result = poller.result()

    except HttpResponseError as exc:
        print(f"[ERROR] Translation job failed: {exc.message}")
        return
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}")
        return

    # ---- Per-document results ----------------------------------------------
    print("\n  Document results:")
    succeeded = 0
    failed = 0

    for doc in result:
        if doc.status == "Succeeded":
            succeeded += 1
            print(f"    ✓ {doc.source_document_url.split('/')[-1].split('?')[0]}")
            print(f"      → {doc.translated_document_url.split('/')[-1].split('?')[0]}")
            print(f"      Characters: {doc.characters_charged}  "
                  f"Language: {doc.detected_language}")
        else:
            failed += 1
            print(f"    ✗ {doc.source_document_url.split('/')[-1].split('?')[0]}")
            if doc.error:
                print(f"      Error: {doc.error.message}")

    print(f"\n  Summary: {succeeded} succeeded, {failed} failed.")


def list_supported_formats(client: DocumentTranslationClient) -> None:
    """
    Print the document formats that the translation service supports.

    Exam tip: The service supports Word, Excel, PowerPoint, PDF, HTML,
    plain text, OpenDocument formats, and more.
    """
    print("\n" + "=" * 60)
    print("SUPPORTED DOCUMENT FORMATS")
    print("=" * 60)

    try:
        formats = client.get_supported_document_formats()
        for fmt in formats:
            ext_list = ", ".join(fmt.file_extensions)
            print(f"  {fmt.format:<20} extensions: {ext_list}")
    except Exception as exc:
        print(f"[ERROR] Could not retrieve formats: {exc}")


def main() -> None:
    """Entry point."""
    print("Azure Document Translation Demo")
    print("Endpoint:", ENDPOINT or "(not set)")

    client = get_client()

    # Show supported formats (doesn't need blob storage)
    list_supported_formats(client)

    # Run a translation job if blob storage env vars are configured
    if SOURCE_SAS_URL and TARGET_SAS_URL:
        translate_documents(
            client=client,
            source_url=SOURCE_SAS_URL,
            target_url=TARGET_SAS_URL,
            target_language="fr",
        )
    else:
        print(
            "\n[INFO] Skipping translation job – AZURE_SOURCE_SAS_URL and "
            "AZURE_TARGET_SAS_URL are not set.\n"
            "       Set them to SAS URLs for your source and target blob "
            "containers to run a full translation job."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
