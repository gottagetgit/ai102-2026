"""
custom_translator.py
====================
Demonstrates the Custom Translator workflow: preparing parallel training data,
creating a project, uploading document pairs, training a custom translation
model, and using it to translate text with the standard Translator API.

Exam skill: "Implement custom translation, including training, improving, and
            publishing a custom model" (AI-102 Domain 5)

Concepts covered:
- Custom Translator REST API (management plane)
- Parallel document pairs for training (source + target language documents)
- Creating a Custom Translator workspace and project
- Uploading training documents
- Triggering a training job and polling for completion
- Publishing a custom model to a deployment region
- Using the trained model with the standard /translate endpoint via
  the 'category' parameter (custom model category ID)
- Improving the model: adding more training data and retraining

Architecture:
    ┌─────────────────────┐
    │  Parallel Documents  │  (same content in EN + FR, sentence-aligned)
    │  train.en / train.fr │
    └────────┬────────────┘
             │ upload
    ┌─────────▼────────────────────────────┐
    │  Custom Translator Portal / REST API  │
    │  - Create workspace                   │
    │  - Create project (EN→FR, Tech domain)│
    │  - Upload document set                │
    │  - Train model                        │
    │  - Publish model (get category ID)    │
    └─────────┬────────────────────────────┘
              │  category=<model-id>
    ┌──────────▼──────────────────────┐
    │  /translate?to=fr&category=ID   │  ← same Translator REST API
    │  Custom model inference         │
    └─────────────────────────────────┘

Note on the Custom Translator REST API:
    The Custom Translator uses a separate management REST API (not the
    azure-ai-translation-* Python SDK).  This script uses the 'requests'
    library with OAuth 2.0 (service-principal) or subscription-key auth.

Required env vars:
    AZURE_TRANSLATOR_KEY            – subscription key for text translation
    AZURE_TRANSLATOR_ENDPOINT       – e.g. https://api.cognitive.microsofttranslator.com
    AZURE_TRANSLATOR_REGION         – e.g. eastus
    CUSTOM_TRANSLATOR_API_KEY       – key for the Custom Translator management API
                                      (found in Custom Translator portal → Settings)

Optional (Custom Translator management):
    CUSTOM_TRANSLATOR_WORKSPACE_ID  – existing workspace ID (skips create step)
    CUSTOM_TRANSLATOR_PROJECT_ID    – existing project ID (skips create step)

Install:
    pip install requests python-dotenv
"""

import os
import json
import uuid
import time
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

TRANSLATOR_KEY = os.environ.get("AZURE_TRANSLATOR_KEY")
TRANSLATOR_ENDPOINT = os.environ.get(
    "AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com"
).rstrip("/")
TRANSLATOR_REGION = os.environ.get("AZURE_TRANSLATOR_REGION", "eastus")
CUSTOM_TRANSLATOR_API_KEY = os.environ.get("CUSTOM_TRANSLATOR_API_KEY", "")
WORKSPACE_ID = os.environ.get("CUSTOM_TRANSLATOR_WORKSPACE_ID", "")
PROJECT_ID = os.environ.get("CUSTOM_TRANSLATOR_PROJECT_ID", "")

# Custom Translator management API base URL
MGMT_BASE = "https://custom-api.cognitive.microsofttranslator.com"
MGMT_API_VERSION = "v1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _translate_headers() -> dict:
    """Auth headers for the standard Translator /translate endpoint."""
    return {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }


def _mgmt_headers() -> dict:
    """Auth headers for the Custom Translator management API."""
    return {
        "Ocp-Apim-Subscription-Key": CUSTOM_TRANSLATOR_API_KEY,
        "Content-Type": "application/json",
    }


def _mgmt_get(path: str) -> dict:
    """Perform a GET against the Custom Translator management API."""
    url = f"{MGMT_BASE}/{MGMT_API_VERSION}/{path}"
    resp = requests.get(url, headers=_mgmt_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _mgmt_post(path: str, body: dict) -> dict:
    """Perform a POST against the Custom Translator management API."""
    url = f"{MGMT_BASE}/{MGMT_API_VERSION}/{path}"
    resp = requests.post(url, headers=_mgmt_headers(), json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Sample parallel training data
# ---------------------------------------------------------------------------

PARALLEL_EN = """\
Azure is a cloud computing platform operated by Microsoft.
It offers a wide range of services including compute, storage, networking, and AI.
Custom Translator lets you build domain-specific translation models.
Our product documentation is available in English and French.
The service provides high-quality neural machine translation.
Technical terms specific to your industry are handled more accurately.
You can improve the model by adding more parallel training data.
"""

PARALLEL_FR = """\
Azure est une plateforme de cloud computing exploitée par Microsoft.
Elle offre une large gamme de services comprenant le calcul, le stockage, la mise en réseau et l'IA.
Custom Translator vous permet de créer des modèles de traduction spécifiques à un domaine.
La documentation de notre produit est disponible en anglais et en français.
Le service fournit une traduction automatique neuronale de haute qualité.
Les termes techniques spécifiques à votre secteur sont traités avec plus de précision.
Vous pouvez améliorer le modèle en ajoutant davantage de données d'entraînement parallèles.
"""


# ---------------------------------------------------------------------------
# Step 1: Explain the workflow (management API – requires portal setup)
# ---------------------------------------------------------------------------

def explain_workflow() -> None:
    """
    Print a step-by-step explanation of the Custom Translator workflow.

    The Custom Translator management API requires authentication via
    Azure Active Directory (service principal or user auth), which is
    beyond the scope of a single-file demo.

    This function shows the conceptual workflow and REST calls so exam
    candidates understand what each step does.
    """
    print("\n" + "=" * 60)
    print("CUSTOM TRANSLATOR WORKFLOW OVERVIEW")
    print("=" * 60)

    steps = [
        (
            "1. Create a workspace",
            "POST /workspaces",
            '{"name": "MyWorkspace", "regionCode": "eastus"}',
        ),
        (
            "2. Create a project (language pair + domain)",
            "POST /projects",
            json.dumps({
                "name": "EN-FR Technical",
                "languagePairId": 1,            # EN → FR
                "domainId": 1,                  # Technology domain
                "categoryDescriptor": "azure-docs",
                "workspaceId": "<workspace-id>",
            }, indent=4),
        ),
        (
            "3. Upload parallel document pairs",
            "POST /documents/import",
            "(multipart/form-data with source + target file, and document type = training)",
        ),
        (
            "4. Create and train a model",
            "POST /models",
            json.dumps({
                "name": "azure-docs-v1",
                "projectId": "<project-id>",
                "documentIds": ["<doc-id>"],
            }, indent=4),
        ),
        (
            "5. Poll training status",
            "GET /models/<model-id>",
            '→ Check "modelStatus": "trained" | "training" | "failed"',
        ),
        (
            "6. Publish the model",
            "POST /models/<model-id>/publish",
            '{"regionCode": "eastus"}',
        ),
        (
            "7. Use the model in translation",
            "POST https://api.cognitive.microsofttranslator.com/translate?to=fr&category=<categoryId>",
            '(Use the categoryId returned by the publish step)',
        ),
    ]

    for title, endpoint, payload in steps:
        print(f"\n  {title}")
        print(f"    Endpoint: {endpoint}")
        print(f"    Body    : {payload[:120]}{'…' if len(payload) > 120 else ''}")


# ---------------------------------------------------------------------------
# Step 2: Save sample parallel data to disk
# ---------------------------------------------------------------------------

def save_sample_parallel_data() -> None:
    """
    Write sample parallel training documents to disk.

    In a real scenario you would:
        - Gather thousands of sentence-aligned pairs
        - Ensure they are in the target domain (e.g. medical, legal, technical)
        - Provide at least 10,000 sentence pairs for best results
        - Include a dictionary file for domain-specific terminology

    File requirements:
        - Plain text, one sentence per line
        - Identical sentence count in source and target files
        - UTF-8 encoding
        - Maximum 10 MB per file
    """
    print("\n" + "=" * 60)
    print("STEP 1: SAVE SAMPLE PARALLEL TRAINING DATA")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    en_path = os.path.join(output_dir, "train_parallel.en.txt")
    fr_path = os.path.join(output_dir, "train_parallel.fr.txt")

    with open(en_path, "w", encoding="utf-8") as f:
        f.write(PARALLEL_EN)
    with open(fr_path, "w", encoding="utf-8") as f:
        f.write(PARALLEL_FR)

    en_lines = PARALLEL_EN.strip().count("\n") + 1
    print(f"  ✓ Saved {en_lines} parallel sentence pairs:")
    print(f"    EN: {en_path}")
    print(f"    FR: {fr_path}")
    print(
        "\n  Note: Real training needs ≥ 10,000 sentence pairs for good quality."
    )


# ---------------------------------------------------------------------------
# Step 3: Demonstrate translation WITH a custom model category
# ---------------------------------------------------------------------------

def translate_with_custom_model(category_id: str, texts: list[str]) -> None:
    """
    Translate text using a custom-trained model by specifying its category ID.

    Once your Custom Translator model is trained and published, Azure assigns
    it a category ID (a GUID).  Pass this as the 'category' query parameter
    to the standard /translate endpoint to use your custom model instead of
    the default generic neural model.

    Exam tip: The 'category' parameter is the only change needed in your
    client code to switch from the generic model to your custom model.
    The rest of the API call is identical.
    """
    if not TRANSLATOR_KEY:
        print("[SKIP] AZURE_TRANSLATOR_KEY not set – skipping translation demo.")
        return

    print("\n" + "=" * 60)
    print(f"STEP 2: TRANSLATE WITH CUSTOM MODEL  (category: {category_id})")
    print("=" * 60)

    url = f"{TRANSLATOR_ENDPOINT}/translate"
    params = {
        "api-version": "3.0",
        "to": "fr",
        "category": category_id,
    }
    body = [{"text": t} for t in texts]

    try:
        resp = requests.post(
            url, headers=_translate_headers(), params=params, json=body, timeout=30
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        print(f"  [ERROR] HTTP {exc.response.status_code}: {exc.response.text[:200]}")
        return
    except requests.RequestException as exc:
        print(f"  [ERROR] Request failed: {exc}")
        return

    for i, result in enumerate(resp.json()):
        print(f"\n  Source : {texts[i]}")
        for t in result.get("translations", []):
            print(f"  Custom : {t['text']}")


# ---------------------------------------------------------------------------
# Step 4: Compare generic vs custom model translation
# ---------------------------------------------------------------------------

def compare_generic_vs_custom(category_id: str, texts: list[str]) -> None:
    """
    Translate the same texts with both the generic model and the custom model,
    showing the difference in domain-specific terminology handling.

    In technical domains the custom model should produce more accurate
    translations of product names, acronyms, and domain vocabulary.
    """
    if not TRANSLATOR_KEY:
        print("[SKIP] AZURE_TRANSLATOR_KEY not set – skipping comparison demo.")
        return

    print("\n" + "=" * 60)
    print("STEP 3: COMPARE GENERIC vs CUSTOM MODEL")
    print("=" * 60)

    url = f"{TRANSLATOR_ENDPOINT}/translate"
    base_params = {"api-version": "3.0", "to": "fr"}
    body = [{"text": t} for t in texts]

    def _call(params: dict) -> list[str]:
        resp = requests.post(
            url, headers=_translate_headers(), params=params, json=body, timeout=30
        )
        resp.raise_for_status()
        return [r["translations"][0]["text"] for r in resp.json()]

    try:
        generic_results = _call(base_params)
        custom_results = _call({**base_params, "category": category_id})
    except requests.HTTPError as exc:
        print(f"  [ERROR] HTTP {exc.response.status_code}: {exc.response.text[:200]}")
        return

    for i, text in enumerate(texts):
        print(f"\n  Source  : {text}")
        print(f"  Generic : {generic_results[i]}")
        print(f"  Custom  : {custom_results[i]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    print("Azure Custom Translator Demo")
    print("Translator Endpoint:", TRANSLATOR_ENDPOINT)

    # Always show the workflow explanation
    explain_workflow()

    # Save sample parallel data files
    save_sample_parallel_data()

    # -------------------------------------------------------------------
    # If you have already trained and published a model, set CATEGORY_ID
    # below (or read it from an env var) to run the live translation demos.
    # -------------------------------------------------------------------
    CATEGORY_ID = os.environ.get("CUSTOM_TRANSLATOR_CATEGORY_ID", "")

    if CATEGORY_ID:
        test_texts = [
            "Azure Blob Storage provides scalable object storage for unstructured data.",
            "The Cognitive Services endpoint must be set in your environment variables.",
            "Custom Translator improves translation accuracy for domain-specific content.",
        ]
        translate_with_custom_model(CATEGORY_ID, test_texts)
        compare_generic_vs_custom(CATEGORY_ID, test_texts)
    else:
        print(
            "\n[INFO] To test live translation with a custom model:\n"
            "  1. Complete the workflow steps above in the Custom Translator portal\n"
            "     (https://portal.customtranslator.azure.ai)\n"
            "  2. Publish your trained model to get a Category ID\n"
            "  3. Add CUSTOM_TRANSLATOR_CATEGORY_ID=<your-id> to your .env file\n"
            "  4. Re-run this script to see custom model translation results\n"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
