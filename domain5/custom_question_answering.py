"""
custom_question_answering.py
============================
Demonstrates the full Custom Question Answering (CQA) lifecycle using the
Azure AI Language SDK.

Exam skills (AI-102 Domain 5):
    - "Create a custom question answering project"
    - "Add question-and-answer pairs"
    - "Train, test, and publish a knowledge base"
    - "Create a multi-turn conversation"
    - "Add alternate phrasing and chit-chat"
    - "Export a knowledge base"

Concepts covered:
- Creating a CQA project
- Adding QnA pairs with alternate phrasings (question variants)
- Importing QnA content from a URL
- Adding chit-chat pairs (personality)
- Deploying the knowledge base
- Querying with follow-up prompt (multi-turn / chitchat)
- Exporting the knowledge base to JSON
- Handling confidence scores and answer sources

SDK classes used:
    AuthoringClient     – QuestionAnsweringAuthoringClient (manage KB)
    RuntimeClient       – QuestionAnsweringClient (query at runtime)

Required env vars:
    AZURE_LANGUAGE_ENDPOINT  – e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_LANGUAGE_KEY       – 32-character key from Azure portal

Install:
    pip install azure-ai-language-questionanswering python-dotenv
"""

import os
import json
import time
from dotenv import load_dotenv

from azure.ai.language.questionanswering.authoring import AuthoringClient
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.ai.language.questionanswering import models as qna_models
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

ENDPOINT = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
KEY = os.environ.get("AZURE_LANGUAGE_KEY")

PROJECT_NAME = "AzureAIFAQ"
DEPLOYMENT_NAME = "production"


def get_authoring_client() -> AuthoringClient:
    """Create an authenticated QuestionAnsweringAuthoringClient."""
    if not ENDPOINT or not KEY:
        raise EnvironmentError(
            "AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY must be set in your .env file."
        )
    return AuthoringClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def get_runtime_client() -> QuestionAnsweringClient:
    """Create an authenticated QuestionAnsweringClient for querying."""
    if not ENDPOINT or not KEY:
        raise EnvironmentError(
            "AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY must be set in your .env file."
        )
    return QuestionAnsweringClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


# ---------------------------------------------------------------------------
# Step 1: Create a CQA project
# ---------------------------------------------------------------------------

def create_project(client: AuthoringClient) -> None:
    """
    Create a new Custom Question Answering project.

    language       – primary language for the knowledge base
    multilingualResource – allow questions in multiple languages
    defaultAnswer  – fallback when no answer matches above threshold
    """
    print("\n" + "=" * 60)
    print(f"STEP 1: CREATE PROJECT  '{PROJECT_NAME}'")
    print("=" * 60)

    project_body = {
        "description": "Azure AI FAQ knowledge base for AI-102 exam prep demo",
        "language": "en",
        "multilingualResource": False,
        "settings": {
            "defaultAnswer": (
                "I'm sorry, I don't have an answer for that question. "
                "Please visit the Azure documentation for more information."
            )
        },
    }

    try:
        client.create_project(project_name=PROJECT_NAME, body=project_body)
        print(f"  ✓ Project '{PROJECT_NAME}' created.")
    except HttpResponseError as exc:
        if exc.status_code == 409:
            print(f"  Project '{PROJECT_NAME}' already exists – continuing.")
        else:
            print(f"  [ERROR] {exc.message}")
            raise


# ---------------------------------------------------------------------------
# Step 2: Add QnA pairs with alternate phrasings
# ---------------------------------------------------------------------------

def add_qna_pairs(client: AuthoringClient) -> None:
    """
    Add question-and-answer pairs to the knowledge base.

    Each QnA pair can have:
        questions         – list of question variants / phrasings
        answer            – the answer text (supports Markdown)
        metadata          – key-value pairs for filtering at query time
        promptsOnAnswer   – follow-up prompts for multi-turn conversation
        source            – label for the source of this pair
    """
    print("\n" + "=" * 60)
    print("STEP 2: ADD QnA PAIRS WITH ALTERNATE PHRASINGS")
    print("=" * 60)

    qna_pairs = [
        {
            "id": 1,
            "questions": [
                "What is Azure AI Language?",
                "What is the Azure Language service?",
                "Tell me about Azure Cognitive Services for language",
            ],
            "answer": (
                "Azure AI Language is a cloud-based service that provides Natural Language "
                "Processing (NLP) features such as sentiment analysis, key phrase extraction, "
                "named entity recognition, language detection, and question answering. "
                "It is part of Azure AI Services (formerly Cognitive Services)."
            ),
            "metadata": [{"key": "topic", "value": "azure-ai-language"}],
            "source": "editorial",
            "activelearningEnabled": True,
        },
        {
            "id": 2,
            "questions": [
                "How do I get started with Azure?",
                "How can I create an Azure account?",
                "Where do I sign up for Azure?",
            ],
            "answer": (
                "To get started with Azure:\n"
                "1. Visit [https://azure.microsoft.com](https://azure.microsoft.com)\n"
                "2. Click **Start free** to create a free account\n"
                "3. You will receive $200 credit for 30 days\n"
                "4. Many services remain free tier after the trial period"
            ),
            "metadata": [{"key": "topic", "value": "getting-started"}],
            "source": "editorial",
            # Multi-turn: follow-up prompts appear after this answer
            "dialog": {
                "isContextOnly": False,
                "prompts": [
                    {
                        "displayOrder": 1,
                        "qnaId": 3,
                        "displayText": "What services are free?",
                    },
                    {
                        "displayOrder": 2,
                        "qnaId": 4,
                        "displayText": "How do I set up billing alerts?",
                    },
                ],
            },
        },
        {
            "id": 3,
            "questions": [
                "What Azure services are always free?",
                "Which services have a free tier?",
            ],
            "answer": (
                "Azure offers many always-free services including:\n"
                "- Azure Functions (1M requests/month)\n"
                "- Azure Blob Storage (5 GB LRS)\n"
                "- Azure SQL Database (32 GB)\n"
                "- Azure AI Language (5,000 text records/month)\n"
                "- Azure App Service (10 web apps, F1 tier)\n\n"
                "Visit the [Azure free services page](https://azure.microsoft.com/free/free-account-faq/) "
                "for the full list."
            ),
            "metadata": [{"key": "topic", "value": "pricing"}],
            "source": "editorial",
        },
        {
            "id": 4,
            "questions": [
                "How do I set up billing alerts?",
                "How can I avoid unexpected Azure charges?",
            ],
            "answer": (
                "To set up billing alerts in Azure:\n"
                "1. Go to the Azure portal → **Cost Management + Billing**\n"
                "2. Select **Budgets** and click **+ Add**\n"
                "3. Set a monthly budget amount\n"
                "4. Configure alert thresholds (e.g. 80%, 100%)\n"
                "5. Add email recipients for notifications"
            ),
            "metadata": [{"key": "topic", "value": "billing"}],
            "source": "editorial",
        },
        {
            "id": 5,
            "questions": [
                "What is the AI-102 exam?",
                "What does the Azure AI-102 certification cover?",
            ],
            "answer": (
                "The Microsoft Azure AI Engineer Associate (AI-102) exam tests your ability to "
                "design and implement AI solutions using Azure AI Services. Exam domains include:\n"
                "- Plan and manage Azure AI solutions (15-20%)\n"
                "- Implement decision support solutions (10-15%)\n"
                "- Implement computer vision solutions (15-20%)\n"
                "- Implement natural language processing solutions (15-20%)\n"
                "- Implement knowledge mining and document intelligence (15-20%)\n"
                "- Implement generative AI solutions (15-20%)"
            ),
            "metadata": [{"key": "topic", "value": "certification"}],
            "source": "editorial",
        },
    ]

    update_body = {"add": {"qnas": qna_pairs}}

    try:
        poller = client.begin_update_qnas(
            project_name=PROJECT_NAME,
            update_qnas=update_body,
        )
        poller.result()
        print(f"  ✓ Added {len(qna_pairs)} QnA pairs.")
    except HttpResponseError as exc:
        print(f"  [ERROR] {exc.message}")
        raise


# ---------------------------------------------------------------------------
# Step 3: Import content from a URL
# ---------------------------------------------------------------------------

def add_url_source(client: AuthoringClient) -> None:
    """
    Import QnA content by pointing the service at a URL.

    The service will fetch the page and extract question-answer pairs from
    structured content (tables, FAQ pages, etc.).

    Exam tip: Supported source types include URLs (web pages), files uploaded
    to the portal, and manually entered QnA pairs.  The service auto-parses
    FAQ-style pages with question-answer structures.
    """
    print("\n" + "=" * 60)
    print("STEP 3: ADD URL SOURCE")
    print("=" * 60)

    source_url = "https://learn.microsoft.com/en-us/azure/ai-services/language-service/question-answering/faq"

    update_body = {
        "add": {
            "sources": [
                {
                    "displayName": "Azure CQA FAQ",
                    "sourceUri": source_url,
                    "sourceKind": "url",
                    "contentStructureKind": "unstructured",
                }
            ]
        }
    }

    try:
        poller = client.begin_update_sources(
            project_name=PROJECT_NAME,
            update_sources=update_body,
        )
        poller.result()
        print(f"  ✓ URL source added: {source_url}")
    except HttpResponseError as exc:
        print(f"  [WARNING] Could not add URL source: {exc.message}")
        print("    (This is expected if the URL is unreachable from the service)")


# ---------------------------------------------------------------------------
# Step 4: Add chit-chat
# ---------------------------------------------------------------------------

def add_chit_chat(client: AuthoringClient) -> None:
    """
    Add chit-chat pairs to give the bot a conversational personality.

    Built-in chit-chat styles: Professional, Friendly, Witty, Caring, Enthusiastic
    Adding chit-chat imports hundreds of pre-defined social conversation QnA pairs.

    Exam tip: Chit-chat makes the bot feel more natural by handling small-talk
    questions like "How are you?" or "Tell me a joke."
    """
    print("\n" + "=" * 60)
    print("STEP 4: ADD CHIT-CHAT  (Friendly personality)")
    print("=" * 60)

    update_body = {
        "add": {
            "sources": [
                {
                    "displayName": "Chit-chat Friendly",
                    "sourceKind": "url",
                    "sourceUri": "qna_chitchat_friendly.tsv",
                    "contentStructureKind": "unstructured",
                }
            ]
        }
    }

    # Note: Chit-chat is typically added via the portal or REST API with the
    # special chit-chat source kind.  The authoring SDK approach uses a
    # special URL token.  Showing the pattern here.
    print(
        "  [INFO] Chit-chat is typically configured via the Azure Language Studio "
        "portal.\n"
        "         The REST API approach uses the source type 'chatty' with a style parameter.\n"
        "         Skipping automated add to avoid API format inconsistency.\n"
    )


# ---------------------------------------------------------------------------
# Step 5: Deploy the knowledge base
# ---------------------------------------------------------------------------

def deploy_knowledge_base(client: AuthoringClient) -> None:
    """
    Deploy the knowledge base to the 'production' slot.

    After deployment, the knowledge base is live and queryable via the
    QuestionAnsweringClient (runtime client).

    Exam tip: CQA does not require a separate 'training' step like LUIS/CLU –
    deploying the project automatically prepares it for queries.
    """
    print("\n" + "=" * 60)
    print(f"STEP 5: DEPLOY  →  '{DEPLOYMENT_NAME}'")
    print("=" * 60)

    try:
        poller = client.begin_deploy_project(
            project_name=PROJECT_NAME,
            deployment_name=DEPLOYMENT_NAME,
        )
        poller.result()
        print(f"  ✓ Knowledge base deployed to '{DEPLOYMENT_NAME}'.")
    except HttpResponseError as exc:
        print(f"  [ERROR] {exc.message}")
        raise


# ---------------------------------------------------------------------------
# Step 6: Query the knowledge base
# ---------------------------------------------------------------------------

def query_knowledge_base(client: QuestionAnsweringClient) -> None:
    """
    Ask questions against the deployed knowledge base.

    Response includes:
        answers[].answer          – the answer text
        answers[].confidence      – confidence score (0.0 to 1.0)
        answers[].source          – source label
        answers[].metadata        – attached metadata
        answers[].dialog.prompts  – follow-up prompts (for multi-turn)

    Multi-turn:
        To continue a conversation, pass the qnaId of the selected answer
        as context.previousQnaId in the next request.

    Exam tip: The confidenceThreshold parameter (0-1) filters out answers
    below the specified confidence.  Tune this to avoid showing low-quality answers.
    """
    print("\n" + "=" * 60)
    print("STEP 6: QUERY KNOWLEDGE BASE")
    print("=" * 60)

    questions = [
        "What is Azure AI Language?",
        "How do I get started with Azure?",
        "What is the AI-102 exam about?",
        "How do I avoid surprise charges on my bill?",
        "What is the capital of France?",    # should hit default answer
    ]

    for question in questions:
        print(f"\n  Q: \"{question}\"")

        try:
            response = client.get_answers(
                question=question,
                project_name=PROJECT_NAME,
                deployment_name=DEPLOYMENT_NAME,
                confidence_threshold=0.3,
                top=3,
                include_unstructured_sources=True,
            )
        except HttpResponseError as exc:
            print(f"  [ERROR] {exc.message}")
            continue

        if not response.answers:
            print("  A: (no answers returned)")
            continue

        best = response.answers[0]
        print(f"  A: {best.answer[:200]}{'…' if len(best.answer) > 200 else ''}")
        print(f"     Confidence: {best.confidence:.2f}  |  Source: {best.source}")

        # Show follow-up prompts if available (multi-turn)
        if best.dialog and best.dialog.prompts:
            print("     Follow-up prompts:")
            for prompt in best.dialog.prompts:
                print(f"       • {prompt.display_text}")


# ---------------------------------------------------------------------------
# Step 7: Export the knowledge base
# ---------------------------------------------------------------------------

def export_knowledge_base(client: AuthoringClient) -> None:
    """
    Export the knowledge base to JSON for backup or migration.

    Exam tip: Exporting allows you to import the same knowledge base into
    another Azure AI Language resource or migrate between environments.
    The exported JSON can be re-imported with begin_import_project().
    """
    print("\n" + "=" * 60)
    print("STEP 7: EXPORT KNOWLEDGE BASE")
    print("=" * 60)

    try:
        poller = client.begin_export(
            project_name=PROJECT_NAME,
            export_type="JSON",
            asset_kind="qnas",
        )
        result = poller.result()

        # The result contains a URL to download the exported JSON
        export_url = result.get("resultUrl", "")
        print(f"  ✓ Export URL: {export_url}")
        print(
            "    Download this URL to get a JSON file you can import into another resource."
        )

    except HttpResponseError as exc:
        print(f"  [ERROR] Export failed: {exc.message}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point – run the full CQA lifecycle."""
    print("Azure AI Language – Custom Question Answering Demo")
    print("Endpoint:", ENDPOINT or "(not set)")

    authoring_client = get_authoring_client()
    runtime_client = get_runtime_client()

    create_project(authoring_client)
    add_qna_pairs(authoring_client)
    add_url_source(authoring_client)
    add_chit_chat(authoring_client)
    deploy_knowledge_base(authoring_client)
    query_knowledge_base(runtime_client)
    export_knowledge_base(authoring_client)

    print("\nDone.")


if __name__ == "__main__":
    main()
