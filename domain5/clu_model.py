"""
clu_model.py
============
Demonstrates the full lifecycle of a Conversational Language Understanding (CLU)
model using the Azure AI Language SDK.

Exam skills (AI-102 Domain 5):
    - "Create intents, entities, and add utterances"
    - "Train, evaluate, deploy, and test a language understanding model"
    - "Consume a language model from a client application"

Concepts covered:
- Creating a CLU project via the Authoring API
- Defining intents (TurnOn, TurnOff, SetTemperature, CheckStatus)
- Defining list entities (device) and prebuilt entities (temperature)
- Adding labelled training utterances
- Triggering a training job and polling until complete
- Evaluating the trained model
- Deploying the model to a named deployment slot
- Querying the deployed model from a client application

SDK note:
    CLU uses two separate clients:
        ConversationAuthoringClient  – manage projects, train, deploy
        ConversationsClient          – query the deployed model at runtime

Required env vars:
    AZURE_LANGUAGE_ENDPOINT  – e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_LANGUAGE_KEY       – 32-character resource key

Install:
    pip install azure-ai-language-conversations python-dotenv
"""

import os
import time
import json
from dotenv import load_dotenv

from azure.ai.language.conversations.authoring import ConversationAuthoringClient
from azure.ai.language.conversations import ConversationAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

ENDPOINT = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
KEY = os.environ.get("AZURE_LANGUAGE_KEY")

# Project / deployment names – change these to match your environment
PROJECT_NAME = "HomeAutomation"
DEPLOYMENT_NAME = "production"
TRAINING_JOB_NAME = "train-job-001"


def get_authoring_client() -> ConversationAuthoringClient:
    """Create an authenticated ConversationAuthoringClient."""
    if not ENDPOINT or not KEY:
        raise EnvironmentError(
            "AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY must be set in your .env file."
        )
    return ConversationAuthoringClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def get_analysis_client() -> ConversationAnalysisClient:
    """Create an authenticated ConversationAnalysisClient for inference."""
    if not ENDPOINT or not KEY:
        raise EnvironmentError(
            "AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY must be set in your .env file."
        )
    return ConversationAnalysisClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


# ---------------------------------------------------------------------------
# Step 1: Create (or re-use) a CLU project
# ---------------------------------------------------------------------------

def create_or_get_project(client: ConversationAuthoringClient) -> None:
    """
    Create a new CLU project.  If the project already exists, skip creation.

    Project settings:
        language       – primary language of training utterances (en-us)
        projectKind    – Conversation (for task-oriented dialogues)
        multilingual   – True to support utterances in multiple languages
    """
    print("\n" + "=" * 60)
    print(f"STEP 1: CREATE PROJECT  '{PROJECT_NAME}'")
    print("=" * 60)

    project_body = {
        "projectName": PROJECT_NAME,
        "language": "en-us",
        "projectKind": "Conversation",
        "description": "Home automation CLU demo project for AI-102 exam prep",
        "multilingual": False,
        "settings": {"confidenceThreshold": 0.5},
    }

    try:
        client.create_project(project_name=PROJECT_NAME, project=project_body)
        print(f"  ✓ Project '{PROJECT_NAME}' created.")
    except HttpResponseError as exc:
        if exc.status_code == 409:
            print(f"  Project '{PROJECT_NAME}' already exists – continuing.")
        else:
            raise


# ---------------------------------------------------------------------------
# Step 2: Import the project definition (intents, entities, utterances)
# ---------------------------------------------------------------------------

def import_project_assets(client: ConversationAuthoringClient) -> None:
    """
    Import a full project definition including intents, entities, and labelled
    utterances in one operation.

    In production you would load this JSON from a file; here we define it
    inline so the demo is self-contained.

    Project assets:
        intents    – the task goals the model should recognise
        entities   – slots that carry values from the user's utterance
        utterances – example sentences labelled with intent and entity spans
    """
    print("\n" + "=" * 60)
    print("STEP 2: IMPORT PROJECT ASSETS (intents, entities, utterances)")
    print("=" * 60)

    assets = {
        "projectFileVersion": "2022-05-01",
        "metadata": {
            "projectKind": "Conversation",
            "settings": {"confidenceThreshold": 0.5},
            "projectName": PROJECT_NAME,
            "multilingual": False,
            "description": "Home automation demo",
            "language": "en-us",
        },
        "assets": {
            "projectKind": "Conversation",
            "intents": [
                {"category": "TurnOn"},
                {"category": "TurnOff"},
                {"category": "SetTemperature"},
                {"category": "CheckStatus"},
                {"category": "None"},
            ],
            "entities": [
                {
                    "category": "device",
                    "compositionSetting": "combineComponents",
                    "list": {
                        "sublists": [
                            {
                                "listKey": "light",
                                "synonyms": [
                                    {"language": "en-us", "values": ["light", "lights", "lamp", "bulb"]},
                                ],
                            },
                            {
                                "listKey": "thermostat",
                                "synonyms": [
                                    {"language": "en-us", "values": ["thermostat", "heating", "AC", "air conditioning"]},
                                ],
                            },
                            {
                                "listKey": "tv",
                                "synonyms": [
                                    {"language": "en-us", "values": ["TV", "television", "screen"]},
                                ],
                            },
                        ]
                    },
                },
                {
                    "category": "temperature",
                    "compositionSetting": "combineComponents",
                    "prebuilts": [{"category": "Temperature"}],
                },
            ],
            "utterances": [
                # TurnOn
                {
                    "text": "turn on the lights",
                    "language": "en-us",
                    "intent": "TurnOn",
                    "entities": [{"category": "device", "offset": 12, "length": 6}],
                    "dataset": "Train",
                },
                {
                    "text": "switch on the TV",
                    "language": "en-us",
                    "intent": "TurnOn",
                    "entities": [{"category": "device", "offset": 14, "length": 2}],
                    "dataset": "Train",
                },
                {
                    "text": "turn the lamp on",
                    "language": "en-us",
                    "intent": "TurnOn",
                    "entities": [{"category": "device", "offset": 9, "length": 4}],
                    "dataset": "Train",
                },
                {
                    "text": "please turn on the air conditioning",
                    "language": "en-us",
                    "intent": "TurnOn",
                    "entities": [{"category": "device", "offset": 19, "length": 16}],
                    "dataset": "Train",
                },
                # TurnOff
                {
                    "text": "turn off the lights",
                    "language": "en-us",
                    "intent": "TurnOff",
                    "entities": [{"category": "device", "offset": 13, "length": 6}],
                    "dataset": "Train",
                },
                {
                    "text": "switch off the TV",
                    "language": "en-us",
                    "intent": "TurnOff",
                    "entities": [{"category": "device", "offset": 15, "length": 2}],
                    "dataset": "Train",
                },
                {
                    "text": "turn the heating off",
                    "language": "en-us",
                    "intent": "TurnOff",
                    "entities": [{"category": "device", "offset": 9, "length": 7}],
                    "dataset": "Train",
                },
                # SetTemperature
                {
                    "text": "set the temperature to 22 degrees",
                    "language": "en-us",
                    "intent": "SetTemperature",
                    "entities": [{"category": "temperature", "offset": 22, "length": 10}],
                    "dataset": "Train",
                },
                {
                    "text": "make it 20 degrees please",
                    "language": "en-us",
                    "intent": "SetTemperature",
                    "entities": [{"category": "temperature", "offset": 8, "length": 10}],
                    "dataset": "Train",
                },
                # CheckStatus
                {
                    "text": "is the TV on",
                    "language": "en-us",
                    "intent": "CheckStatus",
                    "entities": [{"category": "device", "offset": 7, "length": 2}],
                    "dataset": "Train",
                },
                {
                    "text": "are the lights on",
                    "language": "en-us",
                    "intent": "CheckStatus",
                    "entities": [{"category": "device", "offset": 8, "length": 6}],
                    "dataset": "Train",
                },
                # Test utterances
                {
                    "text": "please switch on the lights",
                    "language": "en-us",
                    "intent": "TurnOn",
                    "entities": [{"category": "device", "offset": 21, "length": 6}],
                    "dataset": "Test",
                },
                {
                    "text": "set thermostat to 18 degrees",
                    "language": "en-us",
                    "intent": "SetTemperature",
                    "entities": [{"category": "temperature", "offset": 18, "length": 10}],
                    "dataset": "Test",
                },
            ],
        },
    }

    try:
        poller = client.begin_import_project(
            project_name=PROJECT_NAME,
            project=assets,
        )
        poller.result()
        print("  ✓ Project assets imported successfully.")
    except HttpResponseError as exc:
        print(f"  [ERROR] Import failed: {exc.message}")
        raise


# ---------------------------------------------------------------------------
# Step 3: Train the model
# ---------------------------------------------------------------------------

def train_model(client: ConversationAuthoringClient) -> None:
    """
    Trigger a training job and poll until it completes.

    trainingMode:
        standard  – full training (slower, better accuracy)
        advanced  – extended training with transformer models (slower, best accuracy)

    evaluationOptions.kind:
        percentage    – automatic train/test split from labelled data
        manual        – use the Dataset field on each utterance (Train / Test)

    Exam tip: After training, call get_model_evaluation_summary() to see
    precision, recall, and F1 scores per intent and entity.
    """
    print("\n" + "=" * 60)
    print("STEP 3: TRAIN MODEL")
    print("=" * 60)

    training_body = {
        "modelLabel": TRAINING_JOB_NAME,
        "trainingMode": "standard",
        "trainingConfigVersion": "latest",
        "evaluationOptions": {
            "kind": "manual",   # use the Dataset labels we set on utterances
        },
    }

    try:
        print(f"  Submitting training job '{TRAINING_JOB_NAME}'…")
        poller = client.begin_train(
            project_name=PROJECT_NAME,
            configuration=training_body,
        )

        # Poll manually so we can log progress
        while not poller.done():
            status = poller.status()
            print(f"    Training status: {status}")
            time.sleep(15)

        result = poller.result()
        print(f"  ✓ Training complete.  Model label: {result.get('modelLabel', TRAINING_JOB_NAME)}")

    except HttpResponseError as exc:
        print(f"  [ERROR] Training failed: {exc.message}")
        raise


# ---------------------------------------------------------------------------
# Step 4: Evaluate the trained model
# ---------------------------------------------------------------------------

def evaluate_model(client: ConversationAuthoringClient) -> None:
    """
    Retrieve and display model evaluation metrics (precision, recall, F1).

    The evaluation is computed against the Test-split utterances.
    Results are available per-intent and per-entity.
    """
    print("\n" + "=" * 60)
    print("STEP 4: EVALUATE MODEL")
    print("=" * 60)

    try:
        summary = client.get_model_evaluation_summary(
            project_name=PROJECT_NAME,
            trained_model_label=TRAINING_JOB_NAME,
        )
    except HttpResponseError as exc:
        print(f"  [ERROR] Could not retrieve evaluation summary: {exc.message}")
        return

    overall = summary.get("entitiesEvaluation", {}).get("microF1", "n/a")
    print(f"  Overall entity micro-F1: {overall}")

    intent_eval = summary.get("intentsEvaluation", {})
    print("\n  Intent evaluation:")
    print(f"  {'Intent':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 50)

    per_intent = intent_eval.get("perIntentEvaluation", {})
    for intent_name, metrics in per_intent.items():
        p = metrics.get("precision", 0)
        r = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        print(f"  {intent_name:<20} {p:>10.2f} {r:>8.2f} {f1:>8.2f}")


# ---------------------------------------------------------------------------
# Step 5: Deploy the trained model
# ---------------------------------------------------------------------------

def deploy_model(client: ConversationAuthoringClient) -> None:
    """
    Deploy the trained model to a named deployment slot.

    A deployment slot (e.g. 'production', 'staging') is what client apps
    target.  Multiple model versions can coexist in different deployment slots.
    """
    print("\n" + "=" * 60)
    print(f"STEP 5: DEPLOY MODEL  →  '{DEPLOYMENT_NAME}'")
    print("=" * 60)

    deploy_body = {"trainedModelLabel": TRAINING_JOB_NAME}

    try:
        poller = client.begin_deploy_project(
            project_name=PROJECT_NAME,
            deployment_name=DEPLOYMENT_NAME,
            deployment=deploy_body,
        )
        poller.result()
        print(f"  ✓ Model deployed to '{DEPLOYMENT_NAME}'.")
    except HttpResponseError as exc:
        print(f"  [ERROR] Deployment failed: {exc.message}")
        raise


# ---------------------------------------------------------------------------
# Step 6: Query the deployed model
# ---------------------------------------------------------------------------

def query_model(analysis_client: ConversationAnalysisClient, utterances: list[str]) -> None:
    """
    Send utterances to the deployed CLU model and print the results.

    The response includes:
        topIntent         – highest-confidence intent
        confidenceScores  – per-intent scores
        entities          – extracted entity values with category and text
    """
    print("\n" + "=" * 60)
    print("STEP 6: QUERY DEPLOYED MODEL")
    print(f"  Project: {PROJECT_NAME}  |  Deployment: {DEPLOYMENT_NAME}")
    print("=" * 60)

    for utterance in utterances:
        print(f"\n  Query: \"{utterance}\"")

        request_body = {
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "id": "1",
                    "participantId": "user",
                    "text": utterance,
                }
            },
            "parameters": {
                "projectName": PROJECT_NAME,
                "deploymentName": DEPLOYMENT_NAME,
                "verbose": True,
            },
        }

        try:
            response = analysis_client.analyze_conversation(task=request_body)
        except HttpResponseError as exc:
            print(f"  [ERROR] {exc.message}")
            continue

        prediction = response.get("result", {}).get("prediction", {})
        top_intent = prediction.get("topIntent", "unknown")
        scores = prediction.get("intents", {})
        entities = prediction.get("entities", [])

        top_score = scores.get(top_intent, {}).get("confidenceScore", 0) if isinstance(scores, dict) else 0
        print(f"  Top intent: {top_intent}  (confidence: {top_score:.2f})")

        if entities:
            print("  Entities:")
            for entity in entities:
                print(
                    f"    {entity.get('category')}: '{entity.get('text')}'  "
                    f"(score: {entity.get('confidenceScore', 0):.2f})"
                )
        else:
            print("  Entities: (none)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point – run the full CLU lifecycle."""
    print("Azure AI Language – Conversational Language Understanding (CLU) Demo")
    print("Endpoint:", ENDPOINT or "(not set)")

    authoring_client = get_authoring_client()
    analysis_client = get_analysis_client()

    # Full lifecycle
    create_or_get_project(authoring_client)
    import_project_assets(authoring_client)
    train_model(authoring_client)
    evaluate_model(authoring_client)
    deploy_model(authoring_client)

    # Test queries
    test_utterances = [
        "please turn on the lights in the living room",
        "switch off the TV",
        "set the temperature to 21 degrees",
        "are the lights currently on?",
        "what time is it?",  # should map to None intent
    ]
    query_model(analysis_client, test_utterances)

    print("\nDone.")


if __name__ == "__main__":
    main()
