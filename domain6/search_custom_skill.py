"""
search_custom_skill.py
======================
Demonstrates implementing a Custom Skill for Azure AI Search:

Part A — Azure Function endpoint (the skill itself):
  - Shows the exact JSON request/response contract Azure Search expects
  - Implements a sample custom skill that classifies document sentiment
    and assigns a priority score (illustrative business logic)
  - Includes a local Flask development server for testing

Part B — Skillset integration:
  - Defines a WebApiSkill that calls the deployed Azure Function
  - Shows how to wire inputs/outputs and handle custom HTTP headers
  - Updates an existing skillset to include the custom skill

AI-102 Exam Skills Mapped:
  - Implement custom skills and include them in a skillset

Key concepts:
  - Custom skills extend AI Search enrichment with ANY logic (translation,
    classification, custom NLP, database lookups, etc.)
  - Azure Functions is the most common host, but any HTTPS endpoint works
  - The skill contract: receives { values: [{recordId, data}] }
                        returns  { values: [{recordId, data, errors, warnings}] }

Required environment variables (see .env.sample):
  AZURE_SEARCH_ENDPOINT         - https://<service>.search.windows.net
  AZURE_SEARCH_ADMIN_KEY        - Admin API key
  AZURE_SEARCH_INDEX_NAME       - Existing index name
  AZURE_CUSTOM_SKILL_URL        - HTTPS URL of deployed Azure Function skill
  AZURE_FUNCTION_KEY            - Azure Function host/function key (optional)

Package: azure-search-documents>=11.6.0, flask>=3.0.0 (for local dev server)
"""

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# PART A: The Custom Skill endpoint (Azure Function / Flask server)
# ---------------------------------------------------------------------------
# This section shows the full request/response contract.
# Deploy this as an Azure Function (HTTP trigger) or any HTTPS endpoint.
# For local testing, run: python search_custom_skill.py --serve
# ---------------------------------------------------------------------------

# --- Example request that Azure Search sends to your skill endpoint ---
EXAMPLE_SKILL_REQUEST = {
    "values": [
        {
            "recordId": "doc-001",
            "data": {
                "text": "The product arrived damaged and customer service was unhelpful.",
                "language": "en",
            },
        },
        {
            "recordId": "doc-002",
            "data": {
                "text": "Excellent quality and fast delivery. Highly recommend!",
                "language": "en",
            },
        },
        {
            "recordId": "doc-003",
            "data": {
                "text": "",  # Edge case: empty text
                "language": "en",
            },
        },
    ]
}


def classify_and_score(text: str, language: str) -> dict[str, Any]:
    """
    Business logic for the custom skill.
    In a real scenario this could call:
      - Azure AI Language for sentiment
      - A custom ML model endpoint
      - An internal database/API
      - Any external service

    Returns a dict with the enriched fields your index expects.
    """
    if not text or not text.strip():
        return {
            "sentiment": "unknown",
            "priority_score": 0,
            "word_count": 0,
            "contains_complaint": False,
        }

    text_lower = text.lower()

    # Simple heuristic sentiment (replace with real model in production)
    negative_words = {"damaged", "unhelpful", "broken", "terrible", "awful", "bad", "poor"}
    positive_words = {"excellent", "great", "good", "recommend", "fantastic", "outstanding"}

    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)

    if neg_count > pos_count:
        sentiment = "negative"
        priority_score = 90 + (neg_count * 2)   # High priority for negative docs
    elif pos_count > neg_count:
        sentiment = "positive"
        priority_score = 10
    else:
        sentiment = "neutral"
        priority_score = 50

    return {
        "sentiment": sentiment,
        "priority_score": min(priority_score, 100),
        "word_count": len(text.split()),
        "contains_complaint": neg_count > 0,
    }


def process_skill_request(request_body: dict) -> dict:
    """
    Core custom skill handler — processes an AI Search skill request payload.

    Contract (MUST be followed exactly):
      INPUT:  { "values": [ { "recordId": str, "data": { <input fields> } } ] }
      OUTPUT: { "values": [ { "recordId": str, "data": { <output fields> },
                               "errors": [...], "warnings": [...] } ] }

    Rules:
      - Output array must include every recordId from the input
      - errors/warnings are optional arrays of { "message": str }
      - If a record fails, return errors; the indexer continues with other docs
    """
    response_values = []

    for record in request_body.get("values", []):
        record_id = record["recordId"]
        data = record.get("data", {})

        errors = []
        warnings = []
        output_data = {}

        try:
            text = data.get("text", "")
            language = data.get("language", "en")

            if not isinstance(text, str):
                errors.append({"message": f"'text' must be a string, got {type(text).__name__}"})
            else:
                if len(text) > 50000:
                    warnings.append({"message": "Text truncated to 50,000 characters"})
                    text = text[:50000]

                output_data = classify_and_score(text, language)

        except Exception as exc:  # pylint: disable=broad-except
            errors.append({"message": f"Skill processing error: {str(exc)}"})

        response_values.append(
            {
                "recordId": record_id,
                "data": output_data,
                "errors": errors,
                "warnings": warnings,
            }
        )

    return {"values": response_values}


# ---------------------------------------------------------------------------
# Local dev server using Flask (for testing before deploying to Azure)
# ---------------------------------------------------------------------------

def run_local_server():
    """
    Run a local Flask server that mimics the Azure Function endpoint.
    Use this to test the skill contract before deploying.

    Test with:
      curl -X POST http://localhost:7071/api/custom-skill \
           -H "Content-Type: application/json" \
           -d @sample_request.json
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask is not installed. Run: pip install flask")
        print("Alternatively, deploy directly to Azure Functions.")
        return

    app = Flask(__name__)

    @app.route("/api/custom-skill", methods=["POST"])
    def custom_skill_endpoint():
        """Azure Function-compatible HTTP endpoint for the custom skill."""
        try:
            body = request.get_json(force=True)
            if body is None:
                return jsonify({"error": "Request body must be JSON"}), 400

            response = process_skill_request(body)
            return jsonify(response), 200

        except Exception as exc:  # pylint: disable=broad-except
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "skill": "custom-sentiment-classifier"})

    print("Starting local skill server on http://localhost:7071")
    print("Skill endpoint: POST http://localhost:7071/api/custom-skill")
    app.run(host="0.0.0.0", port=7071, debug=True)


# ---------------------------------------------------------------------------
# Azure Function entry point (main.py for Azure Functions v2)
# ---------------------------------------------------------------------------
# To deploy as an Azure Function, create a function_app.py with:
#
# import azure.functions as func
# from search_custom_skill import process_skill_request
#
# app = func.FunctionApp()
#
# @app.route(route="custom-skill", auth_level=func.AuthLevel.FUNCTION)
# def custom_skill(req: func.HttpRequest) -> func.HttpResponse:
#     body = req.get_json()
#     result = process_skill_request(body)
#     return func.HttpResponse(
#         body=json.dumps(result),
#         mimetype="application/json",
#         status_code=200
#     )
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PART B: Register the custom skill in an Azure AI Search skillset
# ---------------------------------------------------------------------------

def add_custom_skill_to_skillset():
    """
    Adds a WebApiSkill pointing at the deployed custom skill endpoint
    to an existing skillset in Azure AI Search.

    The WebApiSkill is how Azure Search calls ANY external HTTPS endpoint,
    including Azure Functions, App Service, or Container Apps.
    """
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    from azure.search.documents.indexes import SearchIndexerClient
    from azure.search.documents.indexes.models import (
        WebApiSkill,
        InputFieldMappingEntry,
        OutputFieldMappingEntry,
        SearchIndexerSkillset,
        CognitiveServicesAccountKey,
    )

    SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
    SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
    INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "ai102-demo-index")
    CUSTOM_SKILL_URL = os.environ["AZURE_CUSTOM_SKILL_URL"]
    FUNCTION_KEY = os.getenv("AZURE_FUNCTION_KEY", "")
    AI_SERVICES_KEY = os.environ["AZURE_AI_SERVICES_KEY"]

    SKILLSET_NAME = f"{INDEX_NAME}-skillset"

    credential = AzureKeyCredential(SEARCH_ADMIN_KEY)
    indexer_client = SearchIndexerClient(endpoint=SEARCH_ENDPOINT, credential=credential)

    try:
        # Retrieve the existing skillset
        print(f"Retrieving skillset '{SKILLSET_NAME}'...")
        skillset = indexer_client.get_skillset(SKILLSET_NAME)

        # Build the WebApiSkill
        # degree_of_parallelism: how many documents to process concurrently (1–5)
        custom_skill = WebApiSkill(
            name="custom-sentiment-skill",
            description="Custom skill: classifies sentiment and assigns priority score",
            context="/document",
            uri=CUSTOM_SKILL_URL,
            http_method="POST",
            timeout="PT30S",          # ISO 8601 duration: 30 second timeout
            batch_size=5,             # Docs per batch sent to the endpoint
            degree_of_parallelism=2,  # Concurrent requests
            http_headers={
                "x-functions-key": FUNCTION_KEY,
                "Content-Type": "application/json",
            } if FUNCTION_KEY else {},
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/merged_text"),
                InputFieldMappingEntry(name="language", source="/document/language"),
            ],
            outputs=[
                OutputFieldMappingEntry(name="sentiment", target_name="custom_sentiment"),
                OutputFieldMappingEntry(name="priority_score", target_name="priority_score"),
                OutputFieldMappingEntry(name="word_count", target_name="word_count"),
                OutputFieldMappingEntry(name="contains_complaint", target_name="contains_complaint"),
            ],
        )

        # Append the custom skill to the existing skills list
        existing_skill_names = [s.name for s in skillset.skills]
        if custom_skill.name in existing_skill_names:
            print(f"Skill '{custom_skill.name}' already exists. Replacing...")
            skillset.skills = [
                s for s in skillset.skills if s.name != custom_skill.name
            ]

        skillset.skills.append(custom_skill)

        # Push the updated skillset back
        updated = indexer_client.create_or_update_skillset(skillset)
        print(f"Skillset '{updated.name}' updated. Total skills: {len(updated.skills)}")
        print("Skills in pipeline:")
        for skill in updated.skills:
            print(f"  - [{skill.__class__.__name__}] {skill.name}")

        print("\nIMPORTANT: Add the custom skill output fields to your index schema:")
        print("  custom_sentiment   (Edm.String, filterable, facetable)")
        print("  priority_score     (Edm.Int32, filterable, sortable)")
        print("  word_count         (Edm.Int32, filterable, sortable)")
        print("  contains_complaint (Edm.Boolean, filterable, facetable)")
        print("\nThen reset and re-run the indexer to re-process all documents.")

    except HttpResponseError as e:
        print(f"Azure Search API error: {e.message}")
        raise
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        raise


# ---------------------------------------------------------------------------
# Demo: Test the skill logic locally without any Azure resources
# ---------------------------------------------------------------------------

def demo_skill_locally():
    """Run the skill against the example request and print the response."""
    print("=== Local Skill Demo ===")
    print("Input:")
    print(json.dumps(EXAMPLE_SKILL_REQUEST, indent=2))

    response = process_skill_request(EXAMPLE_SKILL_REQUEST)

    print("\nOutput:")
    print(json.dumps(response, indent=2))

    # Validate the contract
    assert len(response["values"]) == len(EXAMPLE_SKILL_REQUEST["values"]), \
        "Response must have same number of records as request"

    input_ids = {v["recordId"] for v in EXAMPLE_SKILL_REQUEST["values"]}
    output_ids = {v["recordId"] for v in response["values"]}
    assert input_ids == output_ids, "All recordIds must be present in response"

    print("\nContract validation: PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        # Start local Flask development server
        run_local_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "--register":
        # Register skill in Azure AI Search skillset
        add_custom_skill_to_skillset()
    else:
        # Default: run local demo
        demo_skill_locally()
        print("\nUsage:")
        print("  python search_custom_skill.py           # Run local demo")
        print("  python search_custom_skill.py --serve   # Start local Flask server")
        print("  python search_custom_skill.py --register # Register in Azure skillset")


if __name__ == "__main__":
    main()
