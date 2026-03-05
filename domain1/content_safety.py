"""
content_safety.py
=================
Demonstrates analyzing text and images for harmful content using
Azure AI Content Safety.

Exam Skill: "Implement content moderation" and "Configure responsible AI"
            (Domain 1 - Plan and manage an Azure AI solution)

What this demo shows:
  - Text analysis across all four harm categories
  - Image analysis across all four harm categories
  - Understanding severity levels (0=safe, 2=low, 4=medium, 6=high)
  - Making accept/reject decisions based on severity
  - Handling errors and understanding response structure

Harm Categories:
  - Hate:      Content that attacks individuals/groups based on protected attributes
  - Violence:  Content depicting, glorifying, or inciting physical harm
  - SelfHarm:  Content encouraging or depicting self-harm/suicide
  - Sexual:    Content of a sexual nature

Severity Scale:
  0 = Safe (no harmful content)
  2 = Low severity
  4 = Medium severity
  6 = High severity (most harmful)

Required packages:
  pip install azure-ai-contentsafety python-dotenv

Required environment variables (in .env):
  AZURE_AI_SERVICES_ENDPOINT - your Azure AI Services endpoint
  AZURE_AI_SERVICES_KEY      - your Azure AI Services API key
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (
    TextCategory,
    ImageCategory,
    AnalyzeTextOptions,
    AnalyzeImageOptions,
    ImageData,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
KEY      = os.environ["AZURE_AI_SERVICES_KEY"]

# Severity thresholds - adjust for your use case
# Exam note: You configure these in Azure Portal / SDK, not hardcoded
THRESHOLD_ACCEPT = 2    # Block if severity >= this value (0=only safe, 2=block low+)
THRESHOLD_REVIEW = 0    # Flag for human review if >= this


def create_client() -> ContentSafetyClient:
    """Initialize the Content Safety client."""
    return ContentSafetyClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


# ---------------------------------------------------------------------------
# Text Analysis
# ---------------------------------------------------------------------------

def analyze_text_safety(client: ContentSafetyClient, text: str) -> dict:
    """
    Analyze text for harmful content across all four categories.

    Returns a dict with category severities and an overall decision.
    """
    request = AnalyzeTextOptions(
        text=text,
        categories=[  # Explicitly request all categories (default if omitted)
            TextCategory.HATE,
            TextCategory.VIOLENCE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
        ],
        output_type="FourSeverityLevels",  # 0, 2, 4, 6
    )

    response = client.analyze_text(request)

    results = {}
    max_severity = 0
    for category_result in response.categories_analysis:
        cat = category_result.category
        sev = category_result.severity
        results[cat] = sev
        max_severity = max(max_severity, sev)

    results["max_severity"] = max_severity
    results["decision"] = "BLOCK" if max_severity >= THRESHOLD_ACCEPT else "ALLOW"

    return results


def demo_text_analysis(client: ContentSafetyClient) -> None:
    """Run text safety analysis on a variety of test strings."""
    print("\n" + "=" * 60)
    print("TEXT CONTENT SAFETY ANALYSIS")
    print("=" * 60)

    test_cases = [
        {
            "text": "The weather today is sunny and warm.",
            "expected": "safe - benign content",
        },
        {
            "text": "I love cooking Italian food on weekends.",
            "expected": "safe - benign content",
        },
        {
            "text": "This historical documentary discusses the violence of World War II.",
            "expected": "possibly low violence - educational context",
        },
        {
            "text": "If you are struggling, please reach out to a mental health professional.",
            "expected": "safe - supportive message about mental health",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] '{case['text'][:60]}...' " if len(case['text']) > 60
              else f"\n[Test {i}] '{case['text']}'")
        print(f"  Expected: {case['expected']}")

        try:
            result = analyze_text_safety(client, case["text"])
            print(f"  Results:")
            for category in ["Hate", "Violence", "SelfHarm", "Sexual"]:
                severity = result.get(category, 0)
                bar = "█" * (severity // 2) + "░" * (3 - severity // 2)
                print(f"    {category:<12}: {severity} [{bar}]")
            print(f"  Decision: {result['decision']} (max severity: {result['max_severity']})")

        except HttpResponseError as e:
            print(f"  [ERROR] {e.message}")


# ---------------------------------------------------------------------------
# Image Analysis
# ---------------------------------------------------------------------------

def analyze_image_from_url(client: ContentSafetyClient, image_url: str) -> dict:
    """
    Analyze an image from a URL for harmful content.
    """
    request = AnalyzeImageOptions(
        image=ImageData(url=image_url),
        categories=[
            ImageCategory.HATE,
            ImageCategory.VIOLENCE,
            ImageCategory.SELF_HARM,
            ImageCategory.SEXUAL,
        ],
    )

    response = client.analyze_image(request)

    results = {}
    max_severity = 0
    for category_result in response.categories_analysis:
        cat = category_result.category
        sev = category_result.severity
        results[cat] = sev
        max_severity = max(max_severity, sev)

    results["max_severity"] = max_severity
    results["decision"] = "BLOCK" if max_severity >= THRESHOLD_ACCEPT else "ALLOW"

    return results


def analyze_image_from_bytes(client: ContentSafetyClient, image_path: str) -> dict:
    """
    Analyze an image from local file (base64 encoded).
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    request = AnalyzeImageOptions(
        image=ImageData(content=image_bytes),  # Pass raw bytes
        categories=[
            ImageCategory.HATE,
            ImageCategory.VIOLENCE,
            ImageCategory.SELF_HARM,
            ImageCategory.SEXUAL,
        ],
    )

    response = client.analyze_image(request)

    results = {}
    max_severity = 0
    for category_result in response.categories_analysis:
        results[category_result.category] = category_result.severity
        max_severity = max(max_severity, category_result.severity)

    results["max_severity"] = max_severity
    results["decision"] = "BLOCK" if max_severity >= THRESHOLD_ACCEPT else "ALLOW"
    return results


def demo_image_analysis(client: ContentSafetyClient) -> None:
    """
    Demonstrate image safety analysis.
    Uses a publicly accessible safe image for the demo.
    """
    print("\n" + "=" * 60)
    print("IMAGE CONTENT SAFETY ANALYSIS")
    print("=" * 60)

    # Use a safe public domain image
    safe_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

    print(f"\n[Image Test] Analyzing: {safe_image_url[:60]}...")
    print("  Expected: safe - transparency demo image from Wikipedia")

    try:
        result = analyze_image_from_url(client, safe_image_url)
        print(f"  Results:")
        for category in ["Hate", "Violence", "SelfHarm", "Sexual"]:
            severity = result.get(category, 0)
            bar = "█" * (severity // 2) + "░" * (3 - severity // 2)
            print(f"    {category:<12}: {severity} [{bar}]")
        print(f"  Decision: {result['decision']} (max severity: {result['max_severity']})")

    except HttpResponseError as e:
        print(f"  [ERROR] {e.message}")
        if "InvalidImageFormat" in str(e):
            print("  Note: Try a different image URL (JPEG or PNG required)")


# ---------------------------------------------------------------------------
# Batch processing example
# ---------------------------------------------------------------------------

def demo_batch_moderation(client: ContentSafetyClient) -> None:
    """
    Demonstrate batch-style moderation for a content pipeline.
    Shows how to make accept/reject decisions at scale.
    """
    print("\n" + "=" * 60)
    print("BATCH CONTENT MODERATION PIPELINE")
    print("=" * 60)

    # Simulate a pipeline of user-generated content
    user_submissions = [
        {"id": "msg_001", "text": "Just finished reading a great book about history!"},
        {"id": "msg_002", "text": "Has anyone tried the new restaurant downtown?"},
        {"id": "msg_003", "text": "Learning Python programming is so rewarding."},
        {"id": "msg_004", "text": "The sunset over the mountains was beautiful today."},
    ]

    allowed = []
    blocked = []
    errors  = []

    for submission in user_submissions:
        try:
            result = analyze_text_safety(client, submission["text"])
            if result["decision"] == "ALLOW":
                allowed.append(submission["id"])
                status = "✅ ALLOWED"
            else:
                blocked.append(submission["id"])
                status = "❌ BLOCKED"
            print(f"  {submission['id']}: {status} (max severity: {result['max_severity']})")
        except HttpResponseError as e:
            errors.append(submission["id"])
            print(f"  {submission['id']}: ⚠️  ERROR - {e.message[:50]}")

    print(f"\n  Summary: {len(allowed)} allowed, {len(blocked)} blocked, {len(errors)} errors")
    print(f"  Allowed IDs: {allowed}")
    if blocked:
        print(f"  Blocked IDs: {blocked}")


if __name__ == "__main__":
    print("=" * 60)
    print("Azure AI Content Safety Demo")
    print("=" * 60)
    print(f"Endpoint: {ENDPOINT}")
    print(f"Block threshold: severity >= {THRESHOLD_ACCEPT}")

    client = create_client()

    demo_text_analysis(client)
    demo_image_analysis(client)
    demo_batch_moderation(client)

    print("\n" + "=" * 60)
    print("Content Safety Demo Complete")
    print("=" * 60)
