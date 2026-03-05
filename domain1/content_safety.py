"""
content_safety.py
=================
Demonstrates using Azure AI Content Safety to analyze text and images
for harmful content across four harm categories.

Exam Skills:
  - "Implement content moderation solutions" (Domain 1)
  - "Configure responsible AI insights, including content safety" (Domain 1)

What this demo shows:
  - Analyzing text for hate speech, violence, self-harm, and sexual content
  - Analyzing images (URL and base64) for harmful visual content
  - Interpreting severity scores (0-7 scale: 0=safe, 2=low, 4=medium, 6=high)
  - Taking action based on severity thresholds
  - Understanding the four harm categories and their sub-categories

Severity score scale (Azure AI Content Safety):
  0    = Safe / Not detected
  2    = Low severity (mild, indirect references)
  4    = Medium severity (moderate content)
  6    = High severity (severe, explicit content)

Required packages:
  pip install azure-ai-contentsafety azure-identity python-dotenv

Required environment variables (in .env):
  AZURE_AI_SERVICES_ENDPOINT  - e.g. https://<name>.cognitiveservices.azure.com/
  AZURE_AI_SERVICES_KEY       - API key for the resource
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (
    AnalyzeTextOptions,
    AnalyzeImageOptions,
    ImageData,
    TextCategory,
    ImageCategory,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

ENDPOINT = os.environ["AZURE_AI_SERVICES_ENDPOINT"]
KEY      = os.environ["AZURE_AI_SERVICES_KEY"]

# ---------------------------------------------------------------------------
# Severity thresholds - tune these per your application's risk tolerance
# ---------------------------------------------------------------------------
# Action mapping:
#   "allow"   = pass content through
#   "review"  = flag for human review
#   "block"   = reject content

SEVERITY_ACTION_MAP = {
    0: "allow",   # No harm detected
    2: "review",  # Low severity - may need review depending on context
    4: "block",   # Medium severity - typically block in consumer apps
    6: "block",   # High severity - always block
}

# For stricter apps (e.g. children's platforms), block at severity 2:
STRICT_MODE_THRESHOLD = 2


def get_client() -> ContentSafetyClient:
    """Create an authenticated Content Safety client."""
    return ContentSafetyClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def severity_to_action(severity: int, strict_mode: bool = False) -> str:
    """
    Map a severity score to an action string.
    In strict mode, block anything at severity >= 2.
    """
    if strict_mode and severity >= STRICT_MODE_THRESHOLD:
        return "block"
    return SEVERITY_ACTION_MAP.get(severity, "block")  # default block for unknown scores


def analyze_text(client: ContentSafetyClient, text: str, strict_mode: bool = False) -> dict:
    """
    Analyze text for all four harm categories:
      - Hate    : Hate speech, discriminatory language
      - Violence: Threats, graphic violence descriptions
      - SelfHarm: Content promoting self-harm or suicide
      - Sexual  : Sexually explicit content

    Returns a dict with per-category severity scores and recommended actions.
    """
    print(f"\n[TEXT ANALYSIS] Analyzing: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

    request = AnalyzeTextOptions(
        text=text,
        categories=[
            TextCategory.HATE,
            TextCategory.VIOLENCE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
        ],
        # output_type="FourSeverityLevels"  # default - returns 0,2,4,6
        # output_type="EightSeverityLevels" # extended - returns 0-7 fine-grained
    )

    try:
        response = client.analyze_text(request)
    except HttpResponseError as e:
        print(f"  [ERROR] Content Safety API error: {e.message}")
        return {}

    results = {}
    overall_action = "allow"

    print(f"  {'Category':<12} | {'Severity':<10} | {'Action':<8}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*8}")

    for category_result in response.categories_analysis:
        category = category_result.category
        severity = category_result.severity
        action   = severity_to_action(severity, strict_mode)

        print(f"  {category:<12} | {severity:<10} | {action:<8}")
        results[category] = {"severity": severity, "action": action}

        # Escalate overall action (allow < review < block)
        if action == "block" or (action == "review" and overall_action == "allow"):
            overall_action = action

    print(f"\n  Overall action: [{overall_action.upper()}]")
    results["_overall"] = overall_action
    return results


def analyze_image_from_url(client: ContentSafetyClient, image_url: str) -> dict:
    """
    Analyze an image by URL for harmful visual content.
    The URL must be publicly accessible.

    Image harm categories:
      - Hate    : Hate symbols, imagery
      - Violence: Graphic violence, weapons
      - SelfHarm: Self-harm imagery
      - Sexual  : Explicit imagery
    """
    print(f"\n[IMAGE ANALYSIS - URL] {image_url[:80]}")

    request = AnalyzeImageOptions(
        image=ImageData(url=image_url),
        categories=[
            ImageCategory.HATE,
            ImageCategory.VIOLENCE,
            ImageCategory.SELF_HARM,
            ImageCategory.SEXUAL,
        ],
    )

    try:
        response = client.analyze_image(request)
    except HttpResponseError as e:
        print(f"  [ERROR] Image analysis failed: {e.message}")
        return {}

    results = {}
    print(f"  {'Category':<12} | {'Severity':<10} | {'Action':<8}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*8}")

    for category_result in response.categories_analysis:
        category = category_result.category
        severity = category_result.severity
        action   = severity_to_action(severity)
        print(f"  {category:<12} | {severity:<10} | {action:<8}")
        results[category] = {"severity": severity, "action": action}

    return results


def analyze_image_from_file(client: ContentSafetyClient, image_path: str) -> dict:
    """
    Analyze a local image file by encoding it as base64.
    Useful when images are not publicly accessible via URL.

    The image must be < 4 MB and in JPEG, PNG, GIF, BMP, or TIFF format.
    """
    print(f"\n[IMAGE ANALYSIS - File] {image_path}")

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        print(f"  Image size: {len(image_bytes) / 1024:.1f} KB")
    except FileNotFoundError:
        print(f"  [ERROR] File not found: {image_path}")
        print("  Skipping file-based image analysis.")
        return {}

    request = AnalyzeImageOptions(
        image=ImageData(content=image_b64),  # Pass base64-encoded content
        categories=[
            ImageCategory.HATE,
            ImageCategory.VIOLENCE,
            ImageCategory.SELF_HARM,
            ImageCategory.SEXUAL,
        ],
    )

    try:
        response = client.analyze_image(request)
    except HttpResponseError as e:
        print(f"  [ERROR] Image analysis failed: {e.message}")
        return {}

    results = {}
    for category_result in response.categories_analysis:
        category = category_result.category
        severity = category_result.severity
        action   = severity_to_action(severity)
        print(f"  {category:<12} | Severity: {severity} | Action: {action}")
        results[category] = {"severity": severity, "action": action}

    return results


def demo_moderation_pipeline(client: ContentSafetyClient) -> None:
    """
    Demonstrates a realistic content moderation pipeline for user-generated content.
    Shows how to use Content Safety in a real app workflow.
    """
    print("\n" + "=" * 60)
    print("Content Moderation Pipeline Demo")
    print("=" * 60)

    test_cases = [
        {
            "text": "I love spending time with my family in the park.",
            "expected": "allow",
            "description": "Safe content",
        },
        {
            "text": "This product is amazing, highly recommend it to everyone!",
            "expected": "allow",
            "description": "Positive product review",
        },
        {
            "text": "I want to hurt someone who makes me feel this way.",
            "expected": "block",
            "description": "Violent threat",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n  Test case {i}: {case['description']}")
        result = analyze_text(client, case["text"])
        action = result.get("_overall", "unknown")
        match = "✓" if action == case["expected"] else "✗"
        print(f"  Expected: {case['expected']} | Got: {action} {match}")


def main():
    print("=" * 60)
    print("Azure AI Content Safety Demo")
    print("=" * 60)
    print(f"Endpoint: {ENDPOINT}")

    try:
        client = get_client()

        # ---------------------------------------------------------------
        # Text analysis examples
        # ---------------------------------------------------------------
        print("\n--- TEXT ANALYSIS EXAMPLES ---")

        # Safe text
        analyze_text(client, "The weather today is lovely, perfect for a walk in the park.")

        # Text with mild risk (review)
        analyze_text(
            client,
            "There was a fierce battle in the movie with lots of action sequences.",
        )

        # Hate speech example (intentionally mild for demo)
        analyze_text(
            client,
            "I really dislike people who disagree with me.",
        )

        # ---------------------------------------------------------------
        # Image analysis via public URL
        # ---------------------------------------------------------------
        print("\n--- IMAGE ANALYSIS EXAMPLES ---")

        # Analyze a safe public image
        analyze_image_from_url(
            client,
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        )

        # ---------------------------------------------------------------
        # Pipeline demo
        # ---------------------------------------------------------------
        demo_moderation_pipeline(client)

        # ---------------------------------------------------------------
        # Strict mode example
        # ---------------------------------------------------------------
        print("\n--- STRICT MODE (block at severity >= 2) ---")
        analyze_text(
            client,
            "The movie had some violent scenes that were difficult to watch.",
            strict_mode=True,
        )

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except HttpResponseError as e:
        print(f"\n[ERROR] Content Safety API error: {e.message}")


if __name__ == "__main__":
    main()
