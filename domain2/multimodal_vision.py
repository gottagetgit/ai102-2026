"""
multimodal_vision.py
====================
Demonstrates sending images to GPT-4o (vision) for analysis.
Covers image description, data extraction from images, and chart analysis.

Exam Skill: "Use large multimodal models in Azure OpenAI"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Sending images via URL to GPT-4o for analysis
  - Sending local images encoded as base64
  - Combining text and image in the same message
  - Multi-image analysis in a single request
  - Extracting structured data from images
  - Controlling detail level (low vs. high)
  - Practical use cases: document extraction, chart analysis, accessibility

Vision API key concepts:
  - Images are included in the "content" array of a message
  - Each image has a "type": "image_url" or "type": "image_base64"
  - "detail": "low" (85 tokens, fast) or "high" (more tokens, more detail)
  - GPT-4o supports up to 20 images per request
  - Max image size: 20 MB, max dimensions: 2048x2048 (resized internally)

Required packages:
  pip install openai requests pillow python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT    - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY         - API key
  AZURE_OPENAI_DEPLOYMENT  - GPT-4o deployment name (must support vision)
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()

ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def get_client() -> openai.AzureOpenAI:
    """Return authenticated AzureOpenAI client."""
    return openai.AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version="2024-12-01-preview",
    )


def image_url_content(url: str, detail: str = "auto") -> dict:
    """
    Create a message content item for an image from a URL.

    detail options:
      "low"   : 85 tokens, 512x512 effective resolution, fast
      "high"  : More tokens, higher resolution, slower (default for high-res images)
      "auto"  : Model chooses based on image size
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": url,
            "detail": detail,
        }
    }


def image_base64_content(image_path: str, detail: str = "auto") -> dict:
    """
    Create a message content item for a local image file (base64 encoded).
    Use this for images that are not publicly accessible via URL.

    The data URI format: data:<mime_type>;base64,<data>
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Determine MIME type from extension
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(path.suffix.lower(), "image/jpeg")

    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{b64_data}",
            "detail": detail,
        }
    }


def analyze_image_url(client: openai.AzureOpenAI, image_url: str, question: str) -> str:
    """
    Analyze an image from a URL with a specific question.
    The model responds based on the visual content of the image.
    """
    print(f"\n[ANALYZE IMAGE URL]")
    print(f"  URL     : {image_url[:80]}...")
    print(f"  Question: {question}")

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    image_url_content(image_url, detail="auto"),
                    {"type": "text", "text": question},
                ],
            }
        ],
        max_tokens=500,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens
    print(f"\n  Answer ({tokens} tokens):\n{answer}")
    return answer


def extract_data_from_image(client: openai.AzureOpenAI, image_url: str) -> str:
    """
    Extract structured data from an image (e.g. receipt, business card, form).
    Demonstrates vision as an OCR + understanding tool.
    """
    print(f"\n[EXTRACT DATA FROM IMAGE]")
    print(f"  Image: {image_url[:80]}...")

    extraction_prompt = """Analyze this image and extract all visible text and data.
Return the information as a structured JSON object with appropriate keys.
If this is a document, extract all fields.
If this is a scene, describe the main objects and their positions."""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You are a document extraction expert. Extract all visible data accurately.",
            },
            {
                "role": "user",
                "content": [
                    image_url_content(image_url, detail="high"),  # High detail for text extraction
                    {"type": "text", "text": extraction_prompt},
                ],
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0.0,  # Deterministic for extraction
    )

    result = response.choices[0].message.content
    print(f"  Extracted data:\n{result}")
    return result


def compare_images(client: openai.AzureOpenAI, image_url_1: str, image_url_2: str) -> str:
    """
    Send multiple images in a single request for comparison.
    GPT-4o can analyze relationships between multiple images.
    """
    print(f"\n[COMPARE TWO IMAGES]")

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images. What are the key similarities and differences?"},
                    image_url_content(image_url_1, detail="low"),
                    image_url_content(image_url_2, detail="low"),
                ],
            }
        ],
        max_tokens=300,
        temperature=0.4,
    )

    comparison = response.choices[0].message.content
    print(f"  Comparison:\n{comparison}")
    return comparison


def analyze_local_image(client: openai.AzureOpenAI, image_path: str, question: str) -> str:
    """
    Analyze a local image file by encoding it as base64.
    Falls back gracefully if the file doesn't exist.
    """
    print(f"\n[ANALYZE LOCAL IMAGE]")
    print(f"  File    : {image_path}")
    print(f"  Question: {question}")

    try:
        image_content = image_base64_content(image_path, detail="auto")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        print("  To test this, provide a local image path in the code.")
        return ""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": question},
                ],
            }
        ],
        max_tokens=300,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    print(f"  Answer:\n{answer}")
    return answer


def demo_image_description(client: openai.AzureOpenAI) -> None:
    """Generate a detailed description of a public image."""
    print("\n--- Demo: Image Description ---")

    # Publicly accessible image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

    analyze_image_url(
        client,
        image_url,
        "Describe this image in detail. Include the subject, background, lighting, and mood.",
    )


def demo_chart_analysis(client: openai.AzureOpenAI) -> None:
    """
    Analyze a chart or graph image to extract data and insights.
    Useful for document intelligence and business analytics workflows.
    """
    print("\n--- Demo: Chart/Graph Analysis ---")

    # Example: A publicly available chart image
    chart_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Chart-line-revenue.png/640px-Chart-line-revenue.png"

    analyze_image_url(
        client,
        chart_url,
        (
            "Analyze this chart. What type of chart is it? "
            "What data does it show? What are the key trends or insights? "
            "Extract any visible data points or values."
        ),
    )


def demo_accessibility_description(client: openai.AzureOpenAI) -> None:
    """
    Generate alt-text descriptions for accessibility.
    A common real-world use case for vision AI.
    """
    print("\n--- Demo: Accessibility Alt-Text Generation ---")

    images = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Andromeda_Galaxy_560mm_FL.jpg/1200px-Andromeda_Galaxy_560mm_FL.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/640px-Camponotus_flavomarginatus_ant.jpg",
    ]

    for url in images:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate concise, descriptive alt-text for images to improve web accessibility. "
                        "Alt-text should be 1-2 sentences, factual, and describe what is visually present."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        image_url_content(url, detail="low"),
                        {"type": "text", "text": "Generate alt-text for this image."},
                    ],
                }
            ],
            max_tokens=100,
            temperature=0.2,
        )
        alt_text = response.choices[0].message.content
        print(f"\n  Image: {url.split('/')[-1]}")
        print(f"  Alt-text: {alt_text}")


def demo_detail_level_comparison(client: openai.AzureOpenAI) -> None:
    """
    Compare low vs. high detail analysis.

    detail="low"  : 85 tokens, fast, good for general questions about scenes
    detail="high" : More tokens, slower, needed for small text, fine details
    """
    print("\n--- Demo: Detail Level Comparison ---")

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    question  = "What color are the cat's eyes? Describe any fine details you can see."

    for detail in ["low", "high"]:
        print(f"\n  detail='{detail}':")
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_url_content(image_url, detail=detail),
                        {"type": "text", "text": question},
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.2,
        )
        tokens = response.usage.total_tokens
        print(f"  Tokens used : {tokens}")
        print(f"  Answer: {response.choices[0].message.content}")


def main():
    print("=" * 60)
    print("Azure OpenAI Multimodal Vision Demo (GPT-4o)")
    print("=" * 60)
    print(f"Endpoint  : {ENDPOINT}")
    print(f"Deployment: {DEPLOYMENT}")

    try:
        client = get_client()

        demo_image_description(client)
        demo_chart_analysis(client)
        demo_accessibility_description(client)
        demo_detail_level_comparison(client)

        # Local image example (set a real path to test)
        analyze_local_image(
            client,
            "./sample_image.png",   # Change to a real local image path
            "What does this image show?",
        )

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except openai.NotFoundError as e:
        print(f"\n[ERROR] Deployment not found: {e}")
        print(f"Ensure '{DEPLOYMENT}' is a GPT-4o (or later) deployment with vision enabled.")
    except openai.APIError as e:
        print(f"\n[ERROR] Azure OpenAI error: {e}")


if __name__ == "__main__":
    main()
