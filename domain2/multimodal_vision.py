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
    """Build an image_url content block for a message."""
    return {
        "type": "image_url",
        "image_url": {"url": url, "detail": detail},
    }


def image_base64_content(image_path: str, detail: str = "auto") -> dict:
    """Read a local file and return a base64 image content block."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp"}
    mime = mime_map.get(suffix, "image/jpeg")
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{data}", "detail": detail},
    }


# ---------------------------------------------------------------------------
# Demo 1 - describe an image from a public URL
# ---------------------------------------------------------------------------
def demo_describe_url_image(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 1: Describe image from URL ===")

    url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    )

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this image."},
                    image_url_content(url, detail="low"),
                ],
            }
        ],
        max_tokens=300,
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Demo 2 - extract structured data from an image
# ---------------------------------------------------------------------------
def demo_extract_data_from_image(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 2: Extract structured data from image ===")

    url = "https://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg"

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract the following from this receipt as JSON:\n"
                            "{\"store\": \"\", \"date\": \"\", \"total\": \"\", "
                            "\"items\": [{\"name\": \"\", \"price\": \"\"}]}\n"
                            "Return only valid JSON."
                        ),
                    },
                    image_url_content(url, detail="high"),
                ],
            }
        ],
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Demo 3 - multi-image comparison
# ---------------------------------------------------------------------------
def demo_multi_image(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 3: Compare two images ===")

    url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
    url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two paintings. Describe the style and mood of each, then explain the key differences."},
                    image_url_content(url1, detail="low"),
                    image_url_content(url2, detail="low"),
                ],
            }
        ],
        max_tokens=400,
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Demo 4 - chart / graph analysis
# ---------------------------------------------------------------------------
def demo_chart_analysis(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 4: Analyze a chart ===")

    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Comparison_of_stock_exchange_indices.svg/1024px-Comparison_of_stock_exchange_indices.svg.png"

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a financial chart. Please:\n"
                            "1. Identify what the chart shows\n"
                            "2. Describe the main trends\n"
                            "3. Note any significant events or anomalies\n"
                            "4. Summarise in 3 bullet points"
                        ),
                    },
                    image_url_content(url, detail="high"),
                ],
            }
        ],
        max_tokens=400,
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Demo 5 - accessibility description (alt-text generation)
# ---------------------------------------------------------------------------
def demo_accessibility(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 5: Generate alt-text for accessibility ===")

    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/640px-Camponotus_flavomarginatus_ant.jpg"

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Generate a concise, descriptive alt-text (under 125 characters) "
                            "for this image suitable for screen readers. "
                            "Then provide a longer detailed description."
                        ),
                    },
                    image_url_content(url, detail="low"),
                ],
            }
        ],
        max_tokens=200,
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Demo 6 - base64 local image
# ---------------------------------------------------------------------------
def demo_base64_image(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 6: Describe a base64-encoded image ===")
    import urllib.request
    import tempfile

    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        img_block = image_base64_content(tmp.name, detail="low")

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image? (It was provided as base64)"},
                    img_block,
                ],
            }
        ],
        max_tokens=200,
    )
    print(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    client = get_client()

    demo_describe_url_image(client)
    demo_extract_data_from_image(client)
    demo_multi_image(client)
    demo_chart_analysis(client)
    demo_accessibility(client)
    demo_base64_image(client)

    print("\n=== Vision demos complete ===")


if __name__ == "__main__":
    main()
