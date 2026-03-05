"""
generate_images_dalle.py
========================
Demonstrates using Azure OpenAI DALL-E to generate images
from text descriptions.

Exam Skill: "Generate images with Azure OpenAI (DALL-E)"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Basic image generation with DALL-E 3
  - Quality options: standard vs hd
  - Style options: vivid vs natural
  - Size options: square, landscape, portrait
  - Saving generated images to disk
  - Inspecting the revised prompt
  - Error handling for content policy violations
  - Image generation URL expiry considerations

DALL-E 3 vs DALL-E 2 in Azure OpenAI:
  - DALL-E 3: Higher quality, larger sizes, better prompt adherence
    Sizes: 1024x1024, 1792x1024, 1024x1792
    Quality: standard or hd
    Style: vivid or natural
  - DALL-E 2: Older model, lower cost
    Sizes: 256x256, 512x512, 1024x1024
    No quality/style parameters

Required packages:
  pip install openai requests pillow python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT         - your Azure OpenAI endpoint
  AZURE_OPENAI_KEY              - your API key
  AZURE_OPENAI_DALLE_DEPLOYMENT - your DALL-E 3 deployment name
"""

import os
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.images_response import ImagesResponse

load_dotenv()

ENDPOINT         = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY          = os.environ["AZURE_OPENAI_KEY"]
DALLE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DALLE_DEPLOYMENT", "dall-e-3")

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version="2024-02-01",
)

# Output directory for saved images
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    n: int = 1,
) -> ImagesResponse:
    """
    Generate an image using DALL-E 3.

    Args:
        prompt:  Text description of the desired image
        size:    Image dimensions (1024x1024, 1792x1024, 1024x1792)
        quality: 'standard' (faster/cheaper) or 'hd' (higher detail)
        style:   'vivid' (dramatic/colorful) or 'natural' (realistic)
        n:       Number of images to generate (DALL-E 3 only supports n=1)

    Returns:
        ImagesResponse with URL and metadata
    """
    return client.images.generate(
        model=DALLE_DEPLOYMENT,
        prompt=prompt,
        size=size,
        quality=quality,
        style=style,
        n=n,
    )


def save_image(image_url: str, filename: str) -> Path:
    """
    Download and save a generated image.

    Note: Generated image URLs expire after ~24 hours.
    Always save images if you need them long-term.
    """
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    output_path = OUTPUT_DIR / filename
    output_path.write_bytes(response.content)
    print(f"  Saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Demo 1: Basic generation
# ---------------------------------------------------------------------------

def demo_basic_generation() -> None:
    """Generate a simple image with default settings."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Image Generation")
    print("=" * 60)

    prompt = "A futuristic Azure data center with glowing blue lights and server racks, digital art style"
    print(f"Prompt: {prompt}")

    response = generate_image(prompt)
    image = response.data[0]

    print(f"URL: {image.url[:80]}...")
    print(f"Revised prompt: {image.revised_prompt[:100] if image.revised_prompt else 'N/A'}...")

    # Save the image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_image(image.url, f"basic_{timestamp}.png")


# ---------------------------------------------------------------------------
# Demo 2: Quality and style comparison
# ---------------------------------------------------------------------------

def demo_quality_style_options() -> None:
    """Show different quality and style combinations."""
    print("\n" + "=" * 60)
    print("DEMO 2: Quality and Style Options")
    print("=" * 60)

    base_prompt = "A serene mountain lake at sunset with reflections in the water"

    configurations = [
        {"quality": "standard", "style": "vivid",   "desc": "Standard + Vivid (dramatic)"},
        {"quality": "standard", "style": "natural",  "desc": "Standard + Natural (realistic)"},
        {"quality": "hd",       "style": "vivid",   "desc": "HD + Vivid (high detail + dramatic)"},
        {"quality": "hd",       "style": "natural",  "desc": "HD + Natural (highest realism)"},
    ]

    for config in configurations:
        print(f"\n  Configuration: {config['desc']}")
        print(f"  Quality: {config['quality']}, Style: {config['style']}")

        try:
            response = generate_image(
                prompt=base_prompt,
                quality=config["quality"],
                style=config["style"],
            )
            image = response.data[0]
            print(f"  Generated: {image.url[:60]}...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{config['quality']}_{config['style']}_{timestamp}.png"
            save_image(image.url, fname)

        except Exception as e:
            print(f"  [ERROR] {e}")


# ---------------------------------------------------------------------------
# Demo 3: Different sizes
# ---------------------------------------------------------------------------

def demo_sizes() -> None:
    """Generate images in different sizes."""
    print("\n" + "=" * 60)
    print("DEMO 3: Image Sizes")
    print("=" * 60)

    sizes = [
        ("1024x1024", "Square - good for profiles, icons, social media posts"),
        ("1792x1024", "Landscape - good for banners, headers, wide scenes"),
        ("1024x1792", "Portrait - good for posters, mobile wallpapers"),
    ]

    prompt = "A professional Azure AI technology diagram with icons and connections"

    for size, description in sizes:
        print(f"\n  Size: {size} - {description}")
        try:
            response = generate_image(prompt=prompt, size=size)
            image = response.data[0]
            print(f"  Generated: {image.url[:60]}...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_image(image.url, f"size_{size.replace('x', '_')}_{timestamp}.png")

        except Exception as e:
            print(f"  [ERROR] {e}")


# ---------------------------------------------------------------------------
# Demo 4: Content policy and error handling
# ---------------------------------------------------------------------------

def demo_error_handling() -> None:
    """Demonstrate handling content policy errors."""
    print("\n" + "=" * 60)
    print("DEMO 4: Error Handling and Content Policy")
    print("=" * 60)

    from openai import BadRequestError

    # This should be a safe prompt
    safe_prompt = "A cartoon robot holding a graduation cap, friendly and colorful"

    print(f"  Testing safe prompt: '{safe_prompt[:60]}'")
    try:
        response = generate_image(safe_prompt)
        print(f"  [OK] Image generated successfully")
        print(f"  Revised prompt: {response.data[0].revised_prompt[:80] if response.data[0].revised_prompt else 'N/A'}...")
    except BadRequestError as e:
        print(f"  Content policy violation: {e}")
        print("  The prompt was rejected by the content filter")
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")

    print()
    print("  Key points about DALL-E content policy:")
    print("  - Prompts are filtered before generation")
    print("  - Generated images are filtered after generation")
    print("  - BadRequestError is raised for policy violations")
    print("  - Revised prompt shows how DALL-E interpreted your prompt")
    print("  - Image URLs expire after ~24 hours - save images you need")


if __name__ == "__main__":
    print("=" * 60)
    print("Azure OpenAI DALL-E Image Generation Demo")
    print("=" * 60)
    print(f"Endpoint: {ENDPOINT}")
    print(f"DALL-E deployment: {DALLE_DEPLOYMENT}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    demo_basic_generation()
    demo_quality_style_options()
    demo_sizes()
    demo_error_handling()

    print("\n" + "=" * 60)
    print("DALL-E Demo Complete")
    print("=" * 60)
