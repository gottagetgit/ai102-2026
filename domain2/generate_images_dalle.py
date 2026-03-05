"""
generate_images_dalle.py
========================
Demonstrates using Azure OpenAI DALL-E to generate images from text prompts.
Shows prompt construction, quality settings, size options, and image retrieval.

Exam Skill: "Use the DALL-E model to generate images"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Generating an image with DALL-E 3 via Azure OpenAI
  - Prompt engineering for image generation
  - Quality options: standard vs. hd
  - Size options: 1024x1024, 1792x1024, 1024x1792
  - Style options: vivid vs. natural
  - Downloading and saving generated images
  - Revised prompt (DALL-E 3 often rewrites your prompt)
  - Generating multiple variations

DALL-E 3 notes:
  - Only 1 image per request (n=1 is the only supported value)
  - Supports HD quality for higher detail
  - Returns a revised_prompt showing what DALL-E actually used
  - Images expire after 24 hours from the URL
  - Avoid prompts with real people by name (policy violation)

Required packages:
  pip install openai requests pillow python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT       - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY            - API key
  AZURE_OPENAI_DALLE_DEPLOYMENT - DALL-E 3 deployment name e.g. "dall-e-3"
"""

import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()

ENDPOINT         = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY          = os.environ["AZURE_OPENAI_KEY"]
DALLE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DALLE_DEPLOYMENT", "dall-e-3")
OUTPUT_DIR       = Path("./generated_images")


def get_client() -> openai.AzureOpenAI:
    """Return authenticated AzureOpenAI client."""
    return openai.AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version="2024-02-01",
    )


def generate_image(
    client: openai.AzureOpenAI,
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
) -> dict:
    """
    Generate a single image using DALL-E 3.

    Parameters:
        prompt  : Text description of the image to generate
        size    : Image dimensions
                  - "1024x1024" (square, default)
                  - "1792x1024" (landscape/wide)
                  - "1024x1792" (portrait/tall)
        quality : "standard" (faster, lower cost) or "hd" (more detail, higher cost)
        style   : "vivid" (hyper-real, dramatic) or "natural" (more subdued, realistic)

    Returns:
        dict with 'url', 'revised_prompt', and 'generation_params'
    """
    print(f"\n[GENERATE IMAGE]")
    print(f"  Prompt  : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"  Size    : {size}")
    print(f"  Quality : {quality}")
    print(f"  Style   : {style}")

    try:
        response = client.images.generate(
            model=DALLE_DEPLOYMENT,
            prompt=prompt,
            n=1,            # DALL-E 3 only supports n=1
            size=size,
            quality=quality,
            style=style,
        )
    except openai.BadRequestError as e:
        print(f"  [ERROR] Request rejected (likely content policy): {e}")
        return {}
    except openai.APIError as e:
        print(f"  [ERROR] API error: {e}")
        return {}

    image_url      = response.data[0].url
    revised_prompt = response.data[0].revised_prompt

    print(f"\n  Image URL: {image_url[:80]}...")
    print(f"  Revised prompt (what DALL-E actually used):")
    print(f"    {revised_prompt[:200]}{'...' if len(revised_prompt) > 200 else ''}")
    print("\n  [NOTE] URLs expire after 24 hours. Download the image to persist it.")

    return {
        "url": image_url,
        "revised_prompt": revised_prompt,
        "params": {"size": size, "quality": quality, "style": style},
    }


def download_image(url: str, filename: str) -> str:
    """
    Download an image from a URL and save it to disk.
    DALL-E image URLs expire after ~24 hours, so save them promptly.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / filename

    print(f"\n[DOWNLOAD] Saving to {filepath}...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  Saved: {filepath} ({size_kb:.0f} KB)")
        return str(filepath)
    except requests.RequestException as e:
        print(f"  [ERROR] Download failed: {e}")
        return ""


def demo_prompt_quality_comparison(client: openai.AzureOpenAI) -> None:
    """
    Shows how prompt quality affects image results.

    Poor prompts: vague, no style, no composition details
    Good prompts: specific subject, style, lighting, composition, mood
    """
    print("\n--- Prompt Quality Comparison ---")

    prompts = [
        {
            "name": "Vague prompt",
            "prompt": "a city",
        },
        {
            "name": "Detailed prompt",
            "prompt": (
                "A futuristic cyberpunk city at night, neon signs in Japanese and English "
                "reflecting on rain-slicked streets, flying vehicles in the distance, "
                "street vendors with glowing food stalls, cinematic composition, "
                "highly detailed, dramatic lighting, 8K render"
            ),
        },
    ]

    for p in prompts:
        print(f"\n  {p['name']}:")
        result = generate_image(client, p["prompt"], quality="standard")
        if result.get("url"):
            safe_name = p["name"].replace(" ", "_").lower()
            download_image(result["url"], f"{safe_name}.png")
        # Brief pause between API calls
        time.sleep(2)


def demo_size_and_quality_options(client: openai.AzureOpenAI) -> None:
    """
    Demonstrate different size and quality combinations.

    Size guide:
      1024x1024 - Social media posts, profile images, app icons
      1792x1024 - Hero banners, landscape scenes, wide-format content
      1024x1792 - Mobile wallpapers, portrait illustrations, book covers

    Quality guide:
      standard  - Marketing assets, quick prototypes
      hd        - Print-quality output, detailed illustrations
    """
    print("\n--- Size and Quality Options ---")

    base_prompt = (
        "A serene Japanese zen garden with raked sand patterns, "
        "moss-covered stones, a small wooden bridge over a koi pond, "
        "cherry blossom trees in spring, soft morning light"
    )

    configs = [
        {"size": "1024x1024", "quality": "standard", "style": "natural",  "name": "square_standard_natural"},
        {"size": "1792x1024", "quality": "standard", "style": "vivid",    "name": "landscape_standard_vivid"},
        {"size": "1024x1792", "quality": "hd",        "style": "natural",  "name": "portrait_hd_natural"},
    ]

    for cfg in configs:
        print(f"\n  Config: {cfg['name']}")
        result = generate_image(
            client,
            base_prompt,
            size=cfg["size"],
            quality=cfg["quality"],
            style=cfg["style"],
        )
        if result.get("url"):
            download_image(result["url"], f"{cfg['name']}.png")
        time.sleep(2)


def demo_style_comparison(client: openai.AzureOpenAI) -> None:
    """
    Compare 'vivid' vs. 'natural' style on the same prompt.

    vivid   : Hyper-real, dramatic colors, more artistic interpretation
    natural : More subdued, photorealistic, closer to what a camera would capture
    """
    print("\n--- Style Comparison: vivid vs. natural ---")

    prompt = "A coffee shop on a rainy day, warm light inside, raindrops on the window"

    for style in ["vivid", "natural"]:
        print(f"\n  Style: {style}")
        result = generate_image(client, prompt, size="1024x1024", quality="standard", style=style)
        if result.get("url"):
            download_image(result["url"], f"coffee_shop_{style}.png")
        time.sleep(2)


def demo_revised_prompt_analysis(client: openai.AzureOpenAI) -> None:
    """
    DALL-E 3 rewrites your prompt (the 'revised_prompt') to add detail and
    comply with content policy. Analyzing the revised prompt helps you
    understand what DALL-E interpreted and improve your prompts iteratively.
    """
    print("\n--- Revised Prompt Analysis ---")

    short_prompt = "a dog in a park"

    print(f"  Original prompt: '{short_prompt}'")
    result = generate_image(client, short_prompt)

    if result.get("revised_prompt"):
        original_words = len(short_prompt.split())
        revised_words  = len(result["revised_prompt"].split())
        print(f"\n  Original: {original_words} words")
        print(f"  Revised : {revised_words} words")
        print(f"\n  Key additions: DALL-E added breed, pose, background, style, and lighting details.")
        print("  Tip: Use the revised prompt as a starting point for your next iteration.")


def main():
    print("=" * 60)
    print("Azure OpenAI DALL-E Image Generation Demo")
    print("=" * 60)
    print(f"Endpoint  : {ENDPOINT}")
    print(f"Deployment: {DALLE_DEPLOYMENT}")
    print(f"Output dir: {OUTPUT_DIR.absolute()}")

    try:
        client = get_client()

        # Run demos sequentially (DALL-E has rate limits)
        demo_prompt_quality_comparison(client)
        demo_style_comparison(client)
        demo_size_and_quality_options(client)
        demo_revised_prompt_analysis(client)

        print(f"\n[COMPLETE] Generated images saved to: {OUTPUT_DIR.absolute()}")

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
        print("Set AZURE_OPENAI_DALLE_DEPLOYMENT in your .env file.")
    except openai.NotFoundError as e:
        print(f"\n[ERROR] DALL-E deployment not found: {e}")
        print(f"Create a DALL-E 3 deployment in Azure OpenAI Studio and set AZURE_OPENAI_DALLE_DEPLOYMENT.")
    except openai.APIError as e:
        print(f"\n[ERROR] Azure OpenAI error: {e}")


if __name__ == "__main__":
    main()
