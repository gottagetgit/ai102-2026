"""
text_translation.py
===================
Demonstrates translating text using the Azure Translator REST API.

Exam skill: "Translate text and documents by using the Azure Translator in
            Foundry Tools service" (AI-102 Domain 5)

Concepts covered:
- Multi-target translation (one call → multiple target languages)
- Language auto-detection (omitting 'from' parameter)
- Transliteration (script/alphabet conversion without translation)
- Dictionary lookup (bilingual dictionary with context)
- Accessing detected language and alternative translations

The Azure Translator exposes a REST API rather than a dedicated Python SDK,
so this demo uses the 'requests' library with the standard auth headers.

Required env vars:
    AZURE_TRANSLATOR_KEY      – subscription key from Azure portal
    AZURE_TRANSLATOR_ENDPOINT – e.g. https://api.cognitive.microsofttranslator.com
                                (or your custom endpoint if using a named resource)
    AZURE_TRANSLATOR_REGION   – e.g. eastus  (required for multi-service keys)

Install:
    pip install requests python-dotenv
"""

import os
import json
import uuid
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

API_VERSION = "3.0"


def _auth_headers() -> dict:
    """Return the authentication headers required for every Translator request."""
    if not TRANSLATOR_KEY:
        raise EnvironmentError(
            "AZURE_TRANSLATOR_KEY must be set in your .env file."
        )
    return {
        "Ocp-Apim-Subscription-Key": TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": TRANSLATOR_REGION,
        "Content-Type": "application/json",
        # Unique trace ID helps with debugging in Azure Monitor
        "X-ClientTraceId": str(uuid.uuid4()),
    }


def translate_text(texts: list[str], to_languages: list[str], from_language: str | None = None) -> None:
    """
    Translate one or more strings into multiple target languages in a single call.

    If from_language is None, the service auto-detects the source language.
    Each target language is specified as a BCP-47 tag, e.g. 'fr', 'de', 'ja'.

    Exam tip: A single /translate call can target up to ~10 languages at once.
    """
    url = f"{TRANSLATOR_ENDPOINT}/translate"
    params = {"api-version": API_VERSION, "to": to_languages}
    if from_language:
        params["from"] = from_language

    body = [{"text": t} for t in texts]

    print("\n" + "=" * 60)
    print("TEXT TRANSLATION")
    print(f"  From: {from_language or '(auto-detect)'}")
    print(f"  To  : {', '.join(to_languages)}")
    print("=" * 60)

    try:
        resp = requests.post(url, headers=_auth_headers(), params=params, json=body, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        print(f"[ERROR] HTTP {exc.response.status_code}: {exc.response.text}")
        return
    except requests.RequestException as exc:
        print(f"[ERROR] Request failed: {exc}")
        return

    results = resp.json()

    for i, result in enumerate(results):
        original = texts[i]
        detected = result.get("detectedLanguage")
        print(f"\n  Source [{i+1}]: \"{original}\"")
        if detected:
            print(f"  Detected language: {detected['language']}  (score: {detected['score']:.2f})")

        for translation in result.get("translations", []):
            print(f"    → [{translation['to'].upper()}] {translation['text']}")


def transliterate_text(text: str, language: str, from_script: str, to_script: str) -> None:
    """
    Convert text between scripts (writing systems) without changing the language.

    Example: convert Japanese hiragana/katakana → Latin script (romanisation).
    The 'language' parameter is the BCP-47 tag for the text's language.
    'from_script' / 'to_script' are IANA script codes (e.g. 'Jpan', 'Latn').

    Exam tip: Transliteration ≠ translation.  The meaning stays in the same
    language but the alphabet/script changes.
    """
    url = f"{TRANSLATOR_ENDPOINT}/transliterate"
    params = {
        "api-version": API_VERSION,
        "language": language,
        "fromScript": from_script,
        "toScript": to_script,
    }
    body = [{"text": text}]

    print("\n" + "=" * 60)
    print("TRANSLITERATION")
    print(f"  Language: {language}  |  {from_script} → {to_script}")
    print("=" * 60)

    try:
        resp = requests.post(url, headers=_auth_headers(), params=params, json=body, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        print(f"[ERROR] HTTP {exc.response.status_code}: {exc.response.text}")
        return
    except requests.RequestException as exc:
        print(f"[ERROR] Request failed: {exc}")
        return

    for result in resp.json():
        print(f"  Input : {text}")
        print(f"  Output: {result['text']}  (script: {result['script']})")


def dictionary_lookup(word: str, from_language: str, to_language: str) -> None:
    """
    Look up a word using the bilingual dictionary.

    The dictionary endpoint returns:
        - Possible translations with part-of-speech tags
        - Back-translations (reverse translations to validate context)
        - Frequency / confidence score for each translation pair

    Exam tip: The dictionary is best suited to single words or short phrases.
    Use /translate for full-sentence translation.
    """
    url = f"{TRANSLATOR_ENDPOINT}/dictionary/lookup"
    params = {
        "api-version": API_VERSION,
        "from": from_language,
        "to": to_language,
    }
    body = [{"text": word}]

    print("\n" + "=" * 60)
    print(f"DICTIONARY LOOKUP: '{word}'  ({from_language} → {to_language})")
    print("=" * 60)

    try:
        resp = requests.post(url, headers=_auth_headers(), params=params, json=body, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        print(f"[ERROR] HTTP {exc.response.status_code}: {exc.response.text}")
        return
    except requests.RequestException as exc:
        print(f"[ERROR] Request failed: {exc}")
        return

    for result in resp.json():
        for trans in result.get("translations", []):
            print(
                f"  {trans['displayTarget']:<20} "
                f"[{trans['posTag']}]  "
                f"confidence: {trans['confidence']:.2f}"
            )
            # Show first two back-translations as validation context
            for bt in trans.get("backTranslations", [])[:2]:
                print(f"      back: '{bt['displayText']}'  freq={bt['frequencyCount']}")


def main() -> None:
    """Entry point – run all three translation demos."""
    print("Azure Translator – Text Translation Demo")
    print("Endpoint:", TRANSLATOR_ENDPOINT)
    print("Region  :", TRANSLATOR_REGION)

    # 1. Multi-target translation with auto language detection
    translate_text(
        texts=[
            "Hello! How are you doing today?",
            "Azure AI makes building intelligent applications easy.",
            "The quick brown fox jumps over the lazy dog.",
        ],
        to_languages=["fr", "de", "ja", "es"],
        from_language=None,   # auto-detect
    )

    # 2. Transliteration: Japanese → Latin script
    transliterate_text(
        text="こんにちは世界",
        language="ja",
        from_script="Jpan",
        to_script="Latn",
    )

    # 3. Dictionary lookup
    dictionary_lookup(word="happy", from_language="en", to_language="fr")

    print("\nDone.")


if __name__ == "__main__":
    main()
