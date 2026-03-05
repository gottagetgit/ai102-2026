"""
sentiment_analysis.py
=====================
Demonstrates sentiment analysis at the document AND sentence level, including
opinion mining (aspect-based sentiment analysis) using Azure AI Language.

Exam skill: "Determine sentiment of text" (AI-102 Domain 5)

Concepts covered:
- Document-level sentiment (Positive / Negative / Neutral / Mixed)
- Sentence-level sentiment with confidence scores
- Opinion mining: target (aspect) + assessment (opinion) pairs
- Batch processing of multiple documents

Required env vars:
    AZURE_LANGUAGE_ENDPOINT  – e.g. https://<resource>.cognitiveservices.azure.com/
    AZURE_LANGUAGE_KEY       – 32-character key from Azure portal

Install:
    pip install azure-ai-textanalytics python-dotenv
"""

import os
from dotenv import load_dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

ENDPOINT = os.environ.get("AZURE_LANGUAGE_ENDPOINT")
KEY = os.environ.get("AZURE_LANGUAGE_KEY")

# ---------------------------------------------------------------------------
# Sample documents covering a range of sentiments
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "1",
        "language": "en",
        "text": (
            "The hotel was absolutely stunning and the staff were incredibly helpful. "
            "However, the food at the restaurant was disappointing and overpriced."
        ),
    },
    {
        "id": "2",
        "language": "en",
        "text": "I love the new Azure AI features. They make development so much easier!",
    },
    {
        "id": "3",
        "language": "en",
        "text": "The product arrived damaged and customer support never responded. Terrible experience.",
    },
    {
        "id": "4",
        "language": "en",
        "text": "The conference was held in Seattle. There were several sessions on AI.",
    },
    {
        "id": "5",
        "language": "en",
        "text": (
            "The laptop battery life is fantastic – I get over 12 hours on a charge. "
            "The keyboard feels a bit mushy but the screen is gorgeous."
        ),
    },
]


def get_client() -> TextAnalyticsClient:
    """Create and return an authenticated TextAnalyticsClient."""
    if not ENDPOINT or not KEY:
        raise EnvironmentError(
            "AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY must be set in your .env file."
        )
    return TextAnalyticsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def _sentiment_bar(score: float, width: int = 20) -> str:
    """Return a simple ASCII bar representing a 0-1 confidence score."""
    filled = round(score * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {score:.2f}"


def analyse_sentiment(client: TextAnalyticsClient) -> None:
    """
    Perform sentiment analysis with opinion mining enabled.

    show_opinion_mining=True activates aspect-based sentiment: the API returns
    'mined opinions' for each sentence – a target (noun) and assessment (adjective)
    along with their individual sentiment and confidence scores.

    Exam tip: The sentiment labels are Positive, Negative, Neutral, and Mixed.
    Mixed means the document contains both positive and negative sentences.
    """
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS  (with Opinion Mining)")
    print("=" * 60)

    try:
        # show_opinion_mining requires model_version="latest" or >= "2021-10-01"
        response = client.analyze_sentiment(
            DOCUMENTS,
            show_opinion_mining=True,
        )
    except Exception as exc:
        print(f"[ERROR] analyze_sentiment failed: {exc}")
        return

    for doc in response:
        if doc.is_error:
            print(f"\nDoc {doc.id} ERROR: {doc.error.message}")
            continue

        # ---- Document-level result ----------------------------------------
        print(f"\n--- Document {doc.id} ---")
        print(f"  Overall sentiment : {doc.sentiment.upper()}")
        scores = doc.confidence_scores
        print(f"  Positive  {_sentiment_bar(scores.positive)}")
        print(f"  Neutral   {_sentiment_bar(scores.neutral)}")
        print(f"  Negative  {_sentiment_bar(scores.negative)}")

        # ---- Sentence-level results ----------------------------------------
        for i, sentence in enumerate(doc.sentences, start=1):
            print(f"\n  Sentence {i}: \"{sentence.text}\"")
            print(f"    Sentiment : {sentence.sentiment}")
            s = sentence.confidence_scores
            print(f"    Pos={s.positive:.2f}  Neu={s.neutral:.2f}  Neg={s.negative:.2f}")

            # ---- Opinion mining (aspect-based sentiment) -------------------
            if sentence.mined_opinions:
                print("    Opinion Mining:")
                for opinion in sentence.mined_opinions:
                    target = opinion.target
                    print(f"      Target   : '{target.text}'  [{target.sentiment}]")
                    for assessment in opinion.assessments:
                        neg_marker = "  *** NEGATED ***" if assessment.is_negated else ""
                        print(
                            f"      Assessment: '{assessment.text}'  "
                            f"[{assessment.sentiment}]{neg_marker}"
                        )


def main() -> None:
    """Entry point."""
    print("Azure AI Language – Sentiment Analysis Demo")
    print("Endpoint:", ENDPOINT or "(not set)")

    client = get_client()
    analyse_sentiment(client)
    print("\nDone.")


if __name__ == "__main__":
    main()
