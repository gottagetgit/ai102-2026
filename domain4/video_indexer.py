"""
video_indexer.py
================
Demonstrates using the Azure AI Video Indexer REST API to:
    1. Obtain an access token for the account
    2. Upload a video (from a URL) and start indexing
    3. Poll for indexing completion
    4. Retrieve and display rich insights:
       - Transcript (with timestamps)
       - Detected topics / labels
       - Keywords
       - Named entities (people, locations, brands)
       - Sentiments
       - Faces (if enabled on the account)
       - OCR text found in frames
    5. Generate a URL for the Video Indexer web widget (player + insights)

Video Indexer uses a multi-modal approach combining:
    - Speech-to-text for the transcript
    - NLP for named entity recognition, sentiment, and topics
    - Computer vision for face detection, labels, and OCR
    - Audio analysis for noise events

Exam Skill Mapping:
    - "Use Azure AI Video Indexer to extract insights from a video or live stream"

Required Environment Variables (.env):
    VIDEO_INDEXER_ACCOUNT_ID   - GUID of your Video Indexer account
    VIDEO_INDEXER_LOCATION     - Azure region (e.g. "trial" for free tier, or "eastus")
    VIDEO_INDEXER_API_KEY      - API key from the Video Indexer developer portal
                                 (https://api-portal.videoindexer.ai)

Note on authentication:
    Video Indexer uses its own token-based auth, separate from Azure AD.
    The API key from the developer portal is used to obtain account tokens.

Install:
    pip install requests python-dotenv
"""

import os
import time
import json
from dotenv import load_dotenv
import requests

load_dotenv()

ACCOUNT_ID = os.environ.get("VIDEO_INDEXER_ACCOUNT_ID")
LOCATION   = os.environ.get("VIDEO_INDEXER_LOCATION", "trial")
API_KEY    = os.environ.get("VIDEO_INDEXER_API_KEY")

# Video Indexer API base URL
VI_BASE_URL = "https://api.videoindexer.ai"

# Sample public video URL (short clip for demo)
SAMPLE_VIDEO_URL = (
    "https://download.microsoft.com/download/B/A/8/"
    "BA8E6A07-1573-4FAA-A0B5-0D1CEB67A7D5/BuildVideo.mp4"
)


def get_account_access_token() -> str:
    """Obtain an account-scoped access token using the API key.

    Account tokens allow listing videos, uploading, and reading insights.
    Token expiry is 1 hour; refresh by calling this function again.

    Returns:
        Access token string.
    """
    if not ACCOUNT_ID or not API_KEY:
        raise ValueError(
            "VIDEO_INDEXER_ACCOUNT_ID and VIDEO_INDEXER_API_KEY must be set in .env"
        )

    url = (
        f"{VI_BASE_URL}/auth/{LOCATION}/Accounts/{ACCOUNT_ID}"
        f"/AccessToken?allowEdit=true"
    )
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    # Response is a quoted JSON string, strip quotes
    token = response.json()
    return token


def upload_video_from_url(
    access_token: str,
    video_url: str,
    video_name: str = "demo-video",
    language: str = "en-US",
) -> str:
    """Upload a video from a URL and start indexing.

    Args:
        access_token: Account access token from get_account_access_token().
        video_url:    Publicly accessible URL to the video file.
        video_name:   Descriptive name for the video in the portal.
        language:     Primary language of the audio (BCP-47 code).
                      Use "auto" to enable automatic language detection.

    Returns:
        Video ID string assigned by Video Indexer.
    """
    url = f"{VI_BASE_URL}/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos"
    params = {
        "accessToken":  access_token,
        "name":         video_name,
        "videoUrl":     video_url,
        "language":     language,
        "indexingPreset": "Default",
        # indexingPreset options:
        #   "Default"          - Standard: audio + basic video insights
        #   "AudioOnly"        - Faster, transcript and audio insights only
        #   "BasicVideoOnly"   - Fast, labels and scenes only (no audio analysis)
        #   "AdvancedVideo"    - Slow, deep visual analysis including faces, scenes
        #   "AdvancedAudio"    - Deep audio: emotions, speakers, music
    }
    response = requests.post(url, params=params, timeout=60)
    response.raise_for_status()
    video_data = response.json()
    video_id = video_data["id"]
    print(f"  Video uploaded. ID: {video_id}, state: {video_data.get('state', 'unknown')}")
    return video_id


def poll_indexing_status(
    access_token: str,
    video_id: str,
    poll_interval: int = 15,
    max_wait_minutes: int = 20,
) -> dict:
    """Poll until video indexing is complete or fails.

    Indexing time depends on video length and selected preset.
    Typical speeds: ~1x realtime for Default preset.

    Args:
        access_token:      Account access token.
        video_id:          Video ID from upload_video_from_url().
        poll_interval:     Seconds between status checks.
        max_wait_minutes:  Give up after this many minutes.

    Returns:
        Video index JSON dict when processing is complete.
    """
    url = f"{VI_BASE_URL}/{LOCATION}/Accounts/{ACCOUNT_ID}/Videos/{video_id}/Index"
    headers = {"Content-Type": "application/json"}

    max_polls = (max_wait_minutes * 60) // poll_interval
    polls = 0

    while polls < max_polls:
        params = {"accessToken": access_token}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        state    = data.get("state", "unknown")
        progress = data.get("videos", [{}])[0].get("processingProgress", "?")

        print(f"  [{polls+1}] State: {state}, Progress: {progress}")

        if state == "Processed":
            print("  Indexing complete!")
            return data
        elif state == "Failed":
            raise RuntimeError(
                f"Video indexing failed. Error: {data.get('failureMessage', 'unknown')}"
            )

        time.sleep(poll_interval)
        polls += 1

    raise TimeoutError(f"Video indexing did not complete within {max_wait_minutes} minutes.")


def extract_insights(video_index: dict) -> dict:
    """Extract and structure the rich insights from the video index response.

    The Video Indexer index JSON has a nested structure:
        summarizedInsights: rolled-up insights across all clips
        videos[].insights:  per-clip detailed insights

    Args:
        video_index: The full index response JSON dict.

    Returns:
        Dict containing structured insights.
    """
    insights = {}

    # ------------------------------------------------------------------
    # Summarised insights (rolled up across the whole video)
    # ------------------------------------------------------------------
    summarized = video_index.get("summarizedInsights", {})

    # Transcript (from detailed per-video insights)
    transcript_lines = []
    for video in video_index.get("videos", []):
        vi = video.get("insights", {})

        # Transcript: [{id, text, confidence, instances:[{start,end}]}]
        for entry in vi.get("transcript", []):
            for instance in entry.get("instances", []):
                transcript_lines.append({
                    "start":      instance.get("start", ""),
                    "end":        instance.get("end", ""),
                    "text":       entry.get("text", ""),
                    "confidence": round(entry.get("confidence", 0), 4),
                })

        # Keywords
        insights["keywords"] = [
            {
                "text":       k.get("text", ""),
                "confidence": round(k.get("confidence", 0), 4),
            }
            for k in vi.get("keywords", [])
        ]

        # Topics
        insights["topics"] = [
            {
                "name":       t.get("name", ""),
                "confidence": round(t.get("confidence", 0), 4),
                "ipt_ref":    t.get("referenceUrl", ""),
            }
            for t in vi.get("topics", [])
        ]

        # Labels (visual content labels)
        insights["labels"] = [
            {
                "name":       lb.get("name", ""),
                "confidence": round(lb.get("confidence", 0), 4),
            }
            for lb in vi.get("labels", [])
        ]

        # Named entities
        insights["named_entities"] = [
            {
                "name":       ne.get("name", ""),
                "type":       ne.get("type", ""),
                "confidence": round(ne.get("confidence", 0), 4),
            }
            for ne in vi.get("namedLocations", []) + vi.get("namedPeople", [])
        ]

        # Sentiments
        insights["sentiments"] = [
            {
                "sentiment_key": s.get("sentimentKey", ""),
                "score":         round(s.get("averageScore", 0), 4),
            }
            for s in vi.get("sentiments", [])
        ]

        # Faces (requires face recognition to be enabled)
        insights["faces"] = [
            {
                "name":       f.get("name", "Unknown"),
                "confidence": round(f.get("confidence", 0), 4),
                "description": f.get("description", ""),
            }
            for f in vi.get("faces", [])
        ]

        # OCR text found in video frames
        insights["ocr"] = [
            {
                "text":       o.get("text", ""),
                "confidence": round(o.get("confidence", 0), 4),
            }
            for o in vi.get("ocr", [])
        ]

    insights["transcript"] = sorted(transcript_lines, key=lambda x: x["start"])

    # Overall statistics
    insights["duration"]  = summarized.get("duration", {}).get("time", "")
    insights["audio_effects"] = summarized.get("audioEffects", [])

    return insights


def get_widget_url(access_token: str, video_id: str) -> str:
    """Generate a URL for the Video Indexer web player + insights widget.

    This widget can be embedded in web pages using an <iframe>.

    Args:
        access_token: Account access token.
        video_id:     Video ID.

    Returns:
        Widget URL string.
    """
    url = (
        f"{VI_BASE_URL}/{LOCATION}/Accounts/{ACCOUNT_ID}/"
        f"Videos/{video_id}/InsightsWidget"
    )
    params = {
        "accessToken": access_token,
        "widgetType":  "KeyFrames",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.url  # The redirect URL is the widget embed URL


def print_insights(insights: dict) -> None:
    """Display extracted video insights in a readable format."""
    print("\n" + "=" * 60)
    print("VIDEO INDEXER INSIGHTS")
    print("=" * 60)

    print(f"\nDuration: {insights.get('duration', 'unknown')}")

    # Transcript (first 10 lines)
    transcript = insights.get("transcript", [])
    if transcript:
        print(f"\n[TRANSCRIPT] ({len(transcript)} segments — showing first 10)")
        for seg in transcript[:10]:
            print(f"  [{seg['start']} → {seg['end']}] (conf={seg['confidence']:.2f})")
            print(f"  '{seg['text']}'")

    # Topics
    topics = insights.get("topics", [])
    if topics:
        print(f"\n[TOPICS] ({len(topics)} detected)")
        for t in sorted(topics, key=lambda x: x["confidence"], reverse=True)[:10]:
            print(f"  {t['name']:<40} conf={t['confidence']:.4f}")

    # Keywords
    keywords = insights.get("keywords", [])
    if keywords:
        print(f"\n[KEYWORDS] ({len(keywords)} detected)")
        kw_str = ", ".join(k["text"] for k in keywords[:20])
        print(f"  {kw_str}")

    # Labels
    labels = insights.get("labels", [])
    if labels:
        print(f"\n[VISUAL LABELS] ({len(labels)} detected)")
        for lb in sorted(labels, key=lambda x: x["confidence"], reverse=True)[:10]:
            bar = "█" * int(lb["confidence"] * 20)
            print(f"  {lb['name']:<30} {lb['confidence']:.4f}  {bar}")

    # Named entities
    entities = insights.get("named_entities", [])
    if entities:
        print(f"\n[NAMED ENTITIES] ({len(entities)} detected)")
        for ne in entities[:15]:
            print(f"  [{ne['type']}] {ne['name']} (conf={ne['confidence']:.4f})")

    # Sentiments
    sentiments = insights.get("sentiments", [])
    if sentiments:
        print(f"\n[SENTIMENTS]")
        for s in sentiments:
            print(f"  {s['sentiment_key']:<15} score={s['score']:.4f}")

    # Faces
    faces = insights.get("faces", [])
    if faces:
        print(f"\n[FACES] ({len(faces)} detected)")
        for face in faces:
            print(f"  {face['name']} (conf={face['confidence']:.4f})")

    # OCR
    ocr_items = insights.get("ocr", [])
    if ocr_items:
        print(f"\n[OCR TEXT IN FRAMES] ({len(ocr_items)} items)")
        for item in ocr_items[:10]:
            print(f"  '{item['text']}' (conf={item['confidence']:.4f})")

    print("\n" + "=" * 60)


def save_insights_to_file(insights: dict, filename: str = "video_insights.json") -> None:
    """Save the insights dict to a JSON file for further processing."""
    with open(filename, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"\nFull insights saved to: {filename}")


def run_video_indexer_demo():
    """Full Video Indexer demonstration."""
    print("[1/5] Obtaining access token...")
    token = get_account_access_token()
    print(f"  Token obtained (length={len(token)})")

    print("\n[2/5] Uploading video for indexing...")
    video_id = upload_video_from_url(
        access_token=token,
        video_url=SAMPLE_VIDEO_URL,
        video_name="AI102-Demo-Video",
        language="en-US",
    )

    print("\n[3/5] Polling for indexing completion (may take several minutes)...")
    video_index = poll_indexing_status(token, video_id, poll_interval=15)

    print("\n[4/5] Extracting insights...")
    insights = extract_insights(video_index)

    print_insights(insights)
    save_insights_to_file(insights)

    print("\n[5/5] Generating widget URL...")
    try:
        widget_url = get_widget_url(token, video_id)
        print(f"  Widget URL (embed in <iframe>):\n  {widget_url}")
    except Exception as e:
        print(f"  Could not generate widget URL: {e}")

    print("\nDemo complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Video Indexer Demo ===\n")
    try:
        run_video_indexer_demo()
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
