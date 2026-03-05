"""
speech_to_text.py
=================
Demonstrates real-time speech recognition (speech-to-text) using the
Azure Cognitive Services Speech SDK.

Exam skill: "Implement text-to-speech and speech-to-text using Azure Speech
            in Foundry Tools" (AI-102 Domain 5)

Concepts covered:
- One-shot recognition from a WAV file (recognize_once_async)
- Checking ResultReason to handle different outcome types
- Continuous recognition with an event-driven callback pattern
- Using SpeechConfig vs AudioConfig
- Recognising from microphone input (pattern shown, but requires audio hardware)
- CancellationDetails and error handling

Required env vars:
    AZURE_SPEECH_KEY     – subscription key from Azure portal
    AZURE_SPEECH_REGION  – e.g. eastus, westeurope

Optional:
    SPEECH_AUDIO_FILE    – path to a .wav file for recognition (default: creates
                           a short test wav if none provided)

Install:
    pip install azure-cognitiveservices-speech python-dotenv
"""

import os
import time
import wave
import struct
import threading
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")
AUDIO_FILE = os.environ.get("SPEECH_AUDIO_FILE", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_test_wav(path: str) -> None:
    """
    Generate a minimal valid WAV file containing silence.
    This lets the script run end-to-end even without a real audio file –
    the recogniser will return NoMatch since there is no speech.
    Replace with a real WAV recording to get actual transcription results.
    """
    sample_rate = 16000
    duration_secs = 2
    num_samples = sample_rate * duration_secs

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit PCM
        wf.setframerate(sample_rate)
        # Write silence (all zeros)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))

    print(f"  [INFO] Created silent test WAV: {path}")
    print("         Replace with a real recording to see transcription results.\n")


def get_speech_config() -> speechsdk.SpeechConfig:
    """Create and return a SpeechConfig from environment variables."""
    if not SPEECH_KEY:
        raise EnvironmentError(
            "AZURE_SPEECH_KEY must be set in your .env file."
        )
    config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    # Request detailed JSON output (includes confidence, word timings, etc.)
    config.output_format = speechsdk.OutputFormat.Detailed
    return config


# ---------------------------------------------------------------------------
# Demo 1: One-shot recognition from a file
# ---------------------------------------------------------------------------

def recognise_once_from_file(audio_path: str) -> None:
    """
    Perform a single recognition pass on an audio file.

    recognize_once_async() waits for the first utterance (sentence) and returns.
    It is suitable for short commands or single questions.

    ResultReason values to handle:
        RecognizedSpeech  – speech was detected and transcribed successfully
        NoMatch           – audio was received but no speech was recognised
        Canceled          – the operation was cancelled (check CancellationDetails)

    Exam tip: For production use prefer continuous recognition (see below)
    so long utterances and multi-sentence audio are fully captured.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: ONE-SHOT RECOGNITION FROM FILE")
    print(f"  File: {audio_path}")
    print("=" * 60)

    speech_config = get_speech_config()
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    print("  Recognising…")
    result = recognizer.recognize_once_async().get()

    _handle_result(result)


def _handle_result(result: speechsdk.SpeechRecognitionResult) -> None:
    """Inspect a SpeechRecognitionResult and print appropriate output."""
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"\n  ✓ Recognised: \"{result.text}\"")

        # Detailed JSON contains confidence score and word-level timing
        import json
        try:
            detail = json.loads(result.json)
            best = detail.get("NBest", [{}])[0]
            confidence = best.get("Confidence", "n/a")
            print(f"    Confidence: {confidence}")
            words = best.get("Words", [])
            if words:
                print("    Word timings (offset ms, duration ms):")
                for w in words[:10]:   # show first 10 words
                    print(f"      '{w['Word']}'  offset={w['Offset']//10000}ms  "
                          f"dur={w['Duration']//10000}ms")
        except (json.JSONDecodeError, KeyError):
            pass  # detailed JSON not always present

    elif result.reason == speechsdk.ResultReason.NoMatch:
        no_match_detail = speechsdk.NoMatchDetails.from_result(result)
        print(f"\n  ✗ No speech recognised: {no_match_detail.reason}")

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = speechsdk.CancellationDetails.from_result(result)
        print(f"\n  ✗ Recognition cancelled: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f"    Error code   : {cancellation.error_code}")
            print(f"    Error details: {cancellation.error_details}")


# ---------------------------------------------------------------------------
# Demo 2: Continuous recognition from a file
# ---------------------------------------------------------------------------

def recognise_continuous_from_file(audio_path: str) -> None:
    """
    Continuously recognise speech from an audio file until the stream ends.

    Continuous recognition fires callbacks for every recognised utterance.
    Use this pattern when processing long recordings, live streams, or
    when you need to capture multiple sentences.

    Event callbacks used here:
        recognizing  – interim / partial hypothesis (text may change)
        recognized   – final transcript for an utterance
        session_stopped / canceled – end of stream / error
    """
    print("\n" + "=" * 60)
    print("DEMO 2: CONTINUOUS RECOGNITION FROM FILE")
    print(f"  File: {audio_path}")
    print("=" * 60)

    speech_config = get_speech_config()
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    done = threading.Event()
    transcript_parts: list[str] = []

    def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        # Interim result – text may still change
        print(f"  [partial] {evt.result.text}", end="\r")

    def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"  [final]   {evt.result.text}          ")
            transcript_parts.append(evt.result.text)
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("  [no match]")

    def on_session_stopped(evt: speechsdk.SessionEventArgs) -> None:
        print("\n  Session stopped – recognition complete.")
        done.set()

    def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        if evt.result.reason == speechsdk.CancellationReason.Error:
            print(f"\n  [ERROR] {evt.result.cancellation_details.error_details}")
        done.set()

    # Wire up callbacks
    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.canceled.connect(on_canceled)

    recognizer.start_continuous_recognition_async()
    print("  Recognising (press Ctrl+C to stop early)…\n")

    try:
        done.wait(timeout=120)   # max 2 minutes
    except KeyboardInterrupt:
        pass
    finally:
        recognizer.stop_continuous_recognition_async().get()

    if transcript_parts:
        print("\n  Full transcript:")
        print("  " + " ".join(transcript_parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    print("Azure Speech – Speech-to-Text Demo")
    print(f"Region: {SPEECH_REGION}")

    # Resolve audio file path
    audio_path = AUDIO_FILE
    if not audio_path or not os.path.isfile(audio_path):
        audio_path = "/tmp/test_speech.wav"
        _create_test_wav(audio_path)

    recognise_once_from_file(audio_path)
    recognise_continuous_from_file(audio_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
