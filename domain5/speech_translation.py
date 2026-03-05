"""
speech_translation.py
=====================
Demonstrates speech translation using the Azure Cognitive Services Speech SDK:
spoken audio in one language is translated to text (and optionally synthesised
speech) in one or more target languages.

Exam skill: "Translate speech-to-speech and speech-to-text by using the Azure
            Speech in Foundry Tools service" (AI-102 Domain 5)

Concepts covered:
- SpeechTranslationConfig vs SpeechConfig
- Translating speech to text in multiple target languages simultaneously
- Speech-to-speech synthesis of translated output (voice synthesis callback)
- Handling TranslationRecognitionResult with translated text dict
- Continuous speech translation from a file
- Event-driven callbacks: recognizing, recognized, synthesizing, canceled

Architecture:
    Audio input → SpeechTranslationRecognizer → Translated text (N languages)
                                               → Optional voice synthesis

Required env vars:
    AZURE_SPEECH_KEY     – subscription key from Azure portal
    AZURE_SPEECH_REGION  – e.g. eastus, westeurope

Optional:
    SPEECH_AUDIO_FILE    – path to a WAV file (default: generates a silent test WAV)

Install:
    pip install azure-cognitiveservices-speech python-dotenv
"""

import os
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

# Languages to translate INTO (BCP-47 codes)
TARGET_LANGUAGES = ["fr", "de", "ja"]

# Source language – set to None to auto-detect
SOURCE_LANGUAGE = "en-US"

# Voice for speech-to-speech synthesis (one per target language)
SYNTHESIS_VOICE = {
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
}


def _create_test_wav(path: str) -> None:
    """Generate a short silent WAV file for testing (replace with real audio)."""
    sample_rate = 16000
    num_samples = sample_rate * 3   # 3 seconds

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))

    print(f"  [INFO] Created silent test WAV: {path}")
    print("         Provide a real WAV file via SPEECH_AUDIO_FILE for actual translation.\n")


def get_translation_config() -> speechsdk.translation.SpeechTranslationConfig:
    """
    Create a SpeechTranslationConfig.

    Key differences from SpeechConfig:
        - add_target_language() / target_languages: one or more output languages
        - speech_recognition_language: the spoken source language (or auto-detect)
        - voice_name: optional – enables speech-to-speech synthesis
    """
    if not SPEECH_KEY:
        raise EnvironmentError(
            "AZURE_SPEECH_KEY must be set in your .env file."
        )

    config = speechsdk.translation.SpeechTranslationConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION,
    )

    # Set the source (spoken) language
    if SOURCE_LANGUAGE:
        config.speech_recognition_language = SOURCE_LANGUAGE

    # Add one target language at a time
    for lang in TARGET_LANGUAGES:
        config.add_target_language(lang)

    return config


# ---------------------------------------------------------------------------
# Demo 1: One-shot translation
# ---------------------------------------------------------------------------

def translate_once(audio_path: str) -> None:
    """
    Translate a single utterance and print all translated texts.

    result.translations is a dict: { 'fr': '...', 'de': '...', 'ja': '...' }
    """
    print("\n" + "=" * 60)
    print("DEMO 1: ONE-SHOT SPEECH TRANSLATION")
    print(f"  Source: {SOURCE_LANGUAGE}  →  Target: {', '.join(TARGET_LANGUAGES)}")
    print(f"  File  : {audio_path}")
    print("=" * 60)

    config = get_translation_config()
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=config,
        audio_config=audio_config,
    )

    print("  Translating…")
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        print(f"\n  ✓ Recognised (source): \"{result.text}\"")
        print("\n  Translations:")
        for lang, translated_text in result.translations.items():
            print(f"    [{lang.upper()}] {translated_text}")

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print(f"\n  ✗ No speech recognised: {speechsdk.NoMatchDetails.from_result(result).reason}")

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancel = speechsdk.CancellationDetails.from_result(result)
        print(f"\n  ✗ Cancelled: {cancel.reason}")
        if cancel.reason == speechsdk.CancellationReason.Error:
            print(f"    Error: {cancel.error_details}")


# ---------------------------------------------------------------------------
# Demo 2: Continuous translation with speech-to-speech synthesis
# ---------------------------------------------------------------------------

def translate_continuous(audio_path: str) -> None:
    """
    Continuously translate speech from a file, printing partial and final
    results, and synthesising each finalised translation to audio.

    Speech-to-speech synthesis:
        Set config.voice_name to a neural voice for one of the target languages.
        The synthesizing event fires with raw audio data you can play or save.
        Only ONE synthesis voice can be active per recogniser; for multiple
        output languages you would run multiple recognisers.

    Exam tip: For live voice translation between two people, pair continuous
    recognition with a SpeechSynthesizer to play back translated audio.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: CONTINUOUS SPEECH TRANSLATION")
    print(f"  Source: {SOURCE_LANGUAGE}  →  Targets: {', '.join(TARGET_LANGUAGES)}")
    print(f"  File  : {audio_path}")
    print("=" * 60)

    config = get_translation_config()

    # Enable speech-to-speech synthesis for French output
    synthesis_lang = TARGET_LANGUAGES[0]  # e.g. 'fr'
    config.voice_name = SYNTHESIS_VOICE.get(synthesis_lang, "")

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=config,
        audio_config=audio_config,
    )

    done = threading.Event()
    synthesis_output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "translated_audio.raw"
    )
    synthesis_chunks: list[bytes] = []

    def on_recognizing(evt: speechsdk.translation.TranslationRecognitionEventArgs) -> None:
        print(f"  [partial] {evt.result.text}", end="\r")

    def on_recognized(evt: speechsdk.translation.TranslationRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech:
            print(f"  [final]   {evt.result.text}                    ")
            for lang, text in evt.result.translations.items():
                print(f"    → [{lang.upper()}] {text}")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("  [no match]")

    def on_synthesizing(evt: speechsdk.translation.TranslationSynthesisEventArgs) -> None:
        # Collect raw PCM audio chunks from speech-to-speech synthesis
        if evt.result.audio:
            synthesis_chunks.append(evt.result.audio)

    def on_session_stopped(evt: speechsdk.SessionEventArgs) -> None:
        print("\n  Session stopped.")
        done.set()

    def on_canceled(evt: speechsdk.translation.TranslationRecognitionCanceledEventArgs) -> None:
        if evt.result.reason == speechsdk.CancellationReason.Error:
            print(f"\n  [ERROR] {evt.result.cancellation_details.error_details}")
        done.set()

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.synthesizing.connect(on_synthesizing)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.canceled.connect(on_canceled)

    recognizer.start_continuous_recognition_async()
    print("  Translating (continuous)…\n")

    try:
        done.wait(timeout=120)
    except KeyboardInterrupt:
        pass
    finally:
        recognizer.stop_continuous_recognition_async().get()

    # Save any synthesised audio chunks
    if synthesis_chunks:
        audio_data = b"".join(synthesis_chunks)
        with open(synthesis_output_path, "wb") as f:
            f.write(audio_data)
        print(f"\n  Synthesised audio saved to: {synthesis_output_path}")
        print(f"  ({len(audio_data)} bytes of raw PCM, 16 kHz mono 16-bit)")


def main() -> None:
    """Entry point."""
    print("Azure Speech – Speech Translation Demo")
    print(f"Region: {SPEECH_REGION}")

    # Resolve audio file
    audio_path = AUDIO_FILE
    if not audio_path or not os.path.isfile(audio_path):
        audio_path = "/tmp/test_speech_trans.wav"
        _create_test_wav(audio_path)

    translate_once(audio_path)
    translate_continuous(audio_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
