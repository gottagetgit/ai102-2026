"""
text_to_speech.py
=================
Demonstrates text-to-speech (TTS) synthesis using the Azure Cognitive
Services Speech SDK.

Exam skill: "Implement text-to-speech and speech-to-text using Azure Speech
            in Foundry Tools" (AI-102 Domain 5)

Concepts covered:
- Synthesising speech and saving to a WAV file
- Selecting neural voices (multi-lingual and locale-specific)
- Synthesising directly to the speaker (for interactive demos)
- Listing available voices filtered by locale
- Handling SynthesisResult reasons
- Adjusting output audio format

Required env vars:
    AZURE_SPEECH_KEY     – subscription key from Azure portal
    AZURE_SPEECH_REGION  – e.g. eastus, westeurope

Install:
    pip install azure-cognitiveservices-speech python-dotenv
"""

import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")

# Output WAV files will be written next to this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_speech_config(voice_name: str | None = None) -> speechsdk.SpeechConfig:
    """
    Create a SpeechConfig, optionally setting a specific neural voice.

    Voice names follow the pattern:  <Locale>-<Name>Neural
    e.g.  en-US-JennyNeural,  en-GB-RyanNeural,  fr-FR-DeniseNeural
    """
    if not SPEECH_KEY:
        raise EnvironmentError(
            "AZURE_SPEECH_KEY must be set in your .env file."
        )
    config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)

    # Set the desired audio output format (24 kHz, 96 kbps, 16-bit PCM WAV)
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )

    if voice_name:
        config.speech_synthesis_voice_name = voice_name

    return config


def synthesise_to_file(text: str, voice_name: str, filename: str) -> str:
    """
    Synthesise speech for 'text' using 'voice_name' and save to 'filename'.

    Returns the path to the saved WAV file.

    ResultReason values to check:
        SynthesizingAudioCompleted – success
        Canceled                   – error (check CancellationDetails)
    """
    output_path = os.path.join(OUTPUT_DIR, filename)

    speech_config = get_speech_config(voice_name)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    print(f"\n  Synthesising: \"{text[:60]}{'…' if len(text) > 60 else ''}\"")
    print(f"  Voice  : {voice_name}")
    print(f"  Output : {output_path}")

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        size_kb = len(result.audio_data) / 1024
        print(f"  ✓ Done  ({size_kb:.1f} KB)")
        return output_path
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        print(f"  ✗ Synthesis cancelled: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f"    Error code   : {cancellation.error_code}")
            print(f"    Error details: {cancellation.error_details}")
        return ""
    else:
        print(f"  ✗ Unexpected reason: {result.reason}")
        return ""


def list_voices(locale_filter: str = "en-") -> None:
    """
    Retrieve and print available neural voices filtered by locale prefix.

    Exam tip: Azure has hundreds of neural voices across 140+ locales.
    All modern voices are Neural (no Standard voices for new resources).
    Some voices support multiple styles (e.g. cheerful, empathetic, newscast).
    """
    print("\n" + "=" * 60)
    print(f"AVAILABLE VOICES  (filter: '{locale_filter}*')")
    print("=" * 60)

    speech_config = get_speech_config()
    # Use a null audio output so we don't write any audio
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=None,
    )

    try:
        result = synthesizer.get_voices_async(locale_filter).get()
    except Exception as exc:
        print(f"[ERROR] get_voices failed: {exc}")
        return

    if result.reason == speechsdk.ResultReason.VoicesListRetrieved:
        voices = [v for v in result.voices if v.locale.startswith(locale_filter)]
        print(f"  Found {len(voices)} voices:\n")
        print(f"  {'Name':<40} {'Locale':<12} {'Gender':<8} Styles")
        print("  " + "-" * 80)
        for voice in voices[:20]:   # cap at 20 for readability
            styles = ", ".join(voice.style_list[:3]) if voice.style_list else "default"
            if len(voice.style_list) > 3:
                styles += f" (+{len(voice.style_list)-3} more)"
            print(
                f"  {voice.short_name:<40} {voice.locale:<12} "
                f"{voice.gender.name:<8} {styles}"
            )
        if len(voices) > 20:
            print(f"  … and {len(voices)-20} more.")
    else:
        print(f"  Could not retrieve voices: {result.reason}")


def main() -> None:
    """Entry point – demonstrate multiple voices and save WAV files."""
    print("Azure Speech – Text-to-Speech Demo")
    print(f"Region: {SPEECH_REGION}")

    print("\n" + "=" * 60)
    print("DEMO 1: ENGLISH NEURAL VOICES")
    print("=" * 60)

    # en-US Jenny – warm, conversational
    synthesise_to_file(
        text=(
            "Welcome to Azure AI! I'm Jenny, a neural voice from Microsoft. "
            "I can speak in a natural, conversational tone."
        ),
        voice_name="en-US-JennyNeural",
        filename="output_jenny.wav",
    )

    # en-GB Ryan – British English male
    synthesise_to_file(
        text=(
            "Good day! My name is Ryan. I'm a British English neural voice. "
            "Azure Speech supports over a hundred and forty locales worldwide."
        ),
        voice_name="en-GB-RyanNeural",
        filename="output_ryan.wav",
    )

    print("\n" + "=" * 60)
    print("DEMO 2: MULTILINGUAL VOICES")
    print("=" * 60)

    synthesise_to_file(
        text="Bonjour! Je m'appelle Denise. Je suis une voix neurale en français.",
        voice_name="fr-FR-DeniseNeural",
        filename="output_denise.wav",
    )

    synthesise_to_file(
        text="Hola, mi nombre es Elvira. Soy una voz neuronal en español.",
        voice_name="es-ES-ElviraNeural",
        filename="output_elvira.wav",
    )

    # List available English voices
    list_voices(locale_filter="en-")

    print("\nDone.")


if __name__ == "__main__":
    main()
