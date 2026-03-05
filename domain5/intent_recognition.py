"""
intent_recognition.py
=====================
Demonstrates intent and keyword recognition using the Azure Cognitive Services
Speech SDK integrated with a Conversational Language Understanding (CLU) model.

Exam skill: "Implement intent and keyword recognition with Azure Speech in
            Foundry Tools" (AI-102 Domain 5)

Concepts covered:
- Keyword recognition (local, offline detection of a wake word)
- Intent recognition using CLU via SpeechRecognizer + LanguageUnderstandingModel
- Recognising from a WAV file and from the microphone
- Accessing intent name, entities, and confidence from the JSON result
- ConversationLanguageUnderstandingServiceSampleData / IntentRecognitionResult
- Continuous intent recognition pattern

Note on Integration Modes:
    Mode 1 – Keyword Recognition:
        Runs entirely on-device. Detects a custom keyword (e.g. "Hey Cortana")
        before waking the cloud recogniser. Uses a .table model file built
        with the Azure Custom Keyword service.

    Mode 2 – Intent Recognition with CLU:
        The speech audio is sent to Azure Speech for transcription, then the
        transcript is forwarded to your CLU project for intent classification.
        This requires a deployed CLU model (see clu_model.py).

Required env vars:
    AZURE_SPEECH_KEY          – subscription key
    AZURE_SPEECH_REGION       – e.g. eastus
    AZURE_LANGUAGE_ENDPOINT   – CLU resource endpoint
    AZURE_LANGUAGE_KEY        – CLU resource key
    CLU_PROJECT_NAME          – name of the CLU project (e.g. "HomeAutomation")
    CLU_DEPLOYMENT_NAME       – deployment name (e.g. "production")

Optional:
    SPEECH_AUDIO_FILE         – path to WAV file for recognition

Install:
    pip install azure-cognitiveservices-speech python-dotenv
"""

import os
import json
import wave
import struct
import threading
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
import azure.cognitiveservices.speech.intent as intentsdk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")
LANGUAGE_ENDPOINT = os.environ.get("AZURE_LANGUAGE_ENDPOINT", "")
LANGUAGE_KEY = os.environ.get("AZURE_LANGUAGE_KEY", "")
CLU_PROJECT_NAME = os.environ.get("CLU_PROJECT_NAME", "HomeAutomation")
CLU_DEPLOYMENT_NAME = os.environ.get("CLU_DEPLOYMENT_NAME", "production")
AUDIO_FILE = os.environ.get("SPEECH_AUDIO_FILE", "")


def _create_test_wav(path: str) -> None:
    """Generate a short silent WAV file for testing."""
    sample_rate = 16000
    num_samples = sample_rate * 2  # 2 seconds

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * num_samples, *([0] * num_samples)))

    print(f"  [INFO] Created silent test WAV: {path}")
    print("         Replace with a real recording (e.g. 'Turn on the lights') for results.\n")


# ---------------------------------------------------------------------------
# Demo 1: Pattern-based intent recognition (simple, no CLU required)
# ---------------------------------------------------------------------------

def pattern_intent_recognition(audio_path: str) -> None:
    """
    Recognise intents from simple phrase patterns without a CLU model.

    PatternMatchingModel lets you define intents with exact phrase patterns
    and optional entity slots {entity_name} inline.  This is useful for
    prototyping or simple command vocabularies without training a full model.

    Exam tip: PatternMatchingModel is local/offline for entity extraction
    but still sends audio to Azure Speech for the actual transcription.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: PATTERN-BASED INTENT RECOGNITION")
    print("=" * 60)

    if not SPEECH_KEY:
        raise EnvironmentError("AZURE_SPEECH_KEY must be set in your .env file.")

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    # Define a pattern-matching model with inline intents
    model = intentsdk.PatternMatchingModel(model_id="HomeControl")

    intents = [
        intentsdk.PatternMatchingIntent(
            name="TurnOn",
            phrases=["turn on the {device}", "switch on {device}", "turn {device} on"],
        ),
        intentsdk.PatternMatchingIntent(
            name="TurnOff",
            phrases=["turn off the {device}", "switch off {device}", "turn {device} off"],
        ),
        intentsdk.PatternMatchingIntent(
            name="SetTemperature",
            phrases=["set the temperature to {temperature} degrees",
                     "make it {temperature} degrees"],
        ),
    ]

    for intent in intents:
        model.intents.append(intent)

    recognizer = intentsdk.IntentRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )
    recognizer.apply_language_models([model])

    print(f"  Recognising from: {audio_path}")
    result = recognizer.recognize_once_async().get()
    _handle_intent_result(result)


def _handle_intent_result(result: intentsdk.IntentRecognitionResult) -> None:
    """Print intent recognition result in a readable format."""
    if result.reason == speechsdk.ResultReason.RecognizedIntent:
        print(f"\n  ✓ Recognised: \"{result.text}\"")
        print(f"    Intent    : {result.intent_id}")

        # Parse entities from JSON result
        try:
            entities_json = json.loads(result.intent_json)
            entities = entities_json.get("entities", {})
            if entities:
                print("    Entities  :")
                for name, value in entities.items():
                    print(f"      {name}: {value}")
        except (json.JSONDecodeError, AttributeError):
            pass

    elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"\n  Speech recognised but no intent matched: \"{result.text}\"")

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("\n  ✗ No speech or intent recognised.")

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancel = speechsdk.CancellationDetails.from_result(result)
        print(f"\n  ✗ Cancelled: {cancel.reason}")
        if cancel.reason == speechsdk.CancellationReason.Error:
            print(f"    Error: {cancel.error_details}")


# ---------------------------------------------------------------------------
# Demo 2: CLU-backed intent recognition
# ---------------------------------------------------------------------------

def clu_intent_recognition(audio_path: str) -> None:
    """
    Recognise intent using a deployed Conversational Language Understanding (CLU)
    model via the Speech SDK.

    The ConversationLanguageUnderstandingModel connects the Speech recogniser
    to your CLU project.  The audio is first transcribed, then the transcript
    is sent to CLU for intent classification and entity extraction.

    Exam tip: This pattern replaces the legacy LUIS integration.  CLU is the
    modern successor to LUIS and is accessed via Azure AI Language.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: CLU-BACKED INTENT RECOGNITION")
    print(f"  Project   : {CLU_PROJECT_NAME}")
    print(f"  Deployment: {CLU_DEPLOYMENT_NAME}")
    print("=" * 60)

    if not all([SPEECH_KEY, LANGUAGE_ENDPOINT, LANGUAGE_KEY]):
        print(
            "  [SKIP] AZURE_SPEECH_KEY, AZURE_LANGUAGE_ENDPOINT, and AZURE_LANGUAGE_KEY "
            "must all be set to run this demo."
        )
        return

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    # Connect the Speech recogniser to a CLU model
    clu_model = intentsdk.ConversationLanguageUnderstandingModel(
        endpoint=LANGUAGE_ENDPOINT,
        api_key=LANGUAGE_KEY,
        project_name=CLU_PROJECT_NAME,
        deployment_name=CLU_DEPLOYMENT_NAME,
    )

    recognizer = intentsdk.IntentRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )
    recognizer.apply_language_models([clu_model])

    print(f"  Recognising from: {audio_path}")
    result = recognizer.recognize_once_async().get()
    _handle_intent_result(result)

    # The full CLU JSON response is in result.intent_json
    if result.reason == speechsdk.ResultReason.RecognizedIntent:
        print("\n  Full CLU JSON response:")
        try:
            parsed = json.loads(result.intent_json)
            print(json.dumps(parsed, indent=4))
        except (json.JSONDecodeError, AttributeError):
            print(f"  {result.intent_json}")


# ---------------------------------------------------------------------------
# Demo 3: Keyword recognition (wake-word detection)
# ---------------------------------------------------------------------------

def keyword_recognition_demo() -> None:
    """
    Demonstrate keyword spotting – triggering a recogniser only when a
    specific wake word is detected.

    In a real scenario you would:
        1. Train a custom keyword model in the Azure Custom Keyword portal
        2. Download the .table model file
        3. Load it with KeywordRecognitionModel.from_file(path)

    This demo shows the wiring pattern; without a real .table file the
    recogniser is initialised but the keyword model load is skipped.

    Exam tip: Keyword recognition runs locally on the device (edge).
    When the keyword is detected, it wakes the cloud-based speech recogniser.
    This enables privacy-preserving, always-on voice activation.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: KEYWORD RECOGNITION PATTERN")
    print("=" * 60)

    keyword_model_path = os.environ.get("KEYWORD_MODEL_PATH", "")

    if not keyword_model_path or not os.path.isfile(keyword_model_path):
        print(
            "  [INFO] KEYWORD_MODEL_PATH not set or file not found.\n"
            "         Showing wiring pattern only (no live recognition).\n\n"
            "  To use keyword recognition:\n"
            "    1. Build a custom keyword model at:\n"
            "       https://speech.microsoft.com/customkeyword\n"
            "    2. Download the .table file\n"
            "    3. Set KEYWORD_MODEL_PATH in your .env file\n"
        )
        print("  Pattern (what the code would do with a real .table file):\n")
        print(
            "    model = speechsdk.KeywordRecognitionModel('path/to/keyword.table')\n"
            "    recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)\n"
            "    # Start listening for the wake word continuously\n"
            "    recognizer.recognize_keyword_async(model)\n"
            "    # On keyword detected → run full speech recognition\n"
            "    # recognizer.recognized event fires with the utterance\n"
        )
        return

    if not SPEECH_KEY:
        raise EnvironmentError("AZURE_SPEECH_KEY must be set in your .env file.")

    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    model = speechsdk.KeywordRecognitionModel(keyword_model_path)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    done = threading.Event()

    def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        print(f"  Keyword detected! Utterance: \"{evt.result.text}\"")
        done.set()

    recognizer.recognized.connect(on_recognized)

    print("  Listening for wake word…  (say your keyword, then speak)")
    recognizer.recognize_keyword_async(model)

    done.wait(timeout=30)
    recognizer.stop_keyword_recognition_async()
    print("  Keyword recognition stopped.")


def main() -> None:
    """Entry point."""
    print("Azure Speech – Intent & Keyword Recognition Demo")
    print(f"Region: {SPEECH_REGION}")

    # Resolve audio file
    audio_path = AUDIO_FILE
    if not audio_path or not os.path.isfile(audio_path):
        audio_path = "/tmp/test_intent.wav"
        _create_test_wav(audio_path)

    pattern_intent_recognition(audio_path)
    clu_intent_recognition(audio_path)
    keyword_recognition_demo()

    print("\nDone.")


if __name__ == "__main__":
    main()
