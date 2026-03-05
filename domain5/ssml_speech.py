"""
ssml_speech.py
==============
Demonstrates using SSML (Speech Synthesis Markup Language) to control speech
output with the Azure Cognitive Services Speech SDK.

Exam skill: "Improve text-to-speech by using Speech Synthesis Markup Language
            (SSML)" (AI-102 Domain 5)

Concepts covered:
- Using speak_ssml_async() instead of speak_text_async()
- <voice> – selecting a specific neural voice
- <prosody> – adjusting rate, pitch, and volume
- <break> – inserting pauses of defined duration
- <emphasis> – stressing words
- <say-as> – controlling how specific text types are rendered
             (characters, digits, date, time, ordinal, telephone)
- <audio> – embedding audio clips (not used here to keep self-contained)
- <mstts:express-as> – Azure-specific speaking style (e.g. cheerful, sad)
- Saving each SSML example to its own WAV file

SSML Namespace:
    Standard W3C: xmlns="http://www.w3.org/2001/10/synthesis"
    Azure MSTTS : xmlns:mstts="http://www.w3.org/2001/mstts"
    Version     : version="1.0"

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
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_synthesizer(output_filename: str) -> speechsdk.SpeechSynthesizer:
    """Create a SpeechSynthesizer that writes output to a WAV file."""
    if not SPEECH_KEY:
        raise EnvironmentError(
            "AZURE_SPEECH_KEY must be set in your .env file."
        )

    config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)

    return speechsdk.SpeechSynthesizer(
        speech_config=config,
        audio_config=audio_config,
    )


def synthesise_ssml(ssml: str, output_filename: str, description: str) -> None:
    """
    Synthesise speech from an SSML string and save to a WAV file.

    Calls speak_ssml_async() (not speak_text_async()) so the full SSML
    markup is respected by the service.
    """
    print(f"\n  {description}")
    print(f"  Output: {output_filename}")

    synthesizer = get_synthesizer(output_filename)
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        size_kb = len(result.audio_data) / 1024
        print(f"  ✓ Saved  ({size_kb:.1f} KB)")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancel = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        print(f"  ✗ Cancelled: {cancel.reason}")
        if cancel.reason == speechsdk.CancellationReason.Error:
            print(f"    {cancel.error_details}")


# ---------------------------------------------------------------------------
# SSML examples
# ---------------------------------------------------------------------------

SSML_PROSODY = """\
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="en-US-JennyNeural">
    <!-- Default speech rate and pitch -->
    <p>This sentence is spoken at the normal rate and pitch.</p>

    <!-- Slow down to 75% of default rate, raise pitch slightly -->
    <prosody rate="75%" pitch="+5%">
      This sentence is spoken more slowly with a higher pitch.
    </prosody>

    <!-- Speed up to 120%, lower pitch, reduce volume -->
    <prosody rate="120%" pitch="-10%" volume="soft">
      And this one is faster, lower, and quieter.
    </prosody>
  </voice>
</speak>
"""

SSML_BREAKS = """\
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="en-US-AriaNeural">
    <!-- break time controls the pause length -->
    <p>Welcome to the Azure AI Language demo.</p>
    <break time="500ms"/>
    <p>In this section we will cover speech synthesis.</p>
    <break time="1s"/>
    <p>
      You can insert pauses of any length using the break element.
      <break strength="weak"/> Short pauses work well between clauses.
      <break strength="strong"/> Longer pauses separate major sections.
    </p>
  </voice>
</speak>
"""

SSML_EMPHASIS = """\
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="en-US-GuyNeural">
    <!-- emphasis levels: strong | moderate | reduced -->
    <p>
      Please remember to submit your exam <emphasis level="strong">before</emphasis>
      the deadline. Missing the deadline is a
      <emphasis level="moderate">serious</emphasis> issue.
      Minor details can be <emphasis level="reduced">adjusted</emphasis> later.
    </p>
  </voice>
</speak>
"""

SSML_SAY_AS = """\
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="en-US-JennyNeural">
    <!-- Characters: spell out letter-by-letter -->
    <p>Your PIN is <say-as interpret-as="characters">4821</say-as>.</p>

    <!-- Digits: each digit read individually -->
    <p>Call us at <say-as interpret-as="telephone">555-867-5309</say-as>.</p>

    <!-- Date: spoken as a full date -->
    <p>The exam is on <say-as interpret-as="date" format="mdy">03/15/2025</say-as>.</p>

    <!-- Ordinal: first, second, third… -->
    <p>She finished in <say-as interpret-as="ordinal">3</say-as> place.</p>

    <!-- Cardinal number as words -->
    <p>There are <say-as interpret-as="cardinal">1024</say-as> items in the list.</p>

    <!-- Fraction -->
    <p>The score was <say-as interpret-as="fraction">3/4</say-as>.</p>
  </voice>
</speak>
"""

SSML_STYLE = """\
<speak version="1.0"
       xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xml:lang="en-US">

  <!-- mstts:express-as is an Azure-specific extension for speaking styles -->
  <!-- Styles available depend on the voice; Jenny supports: cheerful, sad, angry, etc. -->

  <voice name="en-US-JennyNeural">
    <mstts:express-as style="cheerful" styledegree="1.5">
      Congratulations! You passed the Azure AI-102 exam with a fantastic score!
    </mstts:express-as>
  </voice>

  <voice name="en-US-JennyNeural">
    <mstts:express-as style="sad">
      Unfortunately, we were unable to process your request. Please try again later.
    </mstts:express-as>
  </voice>

  <voice name="en-US-JennyNeural">
    <mstts:express-as style="customerservice">
      Thank you for calling Azure support. How can I help you today?
    </mstts:express-as>
  </voice>
</speak>
"""

SSML_MULTILANG = """\
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <!-- Switch voices mid-document for a multilingual experience -->
  <voice name="en-US-JennyNeural">
    Hello! Welcome to our multilingual demo.
  </voice>
  <voice name="fr-FR-DeniseNeural">
    Bonjour et bienvenue dans notre démonstration multilingue.
  </voice>
  <voice name="de-DE-KatjaNeural">
    Hallo und willkommen zu unserer mehrsprachigen Demonstration.
  </voice>
  <voice name="en-US-JennyNeural">
    <break time="300ms"/>
    Thank you for listening!
  </voice>
</speak>
"""


def main() -> None:
    """Entry point – synthesise all SSML examples."""
    print("Azure Speech – SSML Demo")
    print(f"Region: {SPEECH_REGION}")
    print("\n" + "=" * 60)
    print("SSML SYNTHESIS EXAMPLES")
    print("=" * 60)

    examples = [
        (SSML_PROSODY,     "ssml_prosody.wav",   "Prosody (rate, pitch, volume)"),
        (SSML_BREAKS,      "ssml_breaks.wav",    "Breaks (pauses)"),
        (SSML_EMPHASIS,    "ssml_emphasis.wav",  "Emphasis (strong/moderate/reduced)"),
        (SSML_SAY_AS,      "ssml_say_as.wav",    "Say-as (characters, telephone, date, ordinal)"),
        (SSML_STYLE,       "ssml_style.wav",     "Speaking styles (mstts:express-as)"),
        (SSML_MULTILANG,   "ssml_multilang.wav", "Multi-voice / multilingual"),
    ]

    for ssml, filename, description in examples:
        synthesise_ssml(ssml, filename, description)

    print("\nAll WAV files written to:", OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
