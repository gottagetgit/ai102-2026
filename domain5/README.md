# Domain 5 – Implement Natural Language Processing Solutions

> **AI-102 Exam Weight: 15–20%**

This directory contains Python demo scripts covering every skill in the AI-102
Domain 5 exam objective. Each file is a complete, runnable script that makes
real Azure AI service calls.

---

## Prerequisites

### 1. Install dependencies

```bash
pip install \
  azure-ai-textanalytics \
  azure-ai-translation-document \
  azure-ai-language-conversations \
  azure-ai-language-questionanswering \
  azure-cognitiveservices-speech \
  requests \
  python-dotenv
```

### 2. Create a `.env` file

Copy the template below into a file named `.env` in this directory (or the
repo root) and fill in your Azure resource values:

```dotenv
# Azure AI Language (Text Analytics / CLU / CQA)
AZURE_LANGUAGE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_LANGUAGE_KEY=<32-char-key>

# Azure Speech Service
AZURE_SPEECH_KEY=<32-char-key>
AZURE_SPEECH_REGION=eastus

# Azure Translator
AZURE_TRANSLATOR_KEY=<32-char-key>
AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com
AZURE_TRANSLATOR_REGION=eastus

# Document Translation (uses Language resource endpoint, NOT global translator endpoint)
# AZURE_TRANSLATOR_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/

# Blob storage SAS URLs for document_translation.py
AZURE_SOURCE_SAS_URL=https://<account>.blob.core.windows.net/<container>?<sas-token>
AZURE_TARGET_SAS_URL=https://<account>.blob.core.windows.net/<container>?<sas-token>

# CLU project settings for intent_recognition.py and clu_model.py
CLU_PROJECT_NAME=HomeAutomation
CLU_DEPLOYMENT_NAME=production

# Custom Translator
CUSTOM_TRANSLATOR_CATEGORY_ID=<guid-from-custom-translator-portal>

# Optional: path to a WAV file for speech demos
SPEECH_AUDIO_FILE=/path/to/audio.wav
```

---

## Files at a Glance

| File | Exam Skill | Azure Service | SDK / Package |
|------|-----------|---------------|---------------|
| [`key_phrases_entities.py`](#key_phrases_entitiespy) | Extract key phrases and entities | Azure AI Language | `azure-ai-textanalytics` |
| [`sentiment_analysis.py`](#sentiment_analysispy) | Determine sentiment of text | Azure AI Language | `azure-ai-textanalytics` |
| [`language_detection.py`](#language_detectionpy) | Detect the language used in text | Azure AI Language | `azure-ai-textanalytics` |
| [`pii_detection.py`](#pii_detectionpy) | Detect PII in text | Azure AI Language | `azure-ai-textanalytics` |
| [`text_translation.py`](#text_translationpy) | Translate text with Azure Translator | Azure Translator | `requests` (REST) |
| [`document_translation.py`](#document_translationpy) | Translate documents (async blob) | Azure Document Translation | `azure-ai-translation-document` |
| [`speech_to_text.py`](#speech_to_textpy) | Speech-to-text | Azure Speech | `azure-cognitiveservices-speech` |
| [`text_to_speech.py`](#text_to_speechpy) | Text-to-speech | Azure Speech | `azure-cognitiveservices-speech` |
| [`ssml_speech.py`](#ssml_speechpy) | SSML for TTS control | Azure Speech | `azure-cognitiveservices-speech` |
| [`speech_translation.py`](#speech_translationpy) | Speech-to-speech and speech-to-text translation | Azure Speech | `azure-cognitiveservices-speech` |
| [`intent_recognition.py`](#intent_recognitionpy) | Intent and keyword recognition | Azure Speech + CLU | `azure-cognitiveservices-speech` |
| [`clu_model.py`](#clu_modelpy) | Build, train, deploy, and query a CLU model | Azure AI Language (CLU) | `azure-ai-language-conversations` |
| [`custom_question_answering.py`](#custom_question_answeringpy) | Full CQA knowledge base lifecycle | Azure AI Language (CQA) | `azure-ai-language-questionanswering` |
| [`custom_translator.py`](#custom_translatorpy) | Custom translation model workflow | Azure Custom Translator | `requests` (REST) |

---

## File Details

### key_phrases_entities.py

**Exam skill:** Extract key phrases and entities

Demonstrates three features of the Azure AI Language `TextAnalyticsClient`:

- **Key phrase extraction** – returns the most salient noun phrases from text
- **Named Entity Recognition (NER)** – returns entities with category
  (Person, Organization, Location, DateTime, etc.), sub-category, and
  confidence score
- **Entity Linking** – resolves entities to Wikipedia entries with a URL

Processes a batch of four documents including English and Spanish text.

```bash
python key_phrases_entities.py
```

---

### sentiment_analysis.py

**Exam skill:** Determine sentiment of text

Shows document-level and sentence-level sentiment analysis with **opinion mining**
(aspect-based sentiment). For each sentence, the API returns:

- `target` – the aspect being discussed (e.g. "food", "battery life")
- `assessments` – opinion words (e.g. "amazing", "disappointing") with their
  own sentiment score and negation flag

Sentiment labels: **Positive**, **Negative**, **Neutral**, **Mixed**

```bash
python sentiment_analysis.py
```

---

### language_detection.py

**Exam skill:** Detect the language used in text

Detects the language of 11 sample texts (English, Spanish, French, German,
Japanese, Arabic, Russian, Portuguese, Hindi, ambiguous short text, and
mixed-language input). Prints:

- Language name and ISO 639-1 code
- Confidence score (flags low-confidence results)

```bash
python language_detection.py
```

---

### pii_detection.py

**Exam skill:** Detect personally identifiable information (PII) in text

Detects PII across four documents covering names, email, phone, credit cards,
SSNs, passport numbers, driver's licences, and IP addresses. Demonstrates:

- Accessing the service-generated **redacted text** (safe to log/store)
- PII categories and sub-categories
- Raw PII masked in terminal output for safety

```bash
python pii_detection.py
```

---

### text_translation.py

**Exam skill:** Translate text and documents by using the Azure Translator in Foundry Tools service

Uses the Translator REST API (v3.0) to show three operations:

1. **Multi-target translation** – translate to French, German, Japanese, and Spanish in one call with auto-detected source language
2. **Transliteration** – convert Japanese text to Latin script
3. **Dictionary lookup** – bilingual dictionary with part-of-speech tags and back-translations

```bash
python text_translation.py
```

---

### document_translation.py

**Exam skill:** Translate text and documents by using the Azure Translator in Foundry Tools service

Demonstrates the **asynchronous** `begin_translation()` pattern:

- Translates all documents in a source Azure Blob container to a target container
- Polls the long-running operation (LRO) and prints per-document results
- Lists all supported document formats (DOCX, PDF, HTML, XLSX, PPTX, TXT, …)

Requires `AZURE_SOURCE_SAS_URL` and `AZURE_TARGET_SAS_URL` to run the translation job; format listing works without them.

```bash
python document_translation.py
```

---

### speech_to_text.py

**Exam skill:** Implement text-to-speech and speech-to-text using Azure Speech in Foundry Tools

Covers two recognition patterns:

1. **One-shot** (`recognize_once_async`) – returns after the first utterance; includes confidence score and word-level timing from detailed JSON output
2. **Continuous** – event-driven callbacks (`recognizing`, `recognized`, `session_stopped`) for long recordings

Creates a silent test WAV if no real audio file is provided (set `SPEECH_AUDIO_FILE` for actual results).

```bash
python speech_to_text.py
```

---

### text_to_speech.py

**Exam skill:** Implement text-to-speech and speech-to-text using Azure Speech in Foundry Tools

Shows:

- Synthesising speech to WAV files with `speak_text_async()`
- English voices: `en-US-JennyNeural`, `en-GB-RyanNeural`
- Multilingual voices: French (`fr-FR-DeniseNeural`), Spanish (`es-ES-ElviraNeural`)
- Listing available voices filtered by locale prefix
- Checking `SynthesizingAudioCompleted` result reason

```bash
python text_to_speech.py
# Output WAV files: output_jenny.wav, output_ryan.wav, output_denise.wav, output_elvira.wav
```

---

### ssml_speech.py

**Exam skill:** Improve text-to-speech by using Speech Synthesis Markup Language (SSML)

Uses `speak_ssml_async()` with six SSML examples saved to separate WAV files:

| File | SSML Feature |
|------|-------------|
| `ssml_prosody.wav` | `<prosody rate pitch volume>` |
| `ssml_breaks.wav` | `<break time strength>` |
| `ssml_emphasis.wav` | `<emphasis level>` |
| `ssml_say_as.wav` | `<say-as interpret-as>` (characters, telephone, date, ordinal) |
| `ssml_style.wav` | `<mstts:express-as style>` (cheerful, sad, customerservice) |
| `ssml_multilang.wav` | Multiple `<voice>` elements switching language mid-document |

```bash
python ssml_speech.py
```

---

### speech_translation.py

**Exam skill:** Translate speech-to-speech and speech-to-text by using the Azure Speech in Foundry Tools service

Demonstrates `TranslationRecognizer`:

- **One-shot** translation: audio → translated text in French, German, and Japanese
- **Continuous** translation with:
  - Partial (`recognizing`) and final (`recognized`) callbacks
  - Speech-to-speech synthesis via the `synthesizing` event (raw PCM output saved to file)

```bash
python speech_translation.py
```

---

### intent_recognition.py

**Exam skill:** Implement intent and keyword recognition with Azure Speech in Foundry Tools

Three demos:

1. **Pattern-based intent recognition** – offline `PatternMatchingModel` with inline phrase patterns and entity slots `{device}`, `{temperature}`
2. **CLU-backed intent recognition** – `ConversationLanguageUnderstandingModel` connecting Speech SDK to a deployed CLU project
3. **Keyword recognition** – wake-word detection pattern using `KeywordRecognitionModel` (shows wiring; requires a `.table` model file from Azure Custom Keyword portal)

```bash
python intent_recognition.py
```

---

### clu_model.py

**Exam skills:** Create intents/entities/utterances · Train/evaluate/deploy/test a language model · Consume a language model from a client application

Full CLU lifecycle in 6 steps:

| Step | Operation | Client |
|------|-----------|--------|
| 1 | Create project | `ConversationAuthoringClient` |
| 2 | Import intents, entities, utterances | `ConversationAuthoringClient` |
| 3 | Train (`begin_train`) | `ConversationAuthoringClient` |
| 4 | Evaluate (precision / recall / F1) | `ConversationAuthoringClient` |
| 5 | Deploy to named slot | `ConversationAuthoringClient` |
| 6 | Query deployed model | `ConversationAnalysisClient` |

Domain: **Home Automation** – intents `TurnOn`, `TurnOff`, `SetTemperature`, `CheckStatus`, `None`

```bash
python clu_model.py
```

---

### custom_question_answering.py

**Exam skills:** Create a CQA project · Add QnA pairs · Train/test/publish KB · Multi-turn conversation · Alternate phrasing and chit-chat · Export KB

Full CQA lifecycle in 7 steps:

| Step | Operation |
|------|-----------|
| 1 | Create project |
| 2 | Add QnA pairs with alternate phrasings and multi-turn follow-up prompts |
| 3 | Import content from a URL |
| 4 | Add chit-chat (personality) |
| 5 | Deploy knowledge base |
| 6 | Query with confidence thresholding |
| 7 | Export KB to JSON |

```bash
python custom_question_answering.py
```

---

### custom_translator.py

**Exam skill:** Implement custom translation including training, improving, and publishing a custom model

Shows the complete Custom Translator workflow:

1. Explains the management REST API calls (create workspace → project → upload docs → train → publish)
2. Saves sample parallel training data files (`train_parallel.en.txt`, `train_parallel.fr.txt`)
3. Translates text using a custom model via `category=<id>` parameter
4. Compares generic model vs custom model output side-by-side

Set `CUSTOM_TRANSLATOR_CATEGORY_ID` in `.env` after training and publishing your model to run the live translation steps.

```bash
python custom_translator.py
```

---

## Required Environment Variables Summary

| Variable | Used By |
|----------|---------|
| `AZURE_LANGUAGE_ENDPOINT` | key_phrases_entities, sentiment_analysis, language_detection, pii_detection, clu_model, custom_question_answering |
| `AZURE_LANGUAGE_KEY` | same as above |
| `AZURE_SPEECH_KEY` | speech_to_text, text_to_speech, ssml_speech, speech_translation, intent_recognition |
| `AZURE_SPEECH_REGION` | same as above |
| `AZURE_TRANSLATOR_KEY` | text_translation, document_translation, custom_translator |
| `AZURE_TRANSLATOR_ENDPOINT` | text_translation, document_translation |
| `AZURE_TRANSLATOR_REGION` | text_translation, custom_translator |
| `AZURE_SOURCE_SAS_URL` | document_translation |
| `AZURE_TARGET_SAS_URL` | document_translation |
| `CLU_PROJECT_NAME` | intent_recognition (CLU demo), clu_model |
| `CLU_DEPLOYMENT_NAME` | intent_recognition (CLU demo), clu_model |
| `CUSTOM_TRANSLATOR_CATEGORY_ID` | custom_translator |
| `SPEECH_AUDIO_FILE` | speech_to_text, speech_translation, intent_recognition *(optional)* |

---

## Key Exam Tips

- **Sentiment labels** are `Positive`, `Negative`, `Neutral`, and `Mixed` (not "middle" or "moderate").
- **Opinion mining** = aspect-based sentiment; enabled by `show_opinion_mining=True`.
- **PII redacted text** is returned automatically; always use it in logs, never raw PII.
- **Transliteration ≠ Translation** – script changes, language stays the same.
- **Document Translation** uses a *resource-level* endpoint (not the global Translator endpoint).
- **CLU replaces LUIS** for intent recognition; LUIS was retired September 2025.
- **SSML** requires `speak_ssml_async()` not `speak_text_async()`.
- **Keyword recognition** runs *on-device*; intent recognition goes to the cloud.
- **Custom Translator** category ID is passed as `?category=<id>` to the standard `/translate` endpoint.
- **CQA** does not have a separate training step; deploying the project makes it live.
