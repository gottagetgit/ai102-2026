# Domain 4: Implement Computer Vision Solutions (10–15%)

This directory contains Python demo scripts for **AI-102 Domain 4** — implementing
computer vision solutions with Azure AI Vision, Custom Vision, Video Indexer, and
Spatial Analysis.

---

## Files

| File | Purpose | Exam Skill |
|------|---------|------------|
| `image_analysis.py` | Full image analysis: captions, tags, objects, people, smart crops, OCR | Select visual features; detect objects; interpret responses |
| `ocr_read_text.py` | Extract printed text with block/line/word hierarchy and bounding polygons | Extract text from images using Azure Vision |
| `handwriting_recognition.py` | Extract handwritten text; show style classification and confidence scores | Convert handwritten text using Azure Vision |
| `custom_vision_train.py` | Create project, upload/tag images, train model, evaluate precision/recall/AP | Choose model type; label images; train & evaluate custom models |
| `custom_vision_predict.py` | Publish a trained iteration; run prediction on new images | Publish and consume a custom vision model |
| `video_indexer.py` | Upload video, poll indexing, extract transcript, topics, faces, keywords | Use Video Indexer to extract insights from video |
| `spatial_analysis_config.py` | Generate IoT Edge deployment manifests for people counting and zone monitoring | Detect presence and movement of people with Spatial Analysis |

---

## Key Concepts

### Azure AI Vision service tiers
| Feature | Service | SDK |
|---------|---------|-----|
| Image Analysis (captions, tags, objects) | Azure AI Vision | `azure-ai-vision-imageanalysis` |
| OCR / Read (printed + handwriting) | Azure AI Vision | `azure-ai-vision-imageanalysis` |
| Custom image classification | Custom Vision | `azure-cognitiveservices-vision-customvision` |
| Object detection | Custom Vision | `azure-cognitiveservices-vision-customvision` |
| Video insights | Video Indexer | REST API |
| Live video analytics | Spatial Analysis | IoT Edge container |

### Image Analysis visual features
```python
from azure.ai.vision.imageanalysis.models import VisualFeatures

VisualFeatures.CAPTION        # Single image description
VisualFeatures.DENSE_CAPTIONS # Per-region descriptions
VisualFeatures.TAGS           # Taxonomy tags with confidence
VisualFeatures.OBJECTS        # Detected objects with bounding boxes
VisualFeatures.PEOPLE         # Detected people with bounding boxes
VisualFeatures.SMART_CROPS    # Recommended crop regions
VisualFeatures.READ           # OCR text extraction
```

### OCR result hierarchy
```
result.read.blocks[]          → Text blocks (e.g. a paragraph)
  .lines[]                    → Individual lines of text
    .text                     → Full line text
    .bounding_polygon         → List of {x, y} corner points
    .words[]                  → Individual words
      .text                   → Word text
      .confidence             → 0.0–1.0 confidence score
      .bounding_polygon       → Word bounding box
```

### Custom Vision — Classification vs Object Detection
| Aspect | Classification | Object Detection |
|--------|---------------|-----------------|
| Output | Single label (or multi-label) per image | Multiple bounding boxes + labels per image |
| Minimum images per tag | 5 (50+ recommended) | 15 (50+ recommended) |
| API method | `classify_image_url()` | `detect_image_url()` |
| Training data | Just tagged images | Images + bounding box annotations |
| Use case | "Is this a cat or a dog?" | "Where are all the cats in this image?" |

### Custom Vision metrics
| Metric | Meaning | Target |
|--------|---------|--------|
| Precision | Of all images predicted as tag X, what % were correct? | > 85% |
| Recall | Of all images actually tag X, what % were found? | > 85% |
| AP | Area under the Precision-Recall curve | > 0.9 = excellent |

### Video Indexer insights categories
- **Transcript** — speech-to-text with timestamps
- **Topics** — detected subjects (with IAB/Wikipedia references)
- **Keywords** — significant terms throughout the video
- **Labels** — visual objects and scenes
- **Faces** — detected/identified people
- **Named entities** — people, locations, brands mentioned
- **Sentiments** — positive/negative/neutral segments
- **OCR** — text visible in video frames
- **Audio effects** — music, applause, silence, etc.

### Spatial Analysis operations
| Operation | Use Case |
|-----------|----------|
| `PersonCounting` | Count people in a zone; crowding alerts |
| `PersonCrossingLine` | Count entries/exits; track direction |
| `PersonCrossingPolygon` | Entry/exit events + dwell time |
| `PersonDistance` | Proximity alerts between people |

---

## Required Environment Variables

Create a `.env` file with the variables below. Not all scripts need all variables.

```env
# ── Azure AI Vision (image_analysis, ocr_read_text, handwriting_recognition) ──
AZURE_AI_SERVICES_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_AI_SERVICES_KEY=<your-key>

# ── Custom Vision Training (custom_vision_train, custom_vision_predict) ──
CUSTOM_VISION_TRAINING_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
CUSTOM_VISION_TRAINING_KEY=<your-training-key>
CUSTOM_VISION_PROJECT_NAME=AI102-Demo-Project

# ── Custom Vision Prediction (custom_vision_predict) ──
CUSTOM_VISION_PREDICTION_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
CUSTOM_VISION_PREDICTION_KEY=<your-prediction-key>
CUSTOM_VISION_PREDICTION_RESOURCE_ID=/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<pred-resource>
CUSTOM_VISION_PROJECT_ID=<project-id-from-training>
CUSTOM_VISION_ITERATION_ID=<iteration-id-from-training>
CUSTOM_VISION_PUBLISH_NAME=ProductionModel

# ── Azure AI Video Indexer (video_indexer) ──
VIDEO_INDEXER_ACCOUNT_ID=<your-account-guid>
VIDEO_INDEXER_LOCATION=trial   # or azure region e.g. "eastus"
VIDEO_INDEXER_API_KEY=<key-from-api-portal.videoindexer.ai>

# ── Spatial Analysis (spatial_analysis_config — no live keys needed for config generation) ──
# The deployment manifest references ${AZURE_AI_SERVICES_ENDPOINT} and ${AZURE_AI_SERVICES_KEY}
# as environment variables injected into the IoT Edge container at runtime.
```

---

## Installation

```bash
# Core Vision packages
pip install azure-ai-vision-imageanalysis python-dotenv

# Custom Vision
pip install azure-cognitiveservices-vision-customvision

# Video Indexer (uses requests only)
pip install requests
```

---

## Running the Scripts

```bash
# Image analysis (all visual features from URL)
python image_analysis.py

# OCR — printed text extraction
python ocr_read_text.py

# Handwriting recognition with confidence analysis
python handwriting_recognition.py

# Custom Vision: create project, upload images, train, evaluate
python custom_vision_train.py

# Custom Vision: publish trained model and classify images
python custom_vision_predict.py

# Video Indexer: upload, index, extract insights
python video_indexer.py

# Spatial Analysis: generate and validate deployment configurations
python spatial_analysis_config.py   # No API keys needed
```

---

## Exam Tips

1. **VisualFeatures selection**: Only request the features you need — each extra feature slightly increases cost and latency. The exam may ask you to identify the minimum features needed for a task.

2. **OCR vs Image Analysis READ**: Both use the same `VisualFeatures.READ` in the newer SDK. The older "Cognitive Services Computer Vision" API had separate `/ocr` and `/read` endpoints — know both for the exam.

3. **Custom Vision project types**: You must decide at project creation time whether it is Classification or Object Detection. You cannot change it later.

4. **Minimum training images**: Custom Vision requires at least **5 images per tag** to train; more is always better. Exam questions often test this boundary condition.

5. **Video Indexer auth**: Uses its own token system (not Azure AD by default). The developer portal at `api-portal.videoindexer.ai` issues subscription keys used to obtain per-account access tokens.

6. **Spatial Analysis prerequisites**: Requires an Azure Stack Edge Pro or a machine with a GPU; runs as an IoT Edge module. Understand the camera source → operation → sink pipeline architecture.

7. **Bounding box formats**: Custom Vision object detection returns **normalised** coordinates (0–1); Azure AI Vision returns **pixel** coordinates. Know which is which.

8. **Publishing vs training**: A Custom Vision *iteration* must be explicitly **published** before the prediction endpoint can use it. Unpublish it to stop prediction billing while retaining the model.
