# Domain 4: Implement Computer Vision Solutions

This domain covers **15–20%** of the AI-102 exam. You will be assessed on your ability to build, train, and deploy computer vision models using Azure AI Vision services.

---

## Exam Skills Measured

### 4.1 Analyze images
- Analyze images by using Azure AI Vision
- Generate a thumbnail image by using Azure AI Vision (smart cropping)
- Extract text from images by using Azure AI Vision (OCR / Read API)
- **Identify human faces** using Azure AI Vision or Azure AI Face service

### 4.2 Implement image classification
- Train a custom image classification model by using Azure Custom Vision
- Evaluate and validate the model
- Publish/export the Custom Vision model
- **Consume** a Custom Vision classification endpoint in code

### 4.3 Implement object detection
- Train a custom object detection model by using Azure Custom Vision
- Evaluate object detection model performance
- Consume an object detection endpoint in code

### 4.4 Detect, analyze, and recognize faces
- Detect faces using Azure AI Vision (basic)
- Detect, analyze, and recognize faces using **Azure AI Face service**
- Implement **face liveness detection**
- Compare and match faces (**face verification / identification**)

### 4.5 Read text in images and documents
- Use Azure AI Vision **Read API** for multi-page PDFs and images
- Use **Azure AI Document Intelligence** for forms, invoices, receipts, ID documents
- Build and train **custom extraction models** in Document Intelligence

---

## Files in this Domain

| File | Skills Covered |
|---|---|
| `image_analysis.py` | 4.1 – Analyze image, describe, tag, objects, smart crop |
| `ocr_read_text.py` | 4.1 / 4.5 – OCR Read API, extract text from images |
| `handwriting_recognition.py` | 4.5 – Handwriting via Read API + Document Intelligence |
| `custom_vision_train.py` | 4.2 / 4.3 – Train classification & detection models |
| `custom_vision_predict.py` | 4.2 / 4.3 – Consume Custom Vision prediction endpoints |
| `face_detection.py` | 4.4 – Detect and analyze faces with Azure AI Face |
| `document_intelligence.py` | 4.5 – Forms, invoices, receipts with Document Intelligence |

---

## Key Services and SDKs

### Azure AI Vision (Cognitive Services – Computer Vision)
- SDK: `azure-ai-vision-imageanalysis` (v1.x, `ImageAnalysisClient`)
- Capabilities: image description, tagging, object detection, OCR, smart crop, background removal
- Endpoint: `https://<resource>.cognitiveservices.azure.com/`

### Azure Custom Vision
- SDK: `azure-cognitiveservices-vision-customvision` (training + prediction clients)
- Two project types: **Classification** (multi-class / multi-label) and **Object Detection**
- Endpoints: Training endpoint + Prediction endpoint (separate)
- Export formats: CoreML, TensorFlow, ONNX, DockerFile

### Azure AI Face Service
- SDK: `azure-ai-vision-face`
- Operations: detect, verify, identify, find-similar, group, liveness check
- Attributes: age estimate, emotion, head pose, mask detection, blur, exposure

### Azure AI Document Intelligence
- SDK: `azure-ai-documentintelligence`
- Prebuilt models: `prebuilt-read`, `prebuilt-layout`, `prebuilt-invoice`, `prebuilt-receipt`, `prebuilt-idDocument`, `prebuilt-businessCard`
- Custom models: composed models, neural models

---

## Environment Variables Required

```
# Azure AI Vision
AZURE_VISION_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_VISION_KEY=<key>

# Azure Custom Vision – Training
CUSTOM_VISION_TRAINING_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
CUSTOM_VISION_TRAINING_KEY=<key>

# Azure Custom Vision – Prediction
CUSTOM_VISION_PREDICTION_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
CUSTOM_VISION_PREDICTION_KEY=<key>
CUSTOM_VISION_PREDICTION_RESOURCE_ID=/subscriptions/.../resourceGroups/.../providers/Microsoft.CognitiveServices/accounts/<name>

# Azure AI Face
AZURE_FACE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_FACE_KEY=<key>

# Azure AI Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=<key>
```

---

## Quick Revision Notes

### Image Analysis v4 (Florence model)
- `ImageAnalysisClient` from `azure.ai.vision.imageanalysis`
- `analyze()` accepts `VisualFeatures` enum: `CAPTION`, `DENSE_CAPTIONS`, `TAGS`, `OBJECTS`, `PEOPLE`, `SMART_CROPS`, `READ`
- Smart cropping requires specifying aspect ratios
- New in v4: **background removal** (`segment()`), **dense captions**

### OCR / Read API
- Use `VisualFeatures.READ` with `ImageAnalysisClient` for images
- For multi-page documents use `DocumentAnalysisClient` with `prebuilt-read`
- Results: pages → lines → words with bounding polygons and confidence

### Custom Vision workflow
1. Create project (classification or object detection)
2. Create tags
3. Upload images with tags (or regions for object detection)
4. Train iteration → evaluate precision/recall
5. Publish iteration to prediction resource
6. Call prediction endpoint via SDK

### Face Service key facts
- `DetectFromUrl` / `DetectFromStream` → returns `FaceAttributeType` data
- `Verify` = same person? (1:1)
- `Identify` = who is this? (1:N, requires `PersonGroup`)
- Liveness check requires Azure Face Liveness session API
- **Limited Access policy**: some features (identification, verification) require approval

### Document Intelligence key facts
- Async pattern: `begin_analyze_document()` returns a poller
- Access results: `result.documents[0].fields["InvoiceTotal"].value`
- `prebuilt-layout` extracts tables, paragraphs, selection marks
- Custom models trained in Document Intelligence Studio
