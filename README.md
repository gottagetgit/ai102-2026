# AI-102 Exam Prep — Python Demo Repository

Practical Python scripts demonstrating real Azure AI service calls for the **AI-102: Designing and Implementing a Microsoft Azure AI Solution** certification exam (December 2025 revision).

Each domain directory contains fully working scripts that call Azure services using the latest SDKs, along with a README explaining every file.

## AI-102 Exam Domains

| Domain | Weight | Scripts | Directory |
|--------|--------|---------|-----------|
| [Domain 1: Plan and Manage an Azure AI Solution](./domain1/) | 20–25% | 8 | `domain1/` |
| [Domain 2: Implement Generative AI Solutions](./domain2/) | 15–20% | 10 | `domain2/` |
| [Domain 3: Implement an Agentic Solution](./domain3/) | 5–10% | 6 | `domain3/` |
| [Domain 4: Implement Computer Vision Solutions](./domain4/) | 10–15% | 7 | `domain4/` |
| [Domain 5: Implement Natural Language Processing Solutions](./domain5/) | 15–20% | 14 | `domain5/` |
| [Domain 6: Knowledge Mining and Information Extraction](./domain6/) | 15–20% | 8 | `domain6/` |

**53 demo scripts** covering all six exam domains.

## What's in Each Domain

### [Domain 1 — Plan and Manage an Azure AI Solution](./domain1/)
Resource creation, key management, monitoring, content safety, blocklists, prompt shields, Entra ID authentication, container deployment.

### [Domain 2 — Implement Generative AI Solutions](./domain2/)
Chat completions, prompt engineering, DALL-E image generation, multimodal vision, RAG pattern with AI Search, Prompt Flow, model parameters, fine-tuning, orchestration, prompt templates.

### [Domain 3 — Implement an Agentic Solution](./domain3/)
Azure AI Foundry agents, code interpreter tool, file search tool, function calling, multi-agent orchestration with Semantic Kernel, agent evaluation.

### [Domain 4 — Implement Computer Vision Solutions](./domain4/)
Image analysis, OCR/read text, handwriting recognition, Custom Vision training and prediction, Video Indexer, spatial analysis configuration.

### [Domain 5 — Implement Natural Language Processing Solutions](./domain5/)
Key phrase/entity extraction, sentiment analysis, language detection, PII detection, text translation, document translation, speech-to-text, text-to-speech, SSML, speech translation, intent recognition, CLU, custom question answering, custom translator.

### [Domain 6 — Knowledge Mining and Information Extraction](./domain6/)
Search index creation with skillsets, custom skills, querying, semantic and vector search, knowledge store projections, Document Intelligence (prebuilt and custom models), content understanding.

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/gottagetgit/ai102-2026.git
cd ai102-2026
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.sample .env
# Edit .env with your Azure resource keys and endpoints
```

### 4. Authenticate (for management scripts)

```bash
az login
```

### 5. Run any demo

```bash
python domain1/content_safety.py
python domain2/chat_completions.py
python domain3/foundry_agent_basic.py
python domain5/sentiment_analysis.py
```

## Conventions

All scripts follow these patterns:

- **`python-dotenv`** for credentials — `from dotenv import load_dotenv; load_dotenv()`
- **Environment variables** for all keys and endpoints (never hardcoded)
- **Try/except** with helpful error messages
- **Docstrings** mapping each file to the specific AI-102 exam skill it covers
- **Latest Azure SDK packages** (`azure-ai-*`, `openai>=1.0`)

## Resources

- [AI-102 Study Guide — Skills Measured](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-engineer/study-guide)
- [Azure AI Services Documentation](https://learn.microsoft.com/en-us/azure/ai-services/)
- [Azure OpenAI Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure AI Search Documentation](https://learn.microsoft.com/en-us/azure/search/)
