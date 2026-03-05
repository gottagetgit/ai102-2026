# Domain 2: Implement Generative AI Solutions

This directory contains Python demos covering **Domain 2 (15–20%)** of the AI-102 exam.

---

## Files Overview

| File | Description | Exam Skill |
|------|-------------|------------|
| `chat_completions.py` | Send single-turn and multi-turn chat completions; system prompts; streaming | Use the Azure OpenAI chat completions API |
| `prompt_engineering.py` | Zero-shot, few-shot, chain-of-thought, and system prompt techniques | Apply prompt engineering techniques |
| `generate_images_dalle.py` | Generate images with DALL-E 3; quality, style, and size parameters | Generate images with Azure OpenAI (DALL-E) |
| `multimodal_vision.py` | Analyze images with GPT-4 Vision; describe, compare, OCR with GPT | Process images with multimodal models |
| `rag_pattern.py` | Full RAG pipeline with Azure AI Search: index, embed, search, augment | Implement RAG with Azure AI Search |
| `prompt_flow_basic.py` | Create and run a PromptFlow flow; YAML DAG definition | Use Prompt Flow for LLM app orchestration |
| `model_parameters.py` | Temperature, top_p, frequency_penalty, presence_penalty, max_tokens | Configure Azure OpenAI model parameters |
| `fine_tuning.py` | Fine-tune GPT-4o mini; prepare data, upload, train, evaluate | Fine-tune an Azure OpenAI model |
| `orchestrate_models.py` | Route requests to different models; semantic routing | Orchestrate multiple AI models |
| `prompt_templates.py` | Jinja2 prompt templates; variable injection; template management | Use prompt templates |

---

## Setup

### 1. Install dependencies

```bash
pip install \
  openai \
  azure-identity \
  azure-search-documents \
  pillow \
  requests \
  jinja2 \
  promptflow \
  promptflow-tools \
  python-dotenv
```

### 2. Required environment variables

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_DALLE_DEPLOYMENT=dall-e-3

# Azure AI Search (for rag_pattern.py)
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-admin-key
AZURE_SEARCH_INDEX=ai102-rag-demo
```

---

## Script Reference

### `chat_completions.py`
Demonstrates the core Azure OpenAI chat API:
- Single-turn completion
- Multi-turn conversation with history
- System prompt configuration
- Streaming responses
- Token counting and usage tracking

### `prompt_engineering.py`
Covers all major prompt engineering techniques:
- **Zero-shot**: Direct instruction with no examples
- **Few-shot**: Provide example input/output pairs
- **Chain-of-thought (CoT)**: Instruct model to reason step-by-step
- **System prompt engineering**: Control model persona, scope, and style
- **ReAct pattern**: Reasoning + Acting for tool use
- **Output formatting**: JSON mode, structured outputs

### `generate_images_dalle.py`
- DALL-E 3 image generation
- Quality options: `standard`, `hd`
- Style options: `vivid`, `natural`
- Sizes: `1024x1024`, `1792x1024`, `1024x1792`
- Saving generated images
- Revised prompt inspection

### `multimodal_vision.py`
- GPT-4 Vision image description
- Image comparison
- OCR / text extraction from images
- Chart and diagram analysis
- URL and base64 encoded image inputs

### `rag_pattern.py`
Full end-to-end RAG implementation:
1. Create Azure AI Search index with vector field
2. Embed documents with `text-embedding-3-small`
3. Upload documents + vectors to index
4. Query-time embedding + hybrid search
5. Augment prompt with retrieved context
6. Generate answer with citation

### `model_parameters.py`
Detailed exploration of generation parameters:
- **temperature** (0–2): Randomness / creativity
- **top_p** (0–1): Nucleus sampling
- **max_tokens**: Output length limit
- **frequency_penalty** (-2 to 2): Reduce word repetition
- **presence_penalty** (-2 to 2): Encourage topic diversity
- **stop sequences**: Custom stop tokens
- **seed**: Reproducible outputs
- **n**: Multiple completions

### `fine_tuning.py`
Step-by-step fine-tuning workflow:
1. Prepare JSONL training data
2. Upload file to Azure OpenAI
3. Create fine-tuning job
4. Monitor training progress
5. Deploy fine-tuned model
6. Call fine-tuned model
7. Clean up resources

### `orchestrate_models.py`
- Simple rule-based routing
- Semantic routing (embed + cosine similarity)
- Parallel model calls with asyncio
- Model capability mapping

### `prompt_templates.py`
- Jinja2 template rendering
- Variable injection and conditionals
- Template versioning and management
- System/user/assistant template pattern

---

## Key Concepts for AI-102

### Azure OpenAI Deployments
- Each model must be **deployed** before use (not just available)
- Deployment name = what you pass to `model=` parameter
- Same model can have multiple deployments (different configs)
- Deployment types: Standard (shared), Provisioned (dedicated TPM)

### API Versions
- Always specify `api_version` in the AzureOpenAI client
- Use `2024-02-01` or later for production
- Preview features require `*-preview` versions

### Tokens and Billing
- Input tokens + output tokens = total tokens billed
- Embedding models billed on input tokens only
- DALL-E billed per image (quality + size affect cost)
- Fine-tuning: training cost + hosting cost + inference cost

### RAG vs Fine-Tuning
| | RAG | Fine-Tuning |
|--|-----|-------------|
| When | External/changing knowledge | Style/format/behavior |
| Cost | Search + inference | Training + hosting |
| Latency | Higher (retrieval step) | Same as base model |
| Data freshness | Real-time | Static (at training time) |
| Privacy | Data stays in your index | Data sent to Azure OpenAI |
