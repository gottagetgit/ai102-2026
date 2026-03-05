# Domain 2: Implement Generative AI Solutions

This directory contains Python demos covering **Domain 2 (15–20%)** of the AI-102 exam.

---

## Files Overview

| File | Description | Exam Skill |
|------|-------------|------------|
| `chat_completions.py` | Basic and multi-turn chat completions, streaming, code generation | Submit prompts to generate code and natural language responses |
| `prompt_engineering.py` | Zero-shot, few-shot, chain-of-thought, structured output, negative prompting, persona assignment, temperature effects | Apply prompt engineering techniques to improve responses |
| `generate_images_dalle.py` | DALL-E 3 image generation with size, quality, and style parameters | Use the DALL-E model to generate images |
| `multimodal_vision.py` | GPT-4o image analysis via URL and base64, chart analysis, alt-text generation, detail levels | Use large multimodal models in Azure OpenAI |
| `rag_pattern.py` | Full RAG pipeline: embed → search → generate; hybrid search; Azure OpenAI on-your-data | Implement a RAG pattern by grounding a model in your data |
| `prompt_flow_basic.py` | PromptFlow flow creation, execution, batch runs, evaluations, variants | Implement a prompt flow solution |
| `model_parameters.py` | temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop sequences, n, seed | Configure parameters to control generative behavior |
| `fine_tuning.py` | JSONL training data preparation and validation, file upload, job creation and monitoring, fine-tuned model usage | Fine-tune a generative model |
| `orchestrate_models.py` | Full 6-step pipeline: prompt shields → embedding → search → injection check → generation → output safety | Implement orchestration of multiple generative AI models |
| `prompt_templates.py` | str.format() templates, Jinja2 templates with loops/conditionals, file-based templates, versioning | Utilize prompt templates in your generative AI solution |

---

## Setup

### 1. Install dependencies

```bash
pip install \
  openai \
  azure-ai-contentsafety \
  azure-search-documents \
  azure-identity \
  jinja2 \
  requests \
  pillow \
  promptflow \
  promptflow-tools \
  python-dotenv
```

### 2. Create a `.env` file

Copy the template below and fill in your values:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_DALLE_DEPLOYMENT=dall-e-3

# Azure AI Services (for Content Safety)
AZURE_AI_SERVICES_ENDPOINT=https://your-ai-services.cognitiveservices.azure.com/
AZURE_AI_SERVICES_KEY=your-ai-services-key

# Azure AI Search (for RAG)
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-admin-key
AZURE_SEARCH_INDEX=ai102-rag-demo
```

---

## Required Environment Variables per Script

| Script | Required Variables |
|--------|-------------------|
| `chat_completions.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| `prompt_engineering.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| `generate_images_dalle.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DALLE_DEPLOYMENT` |
| `multimodal_vision.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT` (must be GPT-4o+) |
| `rag_pattern.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY` |
| `prompt_flow_basic.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| `model_parameters.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| `fine_tuning.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY` |
| `orchestrate_models.py` | All OpenAI vars + `AZURE_AI_SERVICES_ENDPOINT`, `AZURE_AI_SERVICES_KEY`, `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY` |
| `prompt_templates.py` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT` |

---

## Azure Resources Required

### 1. Azure OpenAI Resource
Create deployments for:
- **GPT-4o** (or gpt-4-turbo): For chat completions, vision, and orchestration
- **text-embedding-3-small** (or ada-002): For RAG and orchestration
- **dall-e-3**: For image generation demos

### 2. Azure AI Services Resource
Used by `content_safety.py`, `content_filters_blocklists.py`, `prompt_shields.py`, `authenticate_entra.py`, `orchestrate_models.py`.

A multi-service resource (`CognitiveServices` kind) works for all demos.

### 3. Azure AI Search Resource
Used by `rag_pattern.py` and `orchestrate_models.py`.

- Free tier (F) works for demos
- Run `rag_pattern.py` first to create and populate the index, then use `orchestrate_models.py`

---

## Key Concepts Covered

### Chat Completions
- Message roles: `system` (instructions), `user` (input), `assistant` (response)
- Multi-turn: pass full history on each call (model has no built-in memory)
- Streaming: `stream=True` returns an iterator for real-time token output
- Finish reasons: `stop` (natural end), `length` (hit max_tokens), `content_filter`

### Prompt Engineering
| Technique | When to Use |
|-----------|-------------|
| Zero-shot | Standard tasks the model knows well |
| Few-shot | Specific format/style, niche classification |
| Chain-of-thought | Math, logic, multi-step reasoning |
| Structured output | Data extraction, API responses |
| Negative prompting | Eliminating unwanted patterns |
| Persona | Domain expertise, communication style |

### DALL-E 3
- Only `n=1` supported per request
- `quality`: `standard` (faster/cheaper) or `hd` (more detail)
- `size`: `1024x1024`, `1792x1024`, `1024x1792`
- `style`: `vivid` (dramatic) or `natural` (realistic)
- `revised_prompt`: What DALL-E actually interpreted — use for iterative improvement

### Vision (GPT-4o)
- Images in `content` array with `type: image_url`
- `detail: "low"` (85 tokens, fast) vs `detail: "high"` (more tokens, more accurate for text/fine details)
- Pass URL or base64-encoded image
- Supports up to 20 images per request

### RAG Pattern
```
User Query → Embed → Vector Search → Retrieved Chunks → GPT-4o → Grounded Answer
```
- Hybrid search (keyword + vector) outperforms either alone
- Chunk size: 512-1024 tokens with 10-20% overlap
- Top-k: retrieve 3-5 chunks typically
- Always cite sources in the generated answer

### Generation Parameters
| Parameter | Range | Effect |
|-----------|-------|--------|
| `temperature` | 0.0–2.0 | Randomness (0=deterministic) |
| `max_tokens` | 1–ctx_len | Max output length |
| `top_p` | 0.0–1.0 | Vocabulary diversity (nucleus sampling) |
| `frequency_penalty` | -2.0–2.0 | Reduce word repetition |
| `presence_penalty` | -2.0–2.0 | Encourage new topics |
| `stop` | list[str] | Stop at these strings |
| `seed` | int | Reproducibility |

### Fine-Tuning
- **Supported models (2025)**: gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06, gpt-35-turbo-0125
- **Training format**: JSONL with `{"messages": [{role, content}, ...]}`
- **Minimum examples**: 10 (Azure requirement), 50+ for good results
- **Epochs**: Default `auto` (typically 3-4 for small datasets)
- **When to use**: Consistent format/style needs; not for knowledge (use RAG)

### Prompt Flow
- Flows are DAGs defined in `flow.dag.yaml`
- Node types: LLM, Python, Prompt
- Variants: A/B test different node configurations
- Evaluations: Measure groundedness, relevance, coherence, fluency, similarity
- Deploy to Azure AI Foundry as managed online endpoints
