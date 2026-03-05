# AI-102 Exam Prep — Python Demo Repository

Practical Python scripts demonstrating real Azure AI service calls for the **AI-102: Designing and Implementing a Microsoft Azure AI Solution** certification exam.

## Repository Structure

```
ai102-repo/
├── domain1/          # Domain 1: Plan and manage an Azure AI solution (20–25%)
│   ├── README.md
│   ├── create_ai_resource.py
│   ├── manage_keys.py
│   ├── monitor_resource.py
│   ├── content_safety.py
│   ├── content_filters_blocklists.py
│   ├── prompt_shields.py
│   ├── authenticate_entra.py
│   └── container_deployment.py
│
├── domain2/          # Domain 2: Implement generative AI solutions (15–20%)
│   ├── README.md
│   ├── chat_completions.py
│   ├── prompt_engineering.py
│   ├── generate_images_dalle.py
│   ├── multimodal_vision.py
│   ├── rag_pattern.py
│   ├── prompt_flow_basic.py
│   ├── model_parameters.py
│   ├── fine_tuning.py
│   ├── orchestrate_models.py
│   └── prompt_templates.py
│
└── .env.example      # Environment variable template
```

## Quick Start

### 1. Clone and set up

```bash
git clone <repo-url>
cd ai102-repo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
# Core packages for all demos
pip install openai azure-identity azure-ai-contentsafety azure-search-documents \
  azure-ai-textanalytics azure-mgmt-cognitiveservices azure-mgmt-monitor \
  jinja2 requests pillow promptflow promptflow-tools python-dotenv
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your Azure resource credentials
```

### 4. Authenticate (for management scripts)

```bash
az login
```

### 5. Run any demo

```bash
python domain2/chat_completions.py
python domain1/content_safety.py
```

## AI-102 Exam Domains

| Domain | Weight | Coverage |
|--------|--------|----------|
| [Domain 1: Plan and manage an Azure AI solution](./domain1/) | 20–25% | Resource management, keys, monitoring, content safety, authentication, containers |
| [Domain 2: Implement generative AI solutions](./domain2/) | 15–20% | Chat completions, prompt engineering, DALL-E, vision, RAG, fine-tuning, parameters |
| Domain 3: Implement AI agents | 5–10% | *(coming soon)* |
| Domain 4: Implement computer vision | 15–20% | *(coming soon)* |
| Domain 5: Implement natural language processing | 15–20% | *(coming soon)* |
| Domain 6: Implement document and speech AI | 10–15% | *(coming soon)* |
| Domain 7: Implement knowledge mining and AI search | 15–20% | *(coming soon)* |

## Conventions

All scripts follow these patterns:

- **`python-dotenv`** for credentials: `from dotenv import load_dotenv; load_dotenv()`
- **Environment variables** for all keys and endpoints (never hardcoded)
- **Try/except** for proper error handling with helpful messages
- **Docstrings** mapping each file to the specific AI-102 exam skill
- **Latest Azure SDK packages** (`azure-ai-*`, `openai>=1.40`)
