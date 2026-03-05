# Domain 1: Plan and Manage an Azure AI Solution

This directory contains Python demos covering **Domain 1 (20–25%)** of the AI-102 exam.

---

## Files Overview

| File | Description | Exam Skill |
|------|-------------|------------|
| `create_ai_resource.py` | Create, list, and inspect Azure AI Services resources using the Management SDK | Create an Azure AI resource |
| `manage_keys.py` | List and rotate API keys; zero-downtime key rotation pattern; Key Vault guidance | Manage and protect account keys |
| `monitor_resource.py` | Query Azure Monitor metrics: call counts, latency, errors, token usage | Monitor an Azure AI resource |
| `content_safety.py` | Analyze text and images across all four harm categories (hate, violence, self-harm, sexual) | Implement content moderation; Configure responsible AI |
| `content_filters_blocklists.py` | Create custom blocklists, add terms/regex patterns, analyze text against blocklist | Implement responsible AI — content filters and blocklists |
| `prompt_shields.py` | Detect jailbreak attacks and indirect prompt injection; integrate shields into RAG pipelines | Prevent harmful behavior — prompt shields and harm detection |
| `authenticate_entra.py` | API key vs. Entra ID (keyless) authentication; DefaultAzureCredential chain; managed identity | Manage authentication for a Microsoft Foundry Service resource |
| `container_deployment.py` | Pull and run Azure AI Services containers locally; call local endpoint; docker run reference | Plan and implement a container deployment |

---

## Setup

### 1. Install dependencies

```bash
pip install \
  azure-identity \
  azure-mgmt-cognitiveservices \
  azure-mgmt-monitor \
  azure-ai-contentsafety \
  azure-ai-textanalytics \
  openai \
  requests \
  python-dotenv
```

### 2. Create a `.env` file

Copy the template below and fill in your values:

```env
# Azure subscription / resource group
AZURE_SUBSCRIPTION_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
AZURE_RESOURCE_GROUP=my-ai-resource-group
AZURE_LOCATION=eastus
AZURE_AI_ACCOUNT_NAME=my-ai-services-account

# Azure AI Services (multi-service resource)
AZURE_AI_SERVICES_ENDPOINT=https://my-ai-services.cognitiveservices.azure.com/
AZURE_AI_SERVICES_KEY=your-ai-services-key

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://my-openai.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Service principal (for service principal auth demo only)
# AZURE_TENANT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# AZURE_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# AZURE_CLIENT_SECRET=your-client-secret
```

### 3. Azure authentication for management scripts

Management SDK scripts (`create_ai_resource.py`, `manage_keys.py`, `monitor_resource.py`) require:
- Azure CLI login: `az login`
- Contributor or Cognitive Services Contributor role on the subscription/resource group

---

## Required Environment Variables per Script

| Script | Required Variables |
|--------|-------------------|
| `create_ai_resource.py` | `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_AI_ACCOUNT_NAME`, `AZURE_LOCATION` |
| `manage_keys.py` | `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_AI_ACCOUNT_NAME` |
| `monitor_resource.py` | `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_AI_ACCOUNT_NAME` |
| `content_safety.py` | `AZURE_AI_SERVICES_ENDPOINT`, `AZURE_AI_SERVICES_KEY` |
| `content_filters_blocklists.py` | `AZURE_AI_SERVICES_ENDPOINT`, `AZURE_AI_SERVICES_KEY` |
| `prompt_shields.py` | `AZURE_AI_SERVICES_ENDPOINT`, `AZURE_AI_SERVICES_KEY` |
| `authenticate_entra.py` | `AZURE_AI_SERVICES_ENDPOINT`, `AZURE_AI_SERVICES_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| `container_deployment.py` | `AZURE_AI_SERVICES_ENDPOINT`, `AZURE_AI_SERVICES_KEY` |

---

## Key Concepts Covered

### Resource Management
- Creating multi-service AI resources (`CognitiveServices` kind) vs. single-service (`OpenAI`, `TextAnalytics`, etc.)
- SKU tiers: F0 (free, limited), S0 (standard, full features)
- Resource properties: endpoint, provisioning state, capabilities

### Key Management
- Two-key rotation pattern for zero-downtime key rotation
- Azure Key Vault integration for secure key storage
- `disable_local_auth=True` to enforce Entra ID only

### Monitoring
- Azure Monitor metrics: `TotalCalls`, `SuccessfulCalls`, `TotalErrors`, `BlockedCalls`, `LatencyE2E`, `TokenTransaction`
- Alert rules for error spikes and quota exhaustion
- Time series querying with granularity and aggregation

### Content Safety
- Harm categories: **Hate**, **Violence**, **SelfHarm**, **Sexual**
- Severity scale: 0 (safe) → 2 (low) → 4 (medium) → 6 (high)
- Custom blocklists with exact string and regex pattern matching
- Prompt shields for jailbreak and indirect prompt injection

### Authentication
- `AzureKeyCredential` — API key, simple but requires key management
- `DefaultAzureCredential` — tries multiple sources, recommended for all environments
- `ManagedIdentityCredential` — for Azure-hosted apps (best practice)
- `ClientSecretCredential` — for service principals in CI/CD
- RBAC roles: `Cognitive Services User`, `Cognitive Services Contributor`, `Cognitive Services OpenAI User`

### Container Deployment
- All containers require `Eula=accept`, `Billing=<endpoint>`, `ApiKey=<key>`
- Containers process requests locally but send billing telemetry to Azure
- Health check: `GET /status`
- Same SDK and REST API surface as cloud endpoint
