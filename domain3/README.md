# Domain 3: Implement an Agentic Solution (5–10%)

This directory contains Python demo scripts for **AI-102 Domain 3** — building agents
with the Azure AI Foundry Agent Service and Semantic Kernel.

---

## Files

| File | Purpose | Exam Skill |
|------|---------|------------|
| `foundry_agent_basic.py` | Create an agent, thread, run, and retrieve response | Create an agent with the Microsoft Foundry Agent Service |
| `agent_with_code_interpreter.py` | Attach Code Interpreter; upload CSV for data analysis | Configure the necessary resources to build an agent |
| `agent_with_file_search.py` | Upload docs, create vector store, enable RAG-based retrieval | Create an agent with the Microsoft Foundry Agent Service |
| `agent_with_functions.py` | Define custom function tools; handle `requires_action` run state | Implement complex agents with Microsoft Agent Framework |
| `multi_agent_orchestration.py` | Researcher + Writer multi-agent workflow with Semantic Kernel | Implement complex workflows including orchestration for a multi-agent solution |
| `agent_evaluation.py` | Test agent against structured test cases; LLM-as-judge scoring | Test, optimize and deploy an agent |

---

## Key Concepts

### Agent lifecycle
```
Create Agent → Create Thread → Add Message → Create Run → Poll Run → Retrieve Messages
```

### Run states
| State | Meaning |
|-------|---------|
| `queued` | Run is waiting to be processed |
| `in_progress` | Model is generating a response |
| `requires_action` | Agent wants to call a tool — you must submit results |
| `completed` | Run finished successfully |
| `failed` | Run encountered an error |
| `cancelled` / `expired` | Run was cancelled or timed out |

### Tools overview
| Tool | Use Case |
|------|----------|
| **Code Interpreter** | Run Python code; analyse files, do maths, generate charts |
| **File Search** | RAG over uploaded documents via managed vector store |
| **Function Tools** | Call your own code/APIs; agent asks, host executes |

### Multi-agent patterns (Semantic Kernel)
- `ChatCompletionAgent` — wraps an LLM with a persona and instructions
- `AgentGroupChat` — orchestrates message routing between agents
- `KernelFunctionSelectionStrategy` — decides which agent speaks next
- `KernelFunctionTerminationStrategy` — detects when to stop the chat

---

## Required Environment Variables

Create a `.env` file in this directory (or the repo root) with:

```env
# Azure AI Foundry project connection string
# Found in: AI Foundry Portal > Your Project > Settings > Overview
AZURE_AI_PROJECT_CONNECTION_STRING=<region>.api.azureml.ms;<subscription>;<resource-group>;<workspace>

# Azure OpenAI deployment name (the model deployment, e.g. gpt-4o)
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# For agent_evaluation.py and multi_agent_orchestration.py — direct OpenAI access
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-api-key>

# Optional: separate judge model for evaluation
AZURE_OPENAI_JUDGE_DEPLOYMENT=gpt-4o
```

---

## Installation

```bash
pip install azure-ai-projects azure-identity semantic-kernel openai python-dotenv
```

Minimum versions:
- `azure-ai-projects >= 1.0.0`
- `semantic-kernel >= 1.3.0` (for AgentGroupChat)
- `openai >= 1.0.0`

---

## Running the Scripts

```bash
# Basic agent (single-turn + multi-turn)
python foundry_agent_basic.py

# Agent with Code Interpreter (CSV analysis)
python agent_with_code_interpreter.py

# Agent with File Search (document Q&A)
python agent_with_file_search.py

# Agent with function tools (weather + currency)
python agent_with_functions.py

# Multi-agent researcher + writer workflow
python multi_agent_orchestration.py

# Agent evaluation report
python agent_evaluation.py
```

---

## Exam Tips

1. **Thread vs Run**: A *thread* holds the conversation history; a *run* is a single execution of the agent against that thread. You can have many runs on one thread.

2. **`requires_action` state**: When a run enters this state, the agent has chosen to call a function tool. You **must** call `submit_tool_outputs_to_run` or the run will expire.

3. **`create_and_process_run` vs manual polling**: `create_and_process_run` is a convenience method that blocks and handles tool calls automatically. Use manual polling (as in `agent_with_functions.py`) when you need control over each step.

4. **Vector stores**: File Search requires a *vector store* — a managed index that chunks and embeds your documents. Vector stores are billed separately from file storage.

5. **Responsible AI**: All Azure AI agent calls pass through content filters. You can configure filter severity levels per deployment.
