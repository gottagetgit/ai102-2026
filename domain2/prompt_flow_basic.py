"""
prompt_flow_basic.py
====================
Demonstrates the core concepts of Azure AI Foundry Prompt flow.

Exam Skill: "Use Azure AI Foundry Prompt flow"
            (Domain 2 - Implement generative AI solutions)

What this file covers:
  - What Prompt flow is and the key concepts (flow, node, variant, run)
  - How to create a simple chat flow using the promptflow SDK
  - How to run a flow locally and inspect outputs
  - How to evaluate a flow with a dataset
  - How to deploy a flow as a REST endpoint
  - How to use variants for A/B testing prompt templates

Note: This script uses the 'promptflow' and 'promptflow-azure' packages.
      If not installed, the Conceptual sections still explain the key ideas.

Required packages:
  pip install promptflow promptflow-azure promptflow-tools python-dotenv

Required environment variables:
  AZURE_OPENAI_ENDPOINT        - https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY             - API key
  AZURE_OPENAI_DEPLOYMENT      - GPT-4o deployment name
  AZURE_AI_SUBSCRIPTION_ID     - Azure subscription (for cloud runs)
  AZURE_AI_RESOURCE_GROUP      - Resource group name
  AZURE_AI_PROJECT_NAME        - AI Foundry project name
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Section 1 - Core concepts (no SDK required)
# ---------------------------------------------------------------------------

CONCEPTS = """
Prompt flow key concepts
========================

FLOW
  A directed acyclic graph (DAG) of nodes. Each node is a step that transforms
  data. The flow has defined inputs and outputs.
  Types:
    - Standard flow   : general data transformation
    - Chat flow       : conversational, tracks chat_history input
    - Evaluation flow : scores other flow outputs

NODE TYPES
  - LLM node   : calls an Azure OpenAI model with a Jinja2 prompt template
  - Python node: runs arbitrary Python code
  - Prompt node: renders a Jinja2 template (no LLM call)
  - Tool node  : uses a built-in tool (e.g., Bing search, Embedding)

VARIANT
  Multiple versions of the same node (different model, temperature, or prompt).
  Used for A/B testing.

RUN
  An execution of the flow against a dataset. Results and metrics are stored
  in the Azure AI Foundry portal.

CONNECTION
  Stores credentials (API keys, endpoints) securely. Nodes reference a
  connection by name rather than embedding credentials.

DEPLOYMENT
  A flow can be deployed as a managed online endpoint. The endpoint exposes
  a REST API that accepts the flow inputs and returns the flow outputs.
"""


def print_concepts() -> None:
    print(CONCEPTS)


# ---------------------------------------------------------------------------
# Section 2 - Minimal inline flow using the promptflow SDK
# ---------------------------------------------------------------------------

def get_inline_flow_yaml() -> str:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    return f"""$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json
environment:
  python_requirements_txt: requirements.txt

inputs:
  question:
    type: string
    default: "What is Azure AI Foundry?"
  chat_history:
    type: list
    default: []

outputs:
  answer:
    type: string
    reference: ${{nodes.llm_node.output}}

nodes:
- name: llm_node
  type: llm
  source:
    type: code
    path: chat_prompt.jinja2
  inputs:
    deployment_name: {deployment}
    max_tokens: 512
    temperature: 0.7
    question: ${{inputs.question}}
    chat_history: ${{inputs.chat_history}}
  connection: azure_openai_connection
  api: chat
"""


def get_chat_prompt_jinja2() -> str:
    return """system:
You are a helpful AI assistant that answers questions about Azure AI services.
Be concise, accurate, and cite specific Azure service names when relevant.

{% for item in chat_history %}
{{ item.role }}:
{{ item.content }}
{% endfor %}
user:
{{question}}
"""


def write_flow_files(output_dir: str = "/tmp/my_chat_flow") -> Path:
    """Write flow.dag.yaml and chat_prompt.jinja2 to a temp directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "flow.dag.yaml").write_text(get_inline_flow_yaml())
    (out / "chat_prompt.jinja2").write_text(get_chat_prompt_jinja2())
    (out / "requirements.txt").write_text("promptflow\nprompflow-tools\nopenai\n")

    print(f"Flow files written to {out}")
    return out


def demo_run_flow_local(flow_dir: Path) -> None:
    print("\n=== Running flow locally ===")
    try:
        from promptflow.client import PFClient

        pf = PFClient()

        result = pf.test(
            flow=str(flow_dir),
            inputs={"question": "What is the difference between GPT-4o and GPT-4o-mini?"},
        )
        print(f"Answer: {result}")

    except ImportError:
        print("  [SKIP] promptflow not installed. Run: pip install promptflow promptflow-tools")
    except Exception as exc:
        print(f"  [ERROR] {exc}")


def demo_batch_run(flow_dir: Path) -> None:
    print("\n=== Batch run against dataset ===")
    try:
        from promptflow.client import PFClient

        pf = PFClient()

        dataset_path = flow_dir / "test_data.jsonl"
        questions = [
            {"question": "What is Azure AI Foundry?"},
            {"question": "How do I deploy a model in Azure AI Foundry?"},
            {"question": "What is the difference between a standard and chat flow?"},
        ]
        with open(dataset_path, "w") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")

        run = pf.run(
            flow=str(flow_dir),
            data=str(dataset_path),
            column_mapping={"question": "${data.question}"},
        )
        pf.stream(run)
        details = pf.get_details(run)
        print(details.to_string())

    except ImportError:
        print("  [SKIP] promptflow not installed.")
    except Exception as exc:
        print(f"  [ERROR] {exc}")


def demo_variants() -> None:
    print("\n=== Prompt flow variants (A/B testing) ===")

    explanation = """
Variants let you test different versions of a node side-by-side.

In flow.dag.yaml you declare variants under the node:

  nodes:
  - name: summariser
    type: llm
    source:
      type: code
      path: summarise.jinja2
    inputs:
      temperature: 0.3          # variant_0 (default)
      max_tokens: 256
    variants:
      variant_1:
        inputs:
          temperature: 0.7       # more creative
          max_tokens: 512
      variant_2:
        inputs:
          temperature: 0.0       # deterministic
          max_tokens: 128

When you run a batch evaluation you can specify which variant to use:
  pf.run(flow=flow_dir, data=dataset, variant="${summariser.variant_1}")

You then compare metrics (e.g., groundedness, coherence) across variants
in the Azure AI Foundry portal to pick the best-performing template.
"""
    print(explanation)


def demo_deploy_endpoint() -> None:
    print("\n=== Deploying a flow as a REST endpoint ===")

    code_snippet = """
from promptflow.azure import PFClient
from azure.identity import DefaultAzureCredential

pf = PFClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_AI_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_AI_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_AI_PROJECT_NAME"],
)

runtime = "automatic"
endpoint_name = "my-chat-flow-endpoint"

deployment = pf.flows.deploy(
    flow="./my_chat_flow",
    name=endpoint_name,
    runtime=runtime,
    environment_variables={"AZURE_OPENAI_KEY": "<key>"},
)
print(f"Endpoint: {deployment.scoring_uri}")

import requests
response = requests.post(
    deployment.scoring_uri,
    headers={"Authorization": f"Bearer {deployment.swagger_uri}"},
    json={"question": "Hello!", "chat_history": []},
)
print(response.json())
"""
    print(code_snippet)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print_concepts()
    flow_dir = write_flow_files()
    demo_run_flow_local(flow_dir)
    demo_batch_run(flow_dir)
    demo_variants()
    demo_deploy_endpoint()
    print("\n=== Prompt flow demos complete ===")


if __name__ == "__main__":
    main()
