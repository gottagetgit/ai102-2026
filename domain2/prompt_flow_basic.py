"""
prompt_flow_basic.py
====================
Demonstrates using the PromptFlow SDK to build and run a simple LLM-powered flow.
Shows how to create flows programmatically and run them locally.

Exam Skill: "Implement a prompt flow solution"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Creating a basic prompt flow with an LLM node in YAML
  - Running a flow locally using the promptflow SDK
  - Understanding flow anatomy: inputs, outputs, nodes
  - Creating a reusable chat flow
  - Key promptflow concepts: DAGs, variants, evaluations

PromptFlow architecture:
  - A Flow is a DAG (Directed Acyclic Graph) of nodes
  - Node types: LLM, Python, Prompt, Tool (web search, etc.)
  - Flows are defined in flow.dag.yaml files
  - Each node has inputs and outputs; outputs can be used by downstream nodes
  - Flows can be deployed to Azure AI Foundry as managed online endpoints

File structure created by this script:
  ./flows/
    basic_chat/
      flow.dag.yaml      - Flow definition
      chat.jinja2        - Prompt template
      requirements.txt   - Python dependencies

Required packages:
  pip install promptflow promptflow-tools openai python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT    - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY         - API key
  AZURE_OPENAI_DEPLOYMENT  - Deployment name e.g. "gpt-4o"
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY      = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT      = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

FLOW_DIR = Path("./flows/basic_chat")


# ---------------------------------------------------------------------------
# Flow file creation helpers
# ---------------------------------------------------------------------------

def create_flow_files() -> None:
    """
    Create the prompt flow directory and required files:
      - flow.dag.yaml    : Flow definition (nodes, connections, inputs/outputs)
      - chat.jinja2      : Jinja2 prompt template with variable substitution
      - requirements.txt : SDK dependencies
    """
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[SETUP] Creating flow files in {FLOW_DIR.absolute()}")

    # 1. Prompt template (Jinja2 format)
    # Variables in {{ }} are substituted at runtime from node inputs
    prompt_template = """system:
You are a knowledgeable AI assistant specializing in Azure AI services.
Answer questions accurately and concisely.
Always mention the relevant Azure service name when applicable.

user:
{{question}}
"""

    (FLOW_DIR / "chat.jinja2").write_text(prompt_template)
    print("  Created: chat.jinja2")

    # 2. Flow DAG definition
    # This YAML defines the nodes and how data flows between them
    flow_dag = f"""# flow.dag.yaml - Basic chat flow
# This flow takes a user question and returns an AI-generated answer
# using Azure OpenAI GPT model.

id: basic_chat_flow
name: Basic Chat Flow
description: >
  A simple single-node flow that answers questions about Azure AI services.
  Demonstrates the fundamental structure of a PromptFlow DAG.

# Flow-level inputs (what the user provides)
inputs:
  question:
    type: string
    description: "The user's question"
    default: "What is Azure AI Content Safety?"

# Flow-level outputs (what the flow returns)
outputs:
  answer:
    type: string
    reference: ${{nodes.llm_node.output}}
  
# Nodes - each node is a processing step
nodes:
  # A single LLM node that calls Azure OpenAI
  - name: llm_node
    type: llm
    source:
      type: code
      path: chat.jinja2           # Jinja2 prompt template
    inputs:
      deployment_name: {DEPLOYMENT}
      max_tokens: 300
      temperature: 0.7
      question: ${{inputs.question}}  # Reference to flow input
    connection: azure_openai_conn   # Named connection (configured below)
    api: chat                       # Use the chat completions API
"""

    (FLOW_DIR / "flow.dag.yaml").write_text(flow_dag)
    print("  Created: flow.dag.yaml")

    # 3. Requirements
    requirements = """promptflow>=1.10.0
promptflow-tools>=1.4.0
openai>=1.40.0
"""
    (FLOW_DIR / "requirements.txt").write_text(requirements)
    print("  Created: requirements.txt")

    print(f"\n  Flow files created in: {FLOW_DIR.absolute()}")


def show_flow_structure() -> None:
    """Display the created flow files for educational purposes."""
    print("\n[FLOW STRUCTURE]")
    for f in sorted(FLOW_DIR.rglob("*")):
        if f.is_file():
            print(f"\n  === {f.name} ===")
            print(f.read_text())


# ---------------------------------------------------------------------------
# Running flows - two approaches
# ---------------------------------------------------------------------------

def run_flow_with_sdk(question: str) -> str:
    """
    Run the prompt flow programmatically using the PromptFlow SDK.

    The PFClient is the main entry point for:
      - Running flows locally (test_flow, run)
      - Bulk testing with datasets
      - Evaluating flow outputs
      - Deploying flows to Azure
    """
    print(f"\n[RUN FLOW SDK] Question: '{question}'")

    try:
        from promptflow.client import PFClient
        from promptflow.entities import AzureOpenAIConnection

        pf = PFClient()

        # Create a connection to Azure OpenAI
        # In practice, connections are configured once via CLI or portal
        connection = AzureOpenAIConnection(
            name="azure_openai_conn",
            api_key=OPENAI_KEY,
            api_base=OPENAI_ENDPOINT,
            api_type="azure",
            api_version="2024-12-01-preview",
        )

        # Register the connection (local test only)
        try:
            pf.connections.create_or_update(connection)
        except Exception:
            pass  # Connection may already exist

        # Run the flow with a single input
        result = pf.test(
            flow=str(FLOW_DIR),
            inputs={"question": question},
        )

        answer = result.get("answer", str(result))
        print(f"  Answer: {answer}")
        return answer

    except ImportError:
        print("  [INFO] promptflow not installed. Run: pip install promptflow promptflow-tools")
        return ""
    except Exception as e:
        print(f"  [INFO] Flow SDK run skipped: {e}")
        print("  This may require setting up connections via 'pf connection create' first.")
        return ""


def run_flow_manually(question: str) -> str:
    """
    Simulate what a prompt flow does manually - without the promptflow SDK.
    This shows the underlying mechanics for exam understanding.

    The flow:
      1. Substitute inputs into the Jinja2 prompt template
      2. Call the LLM with the rendered prompt
      3. Return the output
    """
    print(f"\n[RUN FLOW MANUAL] Simulating flow execution for: '{question}'")

    import openai
    from jinja2 import Template

    # Step 1: Load and render the prompt template
    template_text = (FLOW_DIR / "chat.jinja2").read_text()
    template      = Template(template_text)
    rendered       = template.render(question=question)

    print(f"  Rendered prompt:\n{rendered}")

    # Step 2: Parse rendered prompt into system/user messages
    # Jinja2 templates in PromptFlow use "system:", "user:", "assistant:" sections
    messages = []
    current_role = None
    current_content = []

    for line in rendered.strip().splitlines():
        if line.strip().lower() in ("system:", "user:", "assistant:"):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip(),
                })
            current_role = line.strip().lower().rstrip(":")
            current_content = []
        else:
            current_content.append(line)

    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip(),
        })

    print(f"  Parsed messages: {[m['role'] for m in messages]}")

    # Step 3: Call the LLM
    client = openai.AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version="2024-12-01-preview",
    )

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        max_tokens=300,
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    print(f"\n  LLM Answer: {answer}")
    return answer


# ---------------------------------------------------------------------------
# Batch run and evaluation concepts
# ---------------------------------------------------------------------------

def demo_batch_run_concept() -> None:
    """
    Explains batch runs and evaluation - important PromptFlow concepts.
    Batch runs test your flow against many inputs and collect outputs for analysis.
    """
    print("\n" + "=" * 60)
    print("Batch Run and Evaluation Concepts")
    print("=" * 60)
    print("""
  BATCH RUNS:
  -----------
  A batch run executes your flow against a dataset (JSONL file).
  
  Dataset format (questions.jsonl):
    {"question": "What is Azure OpenAI?"}
    {"question": "How does RAG work?"}
    {"question": "What is Content Safety?"}
  
  CLI command:
    pf run create \\
      --flow ./flows/basic_chat \\
      --data ./questions.jsonl \\
      --name ai102-batch-run-001

  EVALUATIONS:
  ------------
  After a batch run, you can evaluate output quality with an eval flow.
  Built-in evaluators available in Azure AI Foundry:
    - Groundedness   : Is the answer grounded in the provided context?
    - Relevance      : Is the answer relevant to the question?
    - Coherence      : Is the answer well-structured and logical?
    - Fluency        : Is the answer well-written?
    - Similarity     : How similar is it to a ground truth answer?
  
  CLI command:
    pf run create \\
      --flow ./evals/groundedness \\
      --data ./questions.jsonl \\
      --run ai102-batch-run-001 \\  # Reference the batch run output
      --name ai102-eval-001

  VARIANTS:
  ---------
  A variant is an alternate version of a node (e.g., different temperature,
  different prompt template). You can run multiple variants in parallel
  to A/B test prompt changes.
  
    pf run create \\
      --flow ./flows/basic_chat \\
      --data ./questions.jsonl \\
      --variant '${llm_node.variant_1}'  # Use variant 1 of llm_node
""")


def main():
    print("=" * 60)
    print("Azure AI Prompt Flow Demo")
    print("=" * 60)
    print(f"Deployment: {DEPLOYMENT}")

    try:
        # 1. Create flow files
        create_flow_files()

        # 2. Show flow structure
        show_flow_structure()

        # 3. Run using PromptFlow SDK (if installed)
        run_flow_with_sdk("What is Azure AI Content Safety and what harm categories does it detect?")

        # 4. Run manually (always works, shows mechanics)
        run_flow_manually("What Azure OpenAI models are available for embeddings?")

        # 5. Explain batch runs and evaluations
        demo_batch_run_concept()

        print(f"\n[COMPLETE] Flow files available in: {FLOW_DIR.absolute()}")
        print("  To run with CLI: pf flow test --flow ./flows/basic_chat --inputs question='Hello'")

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
