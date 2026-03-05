"""
foundry_agent_basic.py
======================
Demonstrates creating and running a basic AI agent using the Azure AI Projects SDK
(azure-ai-projects). This is the foundational pattern for all agent work on Azure.

Workflow:
    1. Connect to an Azure AI Foundry project
    2. Create an agent with a system prompt and a model
    3. Create a conversation thread
    4. Send a user message to the thread
    5. Create and poll a run until completion
    6. Retrieve and display the assistant's response
    7. Clean up (delete agent when done)

Exam Skill Mapping:
    - "Create an agent with the Microsoft Foundry Agent Service"
    - "Configure the necessary resources to build an agent"

Required Environment Variables (.env):
    AZURE_AI_PROJECT_CONNECTION_STRING  - From AI Foundry > Project settings > Overview
    AZURE_OPENAI_DEPLOYMENT             - Model deployment name (e.g. "gpt-4o")

Install:
    pip install azure-ai-projects azure-identity python-dotenv
"""

import os
import time
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

CONNECTION_STRING = os.environ.get("AZURE_AI_PROJECT_CONNECTION_STRING")
MODEL_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def create_client() -> AIProjectClient:
    """Instantiate an AIProjectClient using DefaultAzureCredential.

    DefaultAzureCredential tries multiple auth methods in order:
    environment variables, workload identity, managed identity, Azure CLI, etc.
    For local dev, running `az login` is the simplest approach.
    """
    if not CONNECTION_STRING:
        raise ValueError(
            "AZURE_AI_PROJECT_CONNECTION_STRING is not set. "
            "Copy it from AI Foundry > your project > Settings > Overview."
        )
    return AIProjectClient.from_connection_string(
        conn_str=CONNECTION_STRING,
        credential=DefaultAzureCredential(),
    )


def run_basic_agent(question: str) -> str:
    """Create a single-turn agent interaction and return the assistant reply."""
    client = create_client()

    # ------------------------------------------------------------------
    # 1. Create the agent
    # ------------------------------------------------------------------
    print("[1/6] Creating agent...")
    agent = client.agents.create_agent(
        model=MODEL_DEPLOYMENT,
        name="basic-demo-agent",
        instructions=(
            "You are a helpful AI assistant. "
            "Answer questions concisely and accurately. "
            "If you are unsure, say so rather than guessing."
        ),
    )
    print(f"      Agent created: id={agent.id}, name={agent.name}")

    try:
        # ------------------------------------------------------------------
        # 2. Create a thread
        # ------------------------------------------------------------------
        print("[2/6] Creating thread...")
        thread = client.agents.create_thread()
        print(f"      Thread created: id={thread.id}")

        # ------------------------------------------------------------------
        # 3. Add a user message to the thread
        # ------------------------------------------------------------------
        print("[3/6] Sending user message...")
        message = client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=question,
        )
        print(f"      Message created: id={message.id}")

        # ------------------------------------------------------------------
        # 4. Create a run and poll until completion
        # ------------------------------------------------------------------
        print("[4/6] Running agent (polling until complete)...")
        run = client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=agent.id,
        )
        print(f"      Run finished: id={run.id}, status={run.status}")

        if run.status == "failed":
            error_msg = run.last_error.message if run.last_error else "Unknown error"
            raise RuntimeError(f"Agent run failed: {error_msg}")

        # ------------------------------------------------------------------
        # 5. Retrieve messages from the thread
        # ------------------------------------------------------------------
        print("[5/6] Retrieving response...")
        messages = client.agents.list_messages(thread_id=thread.id)

        assistant_reply = ""
        for msg in messages.data:
            if msg.role == "assistant":
                for block in msg.content:
                    if hasattr(block, "text"):
                        assistant_reply = block.text.value
                        break
                break

        print("[6/6] Done.")
        return assistant_reply

    finally:
        # ------------------------------------------------------------------
        # 6. Clean up
        # ------------------------------------------------------------------
        print(f"\nCleaning up: deleting agent {agent.id}...")
        client.agents.delete_agent(agent.id)
        print("Agent deleted.")


def demonstrate_multi_turn():
    """Show a multi-turn conversation on the same thread (no agent re-creation)."""
    client = create_client()

    print("\n--- Multi-turn conversation demo ---")
    agent = client.agents.create_agent(
        model=MODEL_DEPLOYMENT,
        name="multi-turn-demo",
        instructions="You are a concise assistant. Keep answers to 1-2 sentences.",
    )

    thread = client.agents.create_thread()

    turns = [
        "What is the capital of France?",
        "What is the population of that city?",
        "Name one famous landmark there.",
    ]

    try:
        for i, user_input in enumerate(turns, 1):
            print(f"\nTurn {i} - User: {user_input}")
            client.agents.create_message(
                thread_id=thread.id, role="user", content=user_input
            )
            run = client.agents.create_and_process_run(
                thread_id=thread.id, agent_id=agent.id
            )
            if run.status != "completed":
                print(f"  Run ended with status: {run.status}")
                continue

            messages = client.agents.list_messages(thread_id=thread.id)
            for msg in messages.data:
                if msg.role == "assistant":
                    for block in msg.content:
                        if hasattr(block, "text"):
                            print(f"Turn {i} - Assistant: {block.text.value}")
                    break
    finally:
        client.agents.delete_agent(agent.id)
        print("\nMulti-turn demo complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Foundry - Basic Agent Demo ===\n")

    question = (
        "Explain the difference between supervised and unsupervised machine learning "
        "in simple terms."
    )
    print(f"User question: {question}\n")

    try:
        answer = run_basic_agent(question)
        print(f"\nAgent answer:\n{answer}")
    except Exception as exc:
        print(f"\nError during single-turn demo: {exc}")

    try:
        demonstrate_multi_turn()
    except Exception as exc:
        print(f"\nError during multi-turn demo: {exc}")
