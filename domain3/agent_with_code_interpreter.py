"""
agent_with_code_interpreter.py
===============================
Demonstrates creating an Azure AI Foundry agent with the Code Interpreter tool enabled.
Code Interpreter allows the agent to write and execute Python code in a sandboxed
environment, making it ideal for data analysis, calculations, and visualisation tasks.

Workflow:
    1. Upload a CSV file to the Foundry project file store
    2. Create an agent with Code Interpreter enabled and the file attached
    3. Send data-analysis questions; the agent writes + runs code internally
    4. Retrieve the response, including any generated file outputs (e.g. charts)
    5. Clean up uploaded files and the agent

Exam Skill Mapping:
    - "Configure the necessary resources to build an agent"
    - "Create an agent with the Microsoft Foundry Agent Service"

Required Environment Variables (.env):
    AZURE_AI_PROJECT_CONNECTION_STRING
    AZURE_OPENAI_DEPLOYMENT

Install:
    pip install azure-ai-projects azure-identity python-dotenv
"""

import os
import io
import csv
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool, MessageAttachment
from azure.identity import DefaultAzureCredential

load_dotenv()

CONNECTION_STRING = os.environ.get("AZURE_AI_PROJECT_CONNECTION_STRING")
MODEL_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# ---------------------------------------------------------------------------
# Helper: generate a sample CSV in memory
# ---------------------------------------------------------------------------
SAMPLE_CSV_DATA = [
    ["month", "revenue", "expenses", "profit"],
    ["Jan", 42000, 31000, 11000],
    ["Feb", 45500, 32500, 13000],
    ["Mar", 39800, 30200, 9600],
    ["Apr", 51000, 34000, 17000],
    ["May", 58000, 37500, 20500],
    ["Jun", 62000, 39000, 23000],
    ["Jul", 59500, 38000, 21500],
    ["Aug", 64000, 41000, 23000],
    ["Sep", 67000, 43000, 24000],
    ["Oct", 72000, 46000, 26000],
    ["Nov", 78000, 48000, 30000],
    ["Dec", 85000, 52000, 33000],
]


def create_sample_csv(path: str) -> None:
    """Write the sample CSV data to the given file path."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(SAMPLE_CSV_DATA)
    print(f"Sample CSV written to: {path}")


def get_assistant_text(messages) -> str:
    """Extract the text from the most recent assistant message."""
    for msg in messages.data:
        if msg.role == "assistant":
            parts = []
            for block in msg.content:
                if hasattr(block, "text"):
                    parts.append(block.text.value)
            return "\n".join(parts)
    return "(no assistant message found)"


def run_code_interpreter_demo():
    """Full demonstration of an agent with Code Interpreter."""

    if not CONNECTION_STRING:
        raise ValueError("AZURE_AI_PROJECT_CONNECTION_STRING is not set.")

    client = AIProjectClient.from_connection_string(
        conn_str=CONNECTION_STRING,
        credential=DefaultAzureCredential(),
    )

    # ------------------------------------------------------------------
    # 1. Create a temporary CSV file and upload it to the project
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="sales_data_"
    ) as tmp:
        csv_path = tmp.name

    create_sample_csv(csv_path)

    print("\n[1/6] Uploading CSV to Foundry project file store...")
    with open(csv_path, "rb") as f:
        uploaded_file = client.agents.upload_file_and_poll(
            file=f,
            purpose="assistants",  # files for agent use must use purpose="assistants"
        )
    print(f"      Uploaded file id: {uploaded_file.id}")
    os.unlink(csv_path)  # Remove local temp file

    # ------------------------------------------------------------------
    # 2. Create the Code Interpreter tool and attach the file
    # ------------------------------------------------------------------
    code_interpreter = CodeInterpreterTool(file_ids=[uploaded_file.id])

    print("[2/6] Creating agent with Code Interpreter tool...")
    agent = client.agents.create_agent(
        model=MODEL_DEPLOYMENT,
        name="code-interpreter-demo",
        instructions=(
            "You are a data analyst assistant. "
            "When given data files, use the code interpreter to perform analysis. "
            "Always show your working by printing intermediate results. "
            "Provide clear, structured summaries of your findings."
        ),
        tools=code_interpreter.definitions,          # tool schema definitions
        tool_resources=code_interpreter.resources,   # file bindings for the tool
    )
    print(f"      Agent id: {agent.id}")

    try:
        thread = client.agents.create_thread()
        print(f"[3/6] Thread created: {thread.id}")

        # ------------------------------------------------------------------
        # 3. Ask the agent to perform analysis tasks
        # ------------------------------------------------------------------
        analysis_questions = [
            (
                "Please analyse the uploaded sales data CSV file. "
                "Calculate the total annual revenue, total expenses, and total profit. "
                "Also identify the best and worst performing months."
            ),
            (
                "Calculate the month-over-month growth rate for revenue "
                "and identify any months where profit margin exceeded 35%."
            ),
        ]

        for turn_num, question in enumerate(analysis_questions, 1):
            print(f"\n--- Turn {turn_num} ---")
            print(f"User: {question}")

            client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=question,
                # Attach the file directly to this message for this turn
                attachments=[
                    MessageAttachment(
                        file_id=uploaded_file.id,
                        tools=code_interpreter.definitions,
                    )
                ],
            )

            run = client.agents.create_and_process_run(
                thread_id=thread.id,
                agent_id=agent.id,
            )
            print(f"Run status: {run.status}")

            if run.status == "failed":
                print(f"Run failed: {run.last_error}")
                continue

            messages = client.agents.list_messages(thread_id=thread.id)
            reply = get_assistant_text(messages)
            print(f"\nAssistant:\n{reply}")

            # ------------------------------------------------------------------
            # 4. Check for generated image/file outputs from code execution
            # ------------------------------------------------------------------
            for msg in messages.data:
                if msg.role == "assistant":
                    for block in msg.content:
                        # ImageFileContent blocks contain generated plots
                        if hasattr(block, "image_file"):
                            file_id = block.image_file.file_id
                            print(f"\n[Generated image output] file_id={file_id}")
                            # Download the generated image
                            image_data = client.agents.get_file_content(file_id)
                            out_path = f"generated_chart_{file_id}.png"
                            with open(out_path, "wb") as img_f:
                                img_f.write(image_data)
                            print(f"      Saved chart to: {out_path}")

    finally:
        # ------------------------------------------------------------------
        # 5. Clean up: delete uploaded file and agent
        # ------------------------------------------------------------------
        print(f"\n[6/6] Cleaning up...")
        client.agents.delete_file(uploaded_file.id)
        print(f"      Deleted file: {uploaded_file.id}")
        client.agents.delete_agent(agent.id)
        print(f"      Deleted agent: {agent.id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Foundry — Agent with Code Interpreter Demo ===\n")
    try:
        run_code_interpreter_demo()
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
