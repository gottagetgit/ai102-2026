"""
agent_with_file_search.py
=========================
Demonstrates creating an Azure AI Foundry agent with the File Search tool.
File Search enables Retrieval-Augmented Generation (RAG) over documents you upload.
The agent automatically chunks documents, creates embeddings, and stores them in a
managed vector store backed by Azure AI Search.

Workflow:
    1. Upload one or more documents (PDF/TXT/DOCX etc.) to the project file store
    2. Create a Vector Store and add the uploaded files to it
    3. Create an agent with the FileSearchTool bound to that vector store
    4. Ask questions - the agent retrieves relevant chunks before answering
    5. Inspect file citations in the response
    6. Clean up: delete vector store, files, and agent

Exam Skill Mapping:
    - "Create an agent with the Microsoft Foundry Agent Service"
    - "Configure the necessary resources to build an agent"

Required Environment Variables (.env):
    AZURE_AI_PROJECT_CONNECTION_STRING
    AZURE_OPENAI_DEPLOYMENT

Install:
    pip install azure-ai-projects azure-identity python-dotenv
"""

import os
import time
import tempfile
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FileSearchTool
from azure.identity import DefaultAzureCredential

load_dotenv()

CONNECTION_STRING = os.environ.get("AZURE_AI_PROJECT_CONNECTION_STRING")
MODEL_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# ---------------------------------------------------------------------------
# Sample documents to upload (written to temp files)
# ---------------------------------------------------------------------------
SAMPLE_DOCS = {
    "azure_ai_services_overview.txt": """\
Azure AI Services Overview
==========================
Azure AI Services (formerly Cognitive Services) provide pre-built AI capabilities
via REST APIs and client libraries. Key service categories include:

Vision:
  - Azure AI Vision: image analysis, OCR, face detection, spatial analysis
  - Custom Vision: train classification and object-detection models on your data
  - Face API: face detection, verification, identification, and emotion recognition

Speech:
  - Speech to Text (STT): real-time and batch transcription
  - Text to Speech (TTS): neural voice synthesis
  - Speech Translation: real-time translation across languages

Language:
  - Azure AI Language: sentiment analysis, NER, key phrase extraction, summarisation
  - Translator: text translation supporting 100+ languages
  - Question Answering: build FAQ bots from knowledge bases

Decision:
  - Anomaly Detector: detect anomalies in time-series data
  - Content Moderator: detect offensive or inappropriate content
  - Personaliser: reinforcement-learning-based content recommendations

All services are accessed via a unique endpoint URL and authentication key,
or via Azure Active Directory / Managed Identity for keyless auth.
""",
    "azure_openai_faq.txt": """\
Azure OpenAI Service - Frequently Asked Questions
==================================================
Q: What models are available in Azure OpenAI?
A: Azure OpenAI hosts OpenAI's GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, DALL-E 3,
   Whisper, and the text-embedding-ada-002 / text-embedding-3 models.

Q: How is Azure OpenAI different from OpenAI's API?
A: Azure OpenAI runs within Microsoft Azure infrastructure, offering enterprise
   SLAs, private networking (VNet integration), Azure RBAC, content filtering,
   and compliance certifications (SOC 2, ISO 27001, HIPAA BAA).

Q: What are deployments?
A: A deployment is an instance of a base model that you provision in your Azure
   OpenAI resource. Each deployment has a name, model version, and token-rate limit.

Q: What is Responsible AI filtering?
A: Azure OpenAI applies content filters that check both prompts and completions
   for hate speech, violence, sexual content, and self-harm. Filters can be
   configured per deployment within Microsoft's Responsible AI policy boundaries.

Q: What is the context window?
A: The context window is the maximum number of tokens (input + output) a model
   can process in one API call. GPT-4o supports up to 128 000 tokens.
""",
}


def write_temp_docs() -> dict:
    """Write sample documents to temp files; return {filename: path} mapping."""
    paths = {}
    for name, content in SAMPLE_DOCS.items():
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix=name.replace(".txt", "_")
        )
        tmp.write(content)
        tmp.close()
        paths[name] = tmp.name
    return paths


def get_assistant_text_with_citations(messages) -> tuple[str, list]:
    """Extract assistant reply text and any file citations.

    Returns:
        (text, citations) where citations is a list of dicts with
        file_id and quoted text.
    """
    for msg in messages.data:
        if msg.role == "assistant":
            full_text = []
            citations = []
            for block in msg.content:
                if hasattr(block, "text"):
                    full_text.append(block.text.value)
                    # Annotations carry file citation metadata
                    for annotation in block.text.annotations:
                        if hasattr(annotation, "file_citation"):
                            citations.append({
                                "file_id": annotation.file_citation.file_id,
                                "quote": getattr(annotation.file_citation, "quote", ""),
                                "placeholder": annotation.text,
                            })
            return "\n".join(full_text), citations
    return "(no assistant message)", []


def run_file_search_demo():
    """Full File Search agent demonstration."""
    if not CONNECTION_STRING:
        raise ValueError("AZURE_AI_PROJECT_CONNECTION_STRING is not set.")

    client = AIProjectClient.from_connection_string(
        conn_str=CONNECTION_STRING,
        credential=DefaultAzureCredential(),
    )

    # ------------------------------------------------------------------
    # 1. Upload documents
    # ------------------------------------------------------------------
    print("[1/7] Writing and uploading sample documents...")
    temp_paths = write_temp_docs()
    uploaded_file_ids = []

    for doc_name, tmp_path in temp_paths.items():
        with open(tmp_path, "rb") as f:
            uploaded = client.agents.upload_file_and_poll(
                file=f, purpose="assistants"
            )
        uploaded_file_ids.append(uploaded.id)
        os.unlink(tmp_path)
        print(f"      Uploaded '{doc_name}' -> file_id={uploaded.id}")

    # ------------------------------------------------------------------
    # 2. Create a Vector Store and add files
    # ------------------------------------------------------------------
    print("\n[2/7] Creating vector store and ingesting files...")
    vector_store = client.agents.create_vector_store_and_poll(
        name="ai102-knowledge-base",
        file_ids=uploaded_file_ids,
    )
    print(f"      Vector store id: {vector_store.id}")
    print(f"      File counts: {vector_store.file_counts}")

    # ------------------------------------------------------------------
    # 3. Create agent with FileSearchTool bound to the vector store
    # ------------------------------------------------------------------
    file_search = FileSearchTool(vector_store_ids=[vector_store.id])

    print("\n[3/7] Creating agent with File Search tool...")
    agent = client.agents.create_agent(
        model=MODEL_DEPLOYMENT,
        name="file-search-demo",
        instructions=(
            "You are a knowledgeable AI assistant. Use the provided documents "
            "to answer questions accurately. Always cite specific information from "
            "the documents when relevant. If the answer is not in the documents, "
            "say so clearly."
        ),
        tools=file_search.definitions,
        tool_resources=file_search.resources,
    )
    print(f"      Agent id: {agent.id}")

    try:
        thread = client.agents.create_thread()
        print(f"\n[4/7] Thread created: {thread.id}")

        # ------------------------------------------------------------------
        # 4. Ask questions that require retrieval from the documents
        # ------------------------------------------------------------------
        questions = [
            "What speech services are available in Azure AI Services?",
            "How does Azure OpenAI differ from the standard OpenAI API?",
            "What is a deployment in the context of Azure OpenAI?",
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i} ---")
            print(f"User: {question}")

            client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=question,
            )

            run = client.agents.create_and_process_run(
                thread_id=thread.id,
                agent_id=agent.id,
            )

            if run.status == "failed":
                print(f"Run failed: {run.last_error}")
                continue

            messages = client.agents.list_messages(thread_id=thread.id)
            reply, citations = get_assistant_text_with_citations(messages)

            print(f"\nAssistant:\n{reply}")

            if citations:
                print(f"\nFile citations ({len(citations)}):")
                for cite in citations:
                    print(f"  - File ID: {cite['file_id']}")
                    if cite["quote"]:
                        print(f"    Quote: \"{cite['quote'][:100]}...\"")

    finally:
        # ------------------------------------------------------------------
        # 5. Clean up
        # ------------------------------------------------------------------
        print(f"\n[7/7] Cleaning up...")
        client.agents.delete_agent(agent.id)
        print(f"      Deleted agent: {agent.id}")
        client.agents.delete_vector_store(vector_store.id)
        print(f"      Deleted vector store: {vector_store.id}")
        for file_id in uploaded_file_ids:
            client.agents.delete_file(file_id)
            print(f"      Deleted file: {file_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Foundry - Agent with File Search Demo ===\n")
    try:
        run_file_search_demo()
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
