"""
fine_tuning.py
==============
Demonstrates the Azure OpenAI fine-tuning workflow:
  1. Prepare training data in the required JSONL format
  2. Upload the training file to Azure OpenAI
  3. Create a fine-tuning job
  4. Monitor the job status until completion
  5. Use the fine-tuned deployment

Exam Skill: "Fine-tune a generative model"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Correct JSONL training data format for chat fine-tuning
  - Validation rules for training data
  - Uploading files via the Azure OpenAI Files API
  - Creating a fine-tuning job with hyperparameters
  - Polling job status and retrieving events/metrics
  - Using the fine-tuned model after deployment
  - When to use fine-tuning vs. prompt engineering vs. RAG

Fine-tuning vs. other techniques:
  - Prompt engineering: Free, fast, no training. Use first.
  - RAG: Adds knowledge without training. Good for large/dynamic knowledge.
  - Fine-tuning: Trains the model on your style/format. Use when prompt
    engineering fails to achieve consistent format/tone, you have 50+ examples,
    and the behavior needs to be baked into the model.

Supported models for fine-tuning in Azure OpenAI (as of 2025):
  - gpt-4o-mini-2024-07-18
  - gpt-4o-2024-08-06
  - gpt-35-turbo-0125

Required packages:
  pip install openai python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT    - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY         - API key
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import openai

load_dotenv()

ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY  = os.environ["AZURE_OPENAI_KEY"]

# The base model to fine-tune (must support fine-tuning)
BASE_MODEL = "gpt-4o-mini-2024-07-18"

TRAINING_FILE_PATH  = Path("./fine_tuning_data/training.jsonl")
VALIDATION_FILE_PATH = Path("./fine_tuning_data/validation.jsonl")


def get_client() -> openai.AzureOpenAI:
    """Return authenticated AzureOpenAI client."""
    return openai.AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version="2024-10-01-preview",  # Fine-tuning API version
    )


# ---------------------------------------------------------------------------
# Step 1: Prepare training data
# ---------------------------------------------------------------------------

TRAINING_EXAMPLES = [
    # Format: each example is a complete conversation showing desired behavior
    # The model learns from the relationship between user messages and assistant responses
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "What is the difference between Azure AI Services and Azure OpenAI?"},
            {"role": "assistant", "content": "Azure AI Services is a collection of pre-built AI capabilities like language detection, sentiment analysis, computer vision, and speech recognition. Azure OpenAI is a specific service that provides access to powerful large language models (GPT-4, GPT-4o) and DALL-E for generative AI tasks.\n\nKey takeaway: Use Azure AI Services for structured AI tasks (classify, detect, transcribe); use Azure OpenAI for generative tasks (write, reason, generate)."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "When should I use Azure AI Content Safety?"},
            {"role": "assistant", "content": "Azure AI Content Safety should be used when your application processes user-generated content or displays AI-generated content to users. It detects harmful content across four categories: hate speech, violence, self-harm, and sexual content, each rated on a 0-6 severity scale.\n\nKey takeaway: Integrate Content Safety in any user-facing application where harmful content could appear — chatbots, content platforms, social features, or AI-generated content pipelines."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "What authentication options are available for Azure AI Services?"},
            {"role": "assistant", "content": "Azure AI Services supports two main authentication methods: API key authentication (simple, uses a static key) and Microsoft Entra ID authentication (keyless, uses identity tokens). For production workloads, Entra ID with managed identity is strongly recommended as it eliminates key management overhead.\n\nKey takeaway: Use API keys for development/testing; use Entra ID + managed identity in production for better security and compliance."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "How do I implement RAG with Azure services?"},
            {"role": "assistant", "content": "RAG (Retrieval-Augmented Generation) combines Azure OpenAI embeddings, Azure AI Search, and a chat model. The pipeline: (1) embed documents with text-embedding-3-small, (2) store in Azure AI Search with vector fields, (3) at query time, embed the user question, search for similar chunks, (4) pass retrieved chunks as context to GPT-4o, (5) return grounded answer.\n\nKey takeaway: RAG is the preferred approach for giving AI models access to your private data without fine-tuning the model itself."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "What are prompt shields in Azure AI Content Safety?"},
            {"role": "assistant", "content": "Prompt shields detect two types of prompt injection attacks: direct injection (jailbreaks) where users try to override system instructions, and indirect injection where malicious instructions are hidden in documents or data that the AI processes. The ShieldPrompt API analyzes both the user message and retrieved documents.\n\nKey takeaway: Always run prompt shields before sending user input and retrieved context to an LLM, especially in RAG pipelines where external documents could contain adversarial instructions."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "What is the difference between temperature and top_p?"},
            {"role": "assistant", "content": "Both temperature and top_p control response randomness but work differently. Temperature scales the probability distribution over all tokens: 0=deterministic, 1.5=very random. Top_p (nucleus sampling) restricts which tokens are considered to those covering the top_p probability mass: top_p=0.1 means only consider the most likely tokens covering 10% of probability.\n\nKey takeaway: Tune either temperature or top_p — not both simultaneously. Use temperature for most cases; top_p gives more precise control over vocabulary diversity."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "How do I monitor an Azure AI resource?"},
            {"role": "assistant", "content": "Azure AI Services emit metrics to Azure Monitor automatically. Key metrics to monitor: TotalCalls (request volume), SuccessfulCalls (success rate), TotalErrors (error rate), BlockedCalls (throttling), LatencyE2E (end-to-end latency), and TokenTransaction (for Azure OpenAI token usage). Set alert rules for error spikes and quota exhaustion.\n\nKey takeaway: Set up Azure Monitor alerts for TotalErrors > threshold and BlockedCalls > 0 to detect issues before users are impacted."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "What is chain-of-thought prompting?"},
            {"role": "assistant", "content": "Chain-of-thought (CoT) prompting instructs the model to reason step by step before giving a final answer. This dramatically improves accuracy on complex reasoning, math, and multi-step tasks. Trigger it with phrases like 'Let's think step by step' or by providing worked examples with reasoning shown.\n\nKey takeaway: For any task requiring multi-step reasoning or math, add 'think step by step' to your prompt — it often improves accuracy significantly at no cost."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "When should I fine-tune a model?"},
            {"role": "assistant", "content": "Fine-tune when: (1) you have 50+ high-quality training examples, (2) prompt engineering alone can't achieve your required format/style/behavior consistency, (3) you need to significantly reduce prompt token count (moving instructions into the model), or (4) you need to match a very specific tone or domain style. Always try prompt engineering first since it's free and instant.\n\nKey takeaway: Fine-tuning is a last resort for style and format consistency, not a substitute for knowledge (use RAG for knowledge). Minimum viable dataset: 50 examples; sweet spot: 100-500 examples."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "What DALL-E parameters affect image quality?"},
            {"role": "assistant", "content": "DALL-E 3 has three key parameters: quality ('standard' or 'hd' — HD adds more detail at higher cost), size ('1024x1024', '1792x1024', '1024x1792' — landscape or portrait), and style ('vivid' for dramatic/hyper-real or 'natural' for subtle/photorealistic). DALL-E 3 also rewrites your prompt (visible as revised_prompt) to improve quality.\n\nKey takeaway: For production images use quality='hd' and style='natural'; for social media use quality='standard' and style='vivid'. Always inspect revised_prompt to understand what DALL-E interpreted."},
        ]
    },
]

# Validation examples (subset of training data, used to monitor overfitting)
VALIDATION_EXAMPLES = TRAINING_EXAMPLES[:3]


def validate_training_data(examples: list) -> tuple[bool, list[str]]:
    """
    Validate training examples against Azure OpenAI fine-tuning requirements.

    Rules:
      - Each example must have a 'messages' key
      - Each example must have at least one system/user/assistant message
      - Each conversation must end with an assistant message
      - Messages must have 'role' and 'content'
      - No function_call messages in training data
      - Recommended: 50+ examples for meaningful fine-tuning
    """
    errors = []

    for i, example in enumerate(examples):
        if "messages" not in example:
            errors.append(f"Example {i}: Missing 'messages' key")
            continue

        msgs = example["messages"]

        if not msgs:
            errors.append(f"Example {i}: Empty messages list")
            continue

        # Must end with assistant message
        if msgs[-1]["role"] != "assistant":
            errors.append(f"Example {i}: Last message must be from 'assistant', got '{msgs[-1]['role']}'")

        # Check each message has role and content
        for j, msg in enumerate(msgs):
            if "role" not in msg:
                errors.append(f"Example {i}, message {j}: Missing 'role'")
            if "content" not in msg:
                errors.append(f"Example {i}, message {j}: Missing 'content'")
            if msg.get("role") not in ("system", "user", "assistant"):
                errors.append(f"Example {i}, message {j}: Invalid role '{msg.get('role')}'")

    if len(examples) < 10:
        errors.append(f"WARNING: Only {len(examples)} examples. Azure recommends at least 50 for good results.")

    return len(errors) == 0, errors


def create_training_files() -> None:
    """Write training and validation data to JSONL files."""
    TRAINING_FILE_PATH.parent.mkdir(exist_ok=True)

    with open(TRAINING_FILE_PATH, "w") as f:
        for example in TRAINING_EXAMPLES:
            f.write(json.dumps(example) + "\n")
    print(f"  Training data: {len(TRAINING_EXAMPLES)} examples → {TRAINING_FILE_PATH}")

    with open(VALIDATION_FILE_PATH, "w") as f:
        for example in VALIDATION_EXAMPLES:
            f.write(json.dumps(example) + "\n")
    print(f"  Validation data: {len(VALIDATION_EXAMPLES)} examples → {VALIDATION_FILE_PATH}")


# ---------------------------------------------------------------------------
# Step 2: Upload training file
# ---------------------------------------------------------------------------

def upload_training_file(client: openai.AzureOpenAI, file_path: Path) -> str:
    """
    Upload a training file to Azure OpenAI.
    The purpose="fine-tune" flag tells the API this is training data.
    Returns the file ID needed for creating the fine-tuning job.
    """
    print(f"\n[UPLOAD] Uploading training file: {file_path}")

    with open(file_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune",
        )

    file_id = response.id
    print(f"  File ID: {file_id}")
    print(f"  Status : {response.status}")
    print(f"  Size   : {response.bytes} bytes")

    # Wait for file processing
    print("  Waiting for file processing...")
    while True:
        file_info = client.files.retrieve(file_id)
        if file_info.status == "processed":
            print("  File processed successfully.")
            break
        elif file_info.status == "error":
            raise RuntimeError(f"File processing failed: {file_info.status_details}")
        print(f"  Status: {file_info.status} - waiting...")
        time.sleep(5)

    return file_id


# ---------------------------------------------------------------------------
# Step 3: Create fine-tuning job
# ---------------------------------------------------------------------------

def create_fine_tuning_job(
    client: openai.AzureOpenAI,
    training_file_id: str,
    validation_file_id: str = None,
) -> str:
    """
    Create a fine-tuning job.

    Hyperparameters:
      - n_epochs      : Training passes over the dataset (default: auto, typically 3-4)
      - batch_size    : Examples per gradient update (default: auto)
      - learning_rate_multiplier: Scales the learning rate (default: auto)

    For most cases, leave hyperparameters as 'auto' and let Azure optimize them.
    """
    print(f"\n[FINE-TUNE] Creating fine-tuning job...")
    print(f"  Base model    : {BASE_MODEL}")
    print(f"  Training file : {training_file_id}")
    if validation_file_id:
        print(f"  Validation file: {validation_file_id}")

    kwargs = {
        "model": BASE_MODEL,
        "training_file": training_file_id,
        "hyperparameters": {
            "n_epochs": "auto",       # Let Azure choose optimal epoch count
            # "n_epochs": 3,          # Or specify explicitly
            # "batch_size": "auto",
            # "learning_rate_multiplier": "auto",
        },
    }

    if validation_file_id:
        kwargs["validation_file"] = validation_file_id

    job = client.fine_tuning.jobs.create(**kwargs)

    print(f"  Job ID      : {job.id}")
    print(f"  Status      : {job.status}")
    print(f"  Created at  : {job.created_at}")
    return job.id


# ---------------------------------------------------------------------------
# Step 4: Monitor job status
# ---------------------------------------------------------------------------

def monitor_fine_tuning_job(client: openai.AzureOpenAI, job_id: str) -> str:
    """
    Poll the fine-tuning job until it completes.
    Returns the fine-tuned model ID when complete.

    Job statuses:
      - validating_files : Validating uploaded training data
      - queued           : Waiting for a training slot
      - running          : Training in progress
      - succeeded        : Training complete, model ready
      - failed           : Training failed (check events for reason)
      - cancelled        : Job was cancelled
    """
    print(f"\n[MONITOR] Monitoring job {job_id}...")

    terminal_statuses = {"succeeded", "failed", "cancelled"}
    last_event_id = None

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        print(f"  [{status}] ", end="")

        # Print new events
        events = list(client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=10,
        ))
        events.reverse()

        new_events = []
        for event in events:
            if last_event_id is None or event.id != last_event_id:
                new_events.append(event)

        if new_events:
            last_event_id = new_events[-1].id
            for event in new_events:
                print(f"\n    Event: {event.message}")
        else:
            print()

        if status in terminal_statuses:
            if status == "succeeded":
                model_id = job.fine_tuned_model
                print(f"\n  [SUCCESS] Fine-tuned model: {model_id}")
                return model_id
            else:
                raise RuntimeError(f"Fine-tuning job {status}. Check job events for details.")

        time.sleep(30)  # Poll every 30 seconds


# ---------------------------------------------------------------------------
# Step 5: Use the fine-tuned model
# ---------------------------------------------------------------------------

def use_fine_tuned_model(client: openai.AzureOpenAI, model_id: str) -> None:
    """
    Use the fine-tuned model. After fine-tuning succeeds, you must deploy
    the model in Azure OpenAI Studio to get a deployment name.
    Then call it exactly like any other deployment.
    """
    print(f"\n[INFER] Testing fine-tuned model: {model_id}")
    print("NOTE: Replace 'model_id' with your deployment name after deploying in Azure Portal.")

    response = client.chat.completions.create(
        model=model_id,  # Use the fine-tuned deployment name
        messages=[
            {"role": "system", "content": "You are a helpful Azure AI support assistant. Always structure your answers with a brief explanation followed by a 'Key takeaway:' line."},
            {"role": "user",   "content": "What is Azure Cognitive Search?"},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    print(f"  Response:\n{response.choices[0].message.content}")


def list_fine_tuning_jobs(client: openai.AzureOpenAI) -> None:
    """List all fine-tuning jobs in this Azure OpenAI resource."""
    print("\n[LIST JOBS] Fine-tuning jobs:")
    jobs = list(client.fine_tuning.jobs.list(limit=10))
    if not jobs:
        print("  No jobs found.")
        return
    for job in jobs:
        print(f"  {job.id[:20]}... | Status: {job.status:15s} | Model: {job.model} | Created: {job.created_at}")


def main():
    print("=" * 60)
    print("Azure OpenAI Fine-Tuning Workflow Demo")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")

    try:
        client = get_client()

        # Step 1: Prepare and validate training data
        print("\n[STEP 1] Preparing and validating training data...")
        is_valid, errors = validate_training_data(TRAINING_EXAMPLES)

        if errors:
            print("  Validation issues found:")
            for err in errors:
                print(f"    - {err}")
        else:
            print(f"  All {len(TRAINING_EXAMPLES)} examples valid.")

        create_training_files()

        # Step 2: Upload files
        print("\n[STEP 2] Uploading training files...")
        training_file_id   = upload_training_file(client, TRAINING_FILE_PATH)
        validation_file_id = upload_training_file(client, VALIDATION_FILE_PATH)

        # Step 3: Create fine-tuning job
        print("\n[STEP 3] Creating fine-tuning job...")
        job_id = create_fine_tuning_job(client, training_file_id, validation_file_id)

        # Step 4: Monitor (can take 20-60 minutes)
        print("\n[STEP 4] Monitoring job (this can take 20-60+ minutes)...")
        print("  Tip: You can safely exit and check back using list_fine_tuning_jobs()")
        print("  Uncomment the monitor line below to poll until completion:")
        print(f"  # fine_tuned_model = monitor_fine_tuning_job(client, '{job_id}')")

        # Uncomment to actually wait for completion:
        # fine_tuned_model = monitor_fine_tuning_job(client, job_id)
        # use_fine_tuned_model(client, fine_tuned_model)

        # List all jobs to see status
        list_fine_tuning_jobs(client)

        print(f"\n[NEXT STEPS]")
        print(f"  1. Check job status: pf fine_tuning.jobs.retrieve('{job_id}')")
        print(f"  2. After job succeeds, deploy model in Azure OpenAI Studio")
        print(f"  3. Use the deployment name as model= in chat.completions.create()")

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except openai.APIError as e:
        print(f"\n[ERROR] Azure OpenAI error: {e}")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()
