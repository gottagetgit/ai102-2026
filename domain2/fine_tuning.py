"""
fine_tuning.py
==============
Demonstrates the Azure OpenAI fine-tuning workflow:
  1. Prepare training data in the required JSONL format
  2. Upload the training file to Azure OpenAI
  3. Create a fine-tuning job
  4. Monitor job progress
  5. Deploy the fine-tuned model
  6. Call the fine-tuned model
  7. Clean up (delete job and deployment)

Exam Skill: "Fine-tune an Azure OpenAI model"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - JSONL format for fine-tuning training data
  - File upload for fine-tuning
  - Creating and monitoring fine-tuning jobs
  - Deploying a fine-tuned model
  - Using the fine-tuned model for inference
  - Cost considerations and when to use fine-tuning

When to fine-tune vs. RAG:
  - Fine-tune: Change model STYLE, FORMAT, or TONE
    e.g., "Always respond in JSON", "Use our brand voice", "Follow this template"
  - RAG: Add external KNOWLEDGE the model doesn't have
    e.g., "Answer from our latest product docs", "Use this year's data"

Fine-tuning is NOT for adding knowledge - it's for changing behavior.

Required packages:
  pip install openai python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT    - your Azure OpenAI endpoint
  AZURE_OPENAI_KEY         - your API key
  AZURE_SUBSCRIPTION_ID    - for deployment management (optional)
"""

import os
import json
import time
import tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.fine_tuning import FineTuningJob

load_dotenv()

ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_KEY"]

# Base model to fine-tune (must support fine-tuning)
# Models that support fine-tuning in Azure OpenAI:
#   - gpt-4o-mini (recommended for fine-tuning, cost-effective)
#   - gpt-35-turbo
#   - gpt-35-turbo-16k
BASE_MODEL = "gpt-4o-mini"

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version="2024-05-01-preview",  # Fine-tuning requires this API version
)


# ---------------------------------------------------------------------------
# Step 1: Prepare training data
# ---------------------------------------------------------------------------

def prepare_training_data() -> str:
    """
    Prepare fine-tuning training data in the required JSONL format.

    Each line is a JSON object with a 'messages' array.
    Same format as chat completions: system, user, assistant messages.

    Example use case: Fine-tune model to always respond in a structured
    JSON format for Azure AI service classification.

    Returns:
        Path to the JSONL training file
    """
    print("\n--- Step 1: Preparing Training Data ---")

    # Training examples: teach the model to classify Azure AI services
    # and respond in a specific JSON format
    training_examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "Service that converts speech to text in real-time"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure AI Speech", "domain": "Speech", "confidence": 0.98}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "API for detecting objects, faces, and text in images"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure AI Vision", "domain": "Computer Vision", "confidence": 0.96}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "Service for extracting named entities from text documents"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure AI Language", "domain": "NLP", "confidence": 0.95}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "Tool for analyzing PDFs, forms, and invoices to extract structured data"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure AI Document Intelligence", "domain": "Document Processing", "confidence": 0.97}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "Service for creating conversational AI with custom intents and entities"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure AI Language (CLU)", "domain": "NLP", "confidence": 0.93}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "Platform for building AI-powered search with semantic ranking"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure AI Search", "domain": "Knowledge Mining", "confidence": 0.94}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "API for generating images from text descriptions"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure OpenAI (DALL-E)", "domain": "Generative AI", "confidence": 0.99}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "Service for detecting harmful content in text and images"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure AI Content Safety", "domain": "Responsible AI", "confidence": 0.98}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "Model that can understand and generate both text and images"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure OpenAI (GPT-4o)", "domain": "Generative AI", "confidence": 0.97}'
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only in this format: {\"service\": \"<name>\", \"domain\": \"<domain>\", \"confidence\": <0-1>}"
                },
                {
                    "role": "user",
                    "content": "Tool for translating text between 100+ languages"
                },
                {
                    "role": "assistant",
                    "content": '{"service": "Azure AI Translator", "domain": "NLP", "confidence": 0.99}'
                }
            ]
        },
    ]

    # Write to a temporary JSONL file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        prefix="ai102_finetune_"
    ) as f:
        for example in training_examples:
            f.write(json.dumps(example) + "\n")
        temp_path = f.name

    print(f"  Created {len(training_examples)} training examples")
    print(f"  Saved to: {temp_path}")
    print(f"  Format: JSONL (one JSON object per line)")
    print(f"  Each example has: system + user + assistant messages")

    return temp_path


# ---------------------------------------------------------------------------
# Step 2: Upload training file
# ---------------------------------------------------------------------------

def upload_training_file(file_path: str) -> str:
    """
    Upload the training file to Azure OpenAI.
    Returns the file ID for use in job creation.
    """
    print("\n--- Step 2: Uploading Training File ---")

    try:
        with open(file_path, "rb") as f:
            upload_response = client.files.create(
                file=f,
                purpose="fine-tune",  # Must be 'fine-tune' for training data
            )

        file_id = upload_response.id
        print(f"  File uploaded successfully")
        print(f"  File ID: {file_id}")
        print(f"  Status: {upload_response.status}")
        print(f"  Purpose: {upload_response.purpose}")

        # Wait for file to be processed
        print("  Waiting for file to be processed...")
        max_wait = 60  # seconds
        start = time.time()
        while time.time() - start < max_wait:
            file_info = client.files.retrieve(file_id)
            if file_info.status == "processed":
                print(f"  File processed successfully")
                break
            elif file_info.status == "error":
                print(f"  File processing failed: {file_info.status_details}")
                raise RuntimeError(f"File processing failed")
            time.sleep(5)
            print(f"  Status: {file_info.status}...")

        return file_id

    except Exception as e:
        print(f"  [ERROR] File upload failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Step 3: Create fine-tuning job
# ---------------------------------------------------------------------------

def create_fine_tuning_job(
    training_file_id: str,
    validation_file_id: Optional[str] = None,
) -> str:
    """
    Create a fine-tuning job using the uploaded training file.
    Returns the job ID.
    """
    print("\n--- Step 3: Creating Fine-Tuning Job ---")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Training file: {training_file_id}")

    try:
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=BASE_MODEL,
            hyperparameters={
                "n_epochs": 3,           # Number of training epochs (default: auto)
                "batch_size": "auto",    # Batch size (default: auto)
                "learning_rate_multiplier": "auto",  # LR multiplier
            },
            suffix="ai102-demo",         # Custom suffix for the model name
        )

        print(f"  Job created successfully")
        print(f"  Job ID: {job.id}")
        print(f"  Status: {job.status}")
        print(f"  Model: {job.model}")

        return job.id

    except Exception as e:
        print(f"  [ERROR] Job creation failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Step 4: Monitor fine-tuning job
# ---------------------------------------------------------------------------

def monitor_job(job_id: str, poll_interval: int = 60) -> str:
    """
    Poll the fine-tuning job until it completes or fails.
    Returns the fine-tuned model name on success.

    Note: Fine-tuning typically takes 30-60 minutes for small datasets.
    """
    print("\n--- Step 4: Monitoring Fine-Tuning Job ---")
    print(f"  Job ID: {job_id}")
    print(f"  Polling every {poll_interval} seconds...")
    print("  (Fine-tuning typically takes 30-60 minutes)")

    while True:
        job: FineTuningJob = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        elapsed = ""

        if job.finished_at and job.created_at:
            elapsed = f" | Elapsed: {(job.finished_at - job.created_at) // 60}m"

        print(f"  Status: {status}{elapsed}")

        # List recent events
        events = client.fine_tuning.jobs.list_events(job_id, limit=5)
        for event in reversed(list(events.data)):
            print(f"    [{event.created_at}] {event.message}")

        if status == "succeeded":
            print(f"  Fine-tuning complete!")
            print(f"  Fine-tuned model: {job.fine_tuned_model}")
            return job.fine_tuned_model

        elif status in ("failed", "cancelled"):
            print(f"  Fine-tuning {status}")
            if job.error:
                print(f"  Error: {job.error}")
            raise RuntimeError(f"Fine-tuning job {status}: {job.error}")

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Step 5: Deploy fine-tuned model
# ---------------------------------------------------------------------------

def deploy_fine_tuned_model(fine_tuned_model: str) -> str:
    """
    Deploy the fine-tuned model to make it available for inference.

    Note: Deployment is done via the Azure OpenAI REST API or Azure Portal.
    The Python SDK doesn't have a direct deployment method.
    We show the Azure CLI command instead.

    Returns the deployment name.
    """
    deployment_name = "ai102-fine-tuned"

    print("\n--- Step 5: Deploying Fine-Tuned Model ---")
    print(f"  Fine-tuned model: {fine_tuned_model}")
    print(f"  Deployment name: {deployment_name}")
    print()
    print("  To deploy via Azure CLI:")
    print(f"    az cognitiveservices account deployment create \\")
    print(f"      --resource-group <rg> \\")
    print(f"      --name <openai-resource-name> \\")
    print(f"      --deployment-name {deployment_name} \\")
    print(f"      --model-name {fine_tuned_model} \\")
    print(f"      --model-version 1 \\")
    print(f"      --model-format OpenAI \\")
    print(f"      --sku-capacity 1 \\")
    print(f"      --sku-name Standard")
    print()
    print("  Or via Azure AI Foundry portal:")
    print("  https://ai.azure.com/ > Deployments > Deploy model > Fine-tuned models")

    return deployment_name


# ---------------------------------------------------------------------------
# Step 6: Use the fine-tuned model
# ---------------------------------------------------------------------------

def use_fine_tuned_model(deployment_name: str) -> None:
    """
    Call the fine-tuned model for inference.
    Same API as the base model - just use the deployment name.
    """
    print("\n--- Step 6: Using Fine-Tuned Model ---")
    print(f"  Deployment: {deployment_name}")

    test_prompts = [
        "API for transcribing audio files to text",
        "Service that creates custom image classification models",
        "Tool for extracting key-value pairs from scanned invoices",
    ]

    for prompt in test_prompts:
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an Azure AI service classifier. Classify the given description and respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1,
            )
            print(f"  Input: '{prompt}'")
            print(f"  Output: {response.choices[0].message.content}")
            print()
        except Exception as e:
            print(f"  [INFO] Model not deployed yet: {e}")
            print("  Deploy the model first to test inference")
            break


# ---------------------------------------------------------------------------
# Step 7: List and clean up
# ---------------------------------------------------------------------------

def list_fine_tuning_jobs() -> None:
    """List all fine-tuning jobs in the Azure OpenAI resource."""
    print("\n--- Fine-Tuning Jobs ---")

    try:
        jobs = client.fine_tuning.jobs.list(limit=10)
        for job in jobs.data:
            print(f"  ID: {job.id}")
            print(f"    Model: {job.model}")
            print(f"    Status: {job.status}")
            print(f"    Fine-tuned model: {job.fine_tuned_model or 'N/A'}")
            print()
    except Exception as e:
        print(f"  [ERROR] {e}")


def delete_training_file(file_id: str) -> None:
    """Delete the uploaded training file."""
    print(f"\n  Deleting training file: {file_id}")
    try:
        client.files.delete(file_id)
        print(f"  File deleted successfully")
    except Exception as e:
        print(f"  [WARN] Could not delete file: {e}")


# ---------------------------------------------------------------------------
# Fine-tuning guidance
# ---------------------------------------------------------------------------

def print_fine_tuning_guidance() -> None:
    """
    Print guidance on when and how to use fine-tuning.
    Key exam concepts.
    """
    print("\n" + "=" * 60)
    print("FINE-TUNING GUIDANCE (AI-102 Key Concepts)")
    print("=" * 60)
    print("""
When to Fine-Tune:
  YES - Consistent output FORMAT (always JSON, always bullet points)
  YES - Specific TONE or STYLE (formal, brand voice, brevity)
  YES - Domain-specific BEHAVIOR (classify our products, use our taxonomy)
  NO  - Adding factual KNOWLEDGE (use RAG instead)
  NO  - Real-time or frequently updated information (use RAG)

Fine-Tunable Models in Azure OpenAI:
  - gpt-4o-mini (recommended - cost-effective, good results)
  - gpt-35-turbo
  - gpt-35-turbo-16k
  - babbage-002 (completion models)
  - davinci-002 (completion models)

Training Data Requirements:
  - Minimum: 10 examples (but 50-100+ recommended for good results)
  - Format: JSONL with 'messages' key (chat format)
  - Each example: system + user + assistant messages
  - Validation set: optional but recommended (10-20% of data)

Hyperparameters:
  - n_epochs: How many times to train on the full dataset (default: auto)
  - batch_size: Training batch size (default: auto)
  - learning_rate_multiplier: LR scaling factor (default: auto)

Cost Structure:
  - Training cost: per 1K tokens in training data
  - Hosting cost: per hour the fine-tuned model is deployed
  - Inference cost: per token (same as base model)
""")


if __name__ == "__main__":
    print("=" * 60)
    print("Azure OpenAI Fine-Tuning Demo")
    print("=" * 60)

    # Show guidance
    print_fine_tuning_guidance()

    # List existing jobs
    list_fine_tuning_jobs()

    print("\n--- Running Fine-Tuning Workflow Demo ---")
    print("Note: This will create real Azure resources and incur costs.")
    print("Uncomment the sections below to run the actual workflow.")
    print()

    # Step 1: Prepare data
    training_file_path = prepare_training_data()
    print(f"Training data prepared at: {training_file_path}")
    print("(File not uploaded - uncomment workflow steps to proceed)")

    # UNCOMMENT TO RUN FULL WORKFLOW:
    # -------------------------------------------------------
    # # Step 2: Upload file
    # file_id = upload_training_file(training_file_path)
    #
    # # Step 3: Create job
    # job_id = create_fine_tuning_job(file_id)
    #
    # # Step 4: Monitor (this will block for 30-60 minutes)
    # fine_tuned_model = monitor_job(job_id, poll_interval=60)
    #
    # # Step 5: Deploy
    # deployment_name = deploy_fine_tuned_model(fine_tuned_model)
    #
    # # Step 6: Use the model
    # use_fine_tuned_model(deployment_name)
    #
    # # Cleanup
    # delete_training_file(file_id)
    # -------------------------------------------------------

    print("\n" + "=" * 60)
    print("Fine-Tuning Demo Complete")
    print("=" * 60)
