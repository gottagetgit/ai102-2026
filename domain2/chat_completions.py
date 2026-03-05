"""
chat_completions.py
===================
Demonstrates sending chat completions to Azure OpenAI with various
configuration options: single-turn, multi-turn, system prompts, streaming.

Exam Skill: "Use the Azure OpenAI chat completions API"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Basic chat completion (single-turn)
  - Multi-turn conversation with history management
  - System prompt configuration
  - Streaming response handling
  - Token usage tracking
  - JSON mode for structured output
  - Error handling and retry patterns

Required packages:
  pip install openai python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT    - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY         - your API key
  AZURE_OPENAI_DEPLOYMENT  - deployment name, e.g. 'gpt-4o'
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv
import openai
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion

load_dotenv()

ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version="2024-02-01",
)


# ---------------------------------------------------------------------------
# Demo 1: Single-turn chat completion
# ---------------------------------------------------------------------------

def demo_single_turn() -> None:
    """Basic single-turn chat completion."""
    print("\n" + "=" * 60)
    print("DEMO 1: Single-Turn Chat Completion")
    print("=" * 60)

    response: ChatCompletion = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "user", "content": "What are the three primary colors?"}
        ],
        max_tokens=100,
        temperature=0.7,
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens} "
          f"(prompt: {response.usage.prompt_tokens}, "
          f"completion: {response.usage.completion_tokens})")
    print(f"Finish reason: {response.choices[0].finish_reason}")


# ---------------------------------------------------------------------------
# Demo 2: System prompt configuration
# ---------------------------------------------------------------------------

def demo_system_prompt() -> None:
    """Demonstrate how system prompts configure model behavior."""
    print("\n" + "=" * 60)
    print("DEMO 2: System Prompt Configuration")
    print("=" * 60)

    system_prompt = """
    You are a concise technical assistant specializing in Azure AI services.
    - Always respond in bullet points
    - Limit responses to 3-5 bullet points
    - Focus only on Azure-specific information
    - If asked about non-Azure topics, redirect to Azure equivalents
    """

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What are the main Azure AI services for NLP?"}
        ],
        max_tokens=300,
        temperature=0.3,
    )

    print(f"System prompt: (configured assistant as Azure specialist)")
    print(f"\nResponse:\n{response.choices[0].message.content}")


# ---------------------------------------------------------------------------
# Demo 3: Multi-turn conversation
# ---------------------------------------------------------------------------

def demo_multi_turn() -> None:
    """Multi-turn conversation with conversation history management."""
    print("\n" + "=" * 60)
    print("DEMO 3: Multi-Turn Conversation")
    print("=" * 60)

    # Build conversation history
    conversation = [
        {"role": "system", "content": "You are a helpful Azure AI exam preparation assistant."},
    ]

    # Turn 1
    user_msg_1 = "What is Azure AI Services?"
    print(f"User: {user_msg_1}")
    conversation.append({"role": "user", "content": user_msg_1})

    response_1 = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=conversation,
        max_tokens=150,
        temperature=0.5,
    )
    assistant_msg_1 = response_1.choices[0].message.content
    print(f"Assistant: {assistant_msg_1}\n")
    conversation.append({"role": "assistant", "content": assistant_msg_1})

    # Turn 2 - follow-up referencing previous answer
    user_msg_2 = "What are three specific services included in that?"
    print(f"User: {user_msg_2}")
    conversation.append({"role": "user", "content": user_msg_2})

    response_2 = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=conversation,
        max_tokens=200,
        temperature=0.5,
    )
    print(f"Assistant: {response_2.choices[0].message.content}")

    print(f"\nTotal conversation turns: {len([m for m in conversation if m['role'] != 'system'])}")


# ---------------------------------------------------------------------------
# Demo 4: Streaming response
# ---------------------------------------------------------------------------

def demo_streaming() -> None:
    """Stream response tokens as they are generated."""
    print("\n" + "=" * 60)
    print("DEMO 4: Streaming Response")
    print("=" * 60)
    print("Streaming: ", end="", flush=True)

    stream = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "user", "content": "List 5 Azure AI services in one sentence each."}
        ],
        max_tokens=300,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            full_response += token

    print()  # newline after streaming
    print(f"\nTotal characters streamed: {len(full_response)}")


# ---------------------------------------------------------------------------
# Demo 5: JSON mode for structured output
# ---------------------------------------------------------------------------

def demo_json_mode() -> None:
    """Use JSON mode to get structured output from the model."""
    print("\n" + "=" * 60)
    print("DEMO 5: JSON Mode (Structured Output)")
    print("=" * 60)

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You are a JSON API. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": "List 3 Azure AI services as JSON: [{\"name\": ..., \"category\": ..., \"use_case\": ...}]"
            }
        ],
        response_format={"type": "json_object"},
        max_tokens=400,
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    print(f"Raw JSON response:\n{raw}")

    try:
        parsed = json.loads(raw)
        print(f"\nParsed successfully. Type: {type(parsed).__name__}")
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")


# ---------------------------------------------------------------------------
# Demo 6: Error handling
# ---------------------------------------------------------------------------

def demo_error_handling() -> None:
    """Demonstrate proper error handling for common Azure OpenAI errors."""
    print("\n" + "=" * 60)
    print("DEMO 6: Error Handling")
    print("=" * 60)

    # Test with an invalid deployment name
    try:
        response = client.chat.completions.create(
            model="non-existent-deployment-xyz",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
        )
    except openai.NotFoundError as e:
        print(f"NotFoundError: Deployment not found")
        print(f"  Status: {e.status_code}")
        print(f"  Message: {str(e)[:100]}")
    except openai.AuthenticationError as e:
        print(f"AuthenticationError: Invalid API key")
        print(f"  Status: {e.status_code}")
    except openai.RateLimitError as e:
        print(f"RateLimitError: Quota exceeded - implement exponential backoff")
        print(f"  Status: {e.status_code}")
    except openai.APIError as e:
        print(f"APIError: {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Azure OpenAI Chat Completions Demo")
    print("=" * 60)

    demo_single_turn()
    demo_system_prompt()
    demo_multi_turn()
    demo_streaming()
    demo_json_mode()
    demo_error_handling()

    print("\n" + "=" * 60)
    print("Chat Completions Demo Complete")
    print("=" * 60)
