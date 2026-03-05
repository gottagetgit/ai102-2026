"""
chat_completions.py
===================
Demonstrates sending chat completions to Azure OpenAI with system/user messages,
multi-turn conversation, and streaming responses.

Exam Skill: "Submit prompts to generate code and natural language responses"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Basic single-turn chat completion
  - System message configuration and its effect
  - Multi-turn conversation with message history
  - Streaming responses token by token
  - Completion metadata (token usage, finish reason)
  - Code generation request

Required packages:
  pip install openai python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT    - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY         - API key
  AZURE_OPENAI_DEPLOYMENT  - Deployment name e.g. "gpt-4o"
"""

import os
from dotenv import load_dotenv
import openai

load_dotenv()

ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
API_VERSION = "2024-12-01-preview"


def get_client() -> openai.AzureOpenAI:
    """Create and return an authenticated AzureOpenAI client."""
    return openai.AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version=API_VERSION,
    )


# ---------------------------------------------------------------------------
# 1. Basic single-turn completion
# ---------------------------------------------------------------------------

def basic_completion(client: openai.AzureOpenAI) -> str:
    """
    Send a simple user message and receive a response.
    The messages list is the core of the Chat Completions API:
      - "system"   : Instructions for the model's behavior
      - "user"     : Messages from the human
      - "assistant": Previous model responses (for multi-turn)
    """
    print("\n--- Basic Single-Turn Completion ---")

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Be concise.",
            },
            {
                "role": "user",
                "content": "What are the three laws of robotics?",
            },
        ],
        max_tokens=200,
        temperature=0.7,
    )

    # Extract the response text
    content = response.choices[0].message.content
    print(f"Response:\n{content}")

    # Log completion metadata
    usage = response.usage
    print(f"\nTokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    # stop        = model finished naturally
    # length      = hit max_tokens limit
    # content_filter = content was filtered
    # tool_calls  = model wants to call a function

    return content


# ---------------------------------------------------------------------------
# 2. System message design - showing how it shapes behavior
# ---------------------------------------------------------------------------

def system_message_demo(client: openai.AzureOpenAI) -> None:
    """
    Show how different system messages produce different output styles
    for the same user question.
    """
    print("\n--- System Message Design Demo ---")

    question = "What is machine learning?"

    system_variants = [
        ("Technical Expert",
         "You are a senior machine learning engineer. Use technical terminology "
         "and assume the reader has a PhD in computer science."),
        ("Children's Teacher",
         "You are a friendly teacher explaining technology to a 10-year-old. "
         "Use simple words and fun analogies."),
        ("Business Consultant",
         "You are a business consultant. Focus on ROI, business value, and "
         "practical applications. Avoid technical jargon."),
    ]

    for name, system_prompt in system_variants:
        print(f"\n  Persona: {name}")
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=100,
            temperature=0.5,
        )
        answer = response.choices[0].message.content
        print(f"  Response: {answer[:200]}{'...' if len(answer) > 200 else ''}")


# ---------------------------------------------------------------------------
# 3. Multi-turn conversation
# ---------------------------------------------------------------------------

def multi_turn_conversation(client: openai.AzureOpenAI) -> None:
    """
    Demonstrate a multi-turn conversation by maintaining conversation history.

    Key pattern: Each API call includes ALL previous messages.
    The model has no built-in memory - you must pass history explicitly.
    This means:
      - Longer conversations = more tokens = higher cost
      - You may need to truncate history to stay within context limits
    """
    print("\n--- Multi-Turn Conversation ---")

    # Initialize conversation with a system message
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. You remember what was said "
                "earlier in the conversation."
            ),
        }
    ]

    # Simulated conversation turns
    user_turns = [
        "Hi! My name is Alex and I'm studying for the AI-102 exam.",
        "What topics does Domain 1 cover?",
        "Can you remind me what my name is?",  # Tests if model remembers earlier context
    ]

    for user_message in user_turns:
        # Add user message to history
        messages.append({"role": "user", "content": user_message})
        print(f"\n  User: {user_message}")

        # Send full conversation history
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )

        assistant_reply = response.choices[0].message.content
        print(f"  Assistant: {assistant_reply}")

        # Add assistant response to history for next turn
        messages.append({"role": "assistant", "content": assistant_reply})

    print(f"\n  Total messages in history: {len(messages)}")
    print(f"  Final token usage: {response.usage.total_tokens} tokens")


# ---------------------------------------------------------------------------
# 4. Streaming responses
# ---------------------------------------------------------------------------

def streaming_completion(client: openai.AzureOpenAI) -> None:
    """
    Stream the response token by token instead of waiting for the full response.

    Streaming is important for:
      - Better perceived performance in chat UIs
      - Showing progress for long responses
      - Stopping generation early if needed

    With stream=True, the response is an iterator of chunks (SSE events).
    Each chunk has a delta with the incremental content.
    """
    print("\n--- Streaming Response ---")
    print("(tokens appear as they are generated)\n")
    print("Response: ", end="", flush=True)

    stream = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List 5 key Azure AI Services with a one-sentence description each."},
        ],
        max_tokens=300,
        temperature=0.5,
        stream=True,  # Enable streaming
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)  # Print each token as it arrives
            full_response += token

    print("\n")  # Newline after streaming completes
    print(f"  [Stream complete] Total characters: {len(full_response)}")


# ---------------------------------------------------------------------------
# 5. Code generation
# ---------------------------------------------------------------------------

def code_generation_demo(client: openai.AzureOpenAI) -> None:
    """
    Demonstrate using chat completions for code generation tasks.
    Shows how to request specific programming language output.
    """
    print("\n--- Code Generation Demo ---")

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer. Respond with only clean, "
                    "well-commented Python code. No explanations outside the code."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write a Python function that calls the Azure OpenAI API to "
                    "classify text sentiment as positive, negative, or neutral. "
                    "Use the openai package. Include type hints."
                ),
            },
        ],
        max_tokens=500,
        temperature=0.2,  # Lower temperature for more deterministic code output
    )

    code = response.choices[0].message.content
    print(f"Generated code:\n{code}")


def main():
    print("=" * 60)
    print("Azure OpenAI Chat Completions Demo")
    print("=" * 60)
    print(f"Endpoint  : {ENDPOINT}")
    print(f"Deployment: {DEPLOYMENT}")

    try:
        client = get_client()

        # 1. Basic completion
        basic_completion(client)

        # 2. System message design
        system_message_demo(client)

        # 3. Multi-turn conversation
        multi_turn_conversation(client)

        # 4. Streaming
        streaming_completion(client)

        # 5. Code generation
        code_generation_demo(client)

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except openai.AuthenticationError as e:
        print(f"\n[ERROR] Authentication failed: {e}")
        print("Check AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in your .env file.")
    except openai.NotFoundError as e:
        print(f"\n[ERROR] Deployment not found: {e}")
        print(f"Ensure deployment '{DEPLOYMENT}' exists in your Azure OpenAI resource.")
    except openai.APIError as e:
        print(f"\n[ERROR] OpenAI API error: {e}")


if __name__ == "__main__":
    main()
