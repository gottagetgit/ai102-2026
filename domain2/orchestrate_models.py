"""
orchestrate_models.py
=====================
Demonstrates orchestrating multiple Azure AI models together to solve
complex, multi-step tasks that a single model cannot handle alone.

Exam Skill: "Orchestrate interactions with multiple Azure AI models"
            (Domain 2 - Implement generative AI solutions)

What this demo covers:
  - Using a router/dispatcher pattern to direct queries to specialist models
  - Chaining model outputs - the output of one model feeds the next
  - Using GPT-4o as an orchestrator that calls specialist models via functions
  - Parallel fan-out: sending the same request to multiple models, then merging
  - Fallback chains: try a cheaper model first, escalate if confidence is low
  - Building a simple RAG pipeline by orchestrating embedding + chat models

Models orchestrated in this demo:
  - gpt-4o-mini   : fast/cheap model for simple tasks and routing decisions
  - gpt-4o        : high-quality model for complex reasoning
  - text-embedding-3-small : embedding model for semantic search
  - dall-e-3      : image generation (last demo)

Required packages:
  pip install openai numpy python-dotenv

Required environment variables:
  AZURE_OPENAI_ENDPOINT           e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY                API key
  AZURE_OPENAI_CHAT_DEPLOYMENT    GPT-4o deployment name
  AZURE_OPENAI_MINI_DEPLOYMENT    GPT-4o-mini deployment name  (optional)
  AZURE_OPENAI_EMBED_DEPLOYMENT   text-embedding-3-small deployment (optional)
  AZURE_OPENAI_IMAGE_DEPLOYMENT   dall-e-3 deployment name (optional)
"""

import os
import json
import math
import time
from typing import Any
from dotenv import load_dotenv
import openai

load_dotenv()

ENDPOINT         = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY          = os.environ["AZURE_OPENAI_KEY"]
CHAT_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
MINI_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-4o-mini")
EMBED_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
IMAGE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_IMAGE_DEPLOYMENT", "dall-e-3")


def get_client() -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version="2024-12-01-preview",
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def chat(client: openai.AzureOpenAI, deployment: str, messages: list,
         temperature: float = 0.3, max_tokens: int = 600,
         tools: list | None = None, tool_choice: str = "auto") -> openai.types.chat.ChatCompletion:
    """Thin wrapper around chat.completions.create."""
    kwargs: dict[str, Any] = dict(
        model=deployment,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# ---------------------------------------------------------------------------
# Demo 1 - Router / Dispatcher pattern
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = """
You are a query router. Classify the user's question into one of these categories:
  - MATH      : arithmetic, algebra, geometry, statistics
  - CODE      : programming, debugging, algorithms, software
  - CREATIVE  : writing, poetry, storytelling, brainstorming
  - FACTUAL   : history, science, geography, general knowledge
  - OTHER     : anything else

Reply with ONLY the category label, nothing else.
"""

SPECIALIST_SYSTEM = {
    "MATH":     "You are an expert mathematician. Show your working clearly.",
    "CODE":     "You are a senior software engineer. Provide clear, well-commented code.",
    "CREATIVE": "You are a creative writing expert. Be vivid and imaginative.",
    "FACTUAL":  "You are a knowledgeable tutor. Be accurate and concise.",
    "OTHER":    "You are a helpful assistant.",
}


def demo_router(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 1: Router / Dispatcher pattern ===")

    queries = [
        "What is 17% of 340?",
        "Write a haiku about autumn rain.",
        "How do I reverse a linked list in Python?",
        "Who painted the Sistine Chapel ceiling?",
    ]

    for query in queries:
        route_resp = chat(client, MINI_DEPLOYMENT,
                          [{"role": "system", "content": ROUTER_SYSTEM},
                           {"role": "user",   "content": query}],
                          temperature=0, max_tokens=10)
        category = route_resp.choices[0].message.content.strip().upper()
        if category not in SPECIALIST_SYSTEM:
            category = "OTHER"

        system_prompt = SPECIALIST_SYSTEM[category]
        answer_resp = chat(client, CHAT_DEPLOYMENT,
                           [{"role": "system", "content": system_prompt},
                            {"role": "user",   "content": query}],
                           max_tokens=300)
        answer = answer_resp.choices[0].message.content.strip()

        print(f"\nQuery   : {query}")
        print(f"Category: {category}")
        print(f"Answer  : {answer[:200]}{'...' if len(answer) > 200 else ''}")


# ---------------------------------------------------------------------------
# Demo 2 - Chain-of-model pipeline
# ---------------------------------------------------------------------------
def demo_chain(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 2: Chain-of-model pipeline ===")

    feedback = (
        "I've been waiting 3 weeks for my order and nobody has contacted me. "
        "The tracking page just says 'processing'. I'm really frustrated and "
        "I might cancel if this isn't resolved today."
    )
    print(f"Customer feedback: {feedback}\n")

    sentiment_resp = chat(
        client, MINI_DEPLOYMENT,
        [{"role": "system", "content": "Classify sentiment as POSITIVE, NEUTRAL, or NEGATIVE. One word only."},
         {"role": "user",   "content": feedback}],
        temperature=0, max_tokens=5,
    )
    sentiment = sentiment_resp.choices[0].message.content.strip()
    print(f"Stage 1 - Sentiment: {sentiment}")

    issues_resp = chat(
        client, MINI_DEPLOYMENT,
        [{"role": "system", "content": "Extract the key issues from the feedback as a brief bullet list."},
         {"role": "user",   "content": feedback}],
        max_tokens=150,
    )
    issues = issues_resp.choices[0].message.content.strip()
    print(f"Stage 2 - Issues:\n{issues}")

    context = (
        f"Customer feedback sentiment: {sentiment}\n"
        f"Key issues identified:\n{issues}\n\n"
        f"Original feedback: {feedback}"
    )
    response_resp = chat(
        client, CHAT_DEPLOYMENT,
        [{"role": "system", "content": "You are a senior customer service manager. Write a professional, empathetic response."},
         {"role": "user",   "content": f"Write a response to this customer:\n{context}"}],
        max_tokens=400,
    )
    response_text = response_resp.choices[0].message.content.strip()
    print(f"Stage 3 - Response:\n{response_text}")


# ---------------------------------------------------------------------------
# Demo 3 - GPT-4o as orchestrator using function-calling
# ---------------------------------------------------------------------------
SPECIALIST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_math_expert",
            "description": "Ask a mathematics expert to solve a calculation or math problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string", "description": "The math problem to solve"}
                },
                "required": ["problem"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_code_expert",
            "description": "Ask a software engineering expert to write or review code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "The coding task or question"}
                },
                "required": ["request"],
            },
        },
    },
]


def call_specialist(client: openai.AzureOpenAI, tool_name: str, args: dict) -> str:
    if tool_name == "ask_math_expert":
        resp = chat(client, CHAT_DEPLOYMENT,
                    [{"role": "system", "content": "You are an expert mathematician."},
                     {"role": "user",   "content": args["problem"]}],
                    max_tokens=300)
    elif tool_name == "ask_code_expert":
        resp = chat(client, CHAT_DEPLOYMENT,
                    [{"role": "system", "content": "You are a senior software engineer."},
                     {"role": "user",   "content": args["request"]}],
                    max_tokens=400)
    else:
        return "Unknown specialist"
    return resp.choices[0].message.content.strip()


def demo_orchestrator(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 3: GPT-4o as orchestrator ===")

    user_request = (
        "I need two things: (1) calculate the compound interest on $5000 at 4.5% annual rate "
        "compounded monthly for 3 years, and (2) write a Python function to do this calculation "
        "for any principal, rate, frequency, and years."
    )
    print(f"User request: {user_request}\n")

    messages = [
        {"role": "system", "content": "You are an orchestrator. Use the available tools to fulfil complex requests."},
        {"role": "user",   "content": user_request},
    ]

    for _ in range(5):
        resp = chat(client, CHAT_DEPLOYMENT, messages, tools=SPECIALIST_TOOLS)
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump(exclude_unset=True))
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = call_specialist(client, tc.function.name, args)
                print(f"Called {tc.function.name}: {result[:150]}...")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            print(f"Final answer:\n{msg.content}")
            break


# ---------------------------------------------------------------------------
# Demo 4 - Parallel fan-out + merge
# ---------------------------------------------------------------------------
def demo_fan_out(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 4: Parallel fan-out + merge ===")

    topic = "The impact of artificial intelligence on employment over the next decade"
    perspectives = [
        ("optimist",  "You are an optimistic economist who focuses on job creation and productivity gains."),
        ("pessimist", "You are a cautious sociologist focused on job displacement and inequality."),
        ("pragmatist","You are a pragmatic policy advisor focused on retraining and transition strategies."),
    ]

    results = {}
    for name, system_prompt in perspectives:
        resp = chat(client, MINI_DEPLOYMENT,
                    [{"role": "system", "content": system_prompt},
                     {"role": "user",   "content": f"In 2-3 sentences, share your view on: {topic}"}],
                    max_tokens=150)
        results[name] = resp.choices[0].message.content.strip()
        print(f"\n{name.upper()} view: {results[name]}")

    synthesis_prompt = "\n\n".join(
        f"{name.upper()}: {text}" for name, text in results.items()
    )
    synth_resp = chat(client, CHAT_DEPLOYMENT,
                      [{"role": "system", "content": "Synthesise the following perspectives into a balanced 3-sentence summary."},
                       {"role": "user",   "content": synthesis_prompt}],
                      max_tokens=200)
    print(f"\nSYNTHESIS: {synth_resp.choices[0].message.content.strip()}")


# ---------------------------------------------------------------------------
# Demo 5 - Fallback chain
# ---------------------------------------------------------------------------
def demo_fallback_chain(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 5: Fallback escalation chain ===")

    questions = [
        "What is 12 * 8?",
        "Explain the concept of quantum entanglement and its implications for cryptography.",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")

        mini_resp = chat(client, MINI_DEPLOYMENT,
                         [{"role": "system", "content":
                           "Answer the question, then on a new line write CONFIDENCE: HIGH or CONFIDENCE: LOW."},
                          {"role": "user", "content": q}],
                         max_tokens=200)
        mini_text = mini_resp.choices[0].message.content.strip()
        print(f"  Mini model: {mini_text[:120]}")

        if "CONFIDENCE: LOW" in mini_text:
            print("  -> Low confidence detected, escalating to GPT-4o...")
            full_resp = chat(client, CHAT_DEPLOYMENT,
                             [{"role": "system", "content": "You are a world-class expert. Answer thoroughly."},
                              {"role": "user", "content": q}],
                             max_tokens=400)
            print(f"  GPT-4o: {full_resp.choices[0].message.content.strip()[:200]}")
        else:
            print("  -> High confidence, using mini model answer.")


# ---------------------------------------------------------------------------
# Demo 6 - Minimal RAG pipeline
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    "Azure AI Search supports vector, keyword, and hybrid search.",
    "Azure OpenAI Service provides access to GPT-4o, GPT-4o-mini, and embedding models.",
    "Prompt flow in Azure AI Foundry lets you build and evaluate LLM pipelines visually.",
    "Azure AI Content Safety detects harmful content in text and images.",
    "The AI-102 exam covers six domains including Generative AI and Computer Vision.",
    "Azure Document Intelligence extracts structured data from forms and invoices.",
    "Responsible AI principles include fairness, reliability, privacy, and transparency.",
    "Azure AI Speech supports both speech-to-text (STR) and text-to-speech (TTS).",
]


def demo_rag(client: openai.AzureOpenAI) -> None:
    print("\n=== Demo 6: Minimal RAG pipeline ===")

    query = "What search types does Azure AI Search support?"
    print(f"Query: {query}")

    q_embed = client.embeddings.create(
        model=EMBED_DEPLOYMENT, input=query
    ).data[0].embedding

    kb_embeddings = [
        client.embeddings.create(model=EMBED_DEPLOYMENT, input=doc).data[0].embedding
        for doc in KNOWLEDGE_BASE
    ]

    scored = sorted(
        enumerate(KNOWLEDGE_BASE),
        key=lambda item: cosine_similarity(q_embed, kb_embeddings[item[0]]),
        reverse=True,
    )
    top_docs = [doc for _, doc in scored[:3]]
    print(f"Top retrieved chunks: {top_docs}")

    context = "\n".join(f"- {d}" for d in top_docs)
    resp = chat(client, CHAT_DEPLOYMENT,
                [{"role": "system", "content": "Answer using only the provided context. If the context doesn't contain the answer, say so."},
                 {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}],
                max_tokens=200)
    print(f"Answer: {resp.choices[0].message.content.strip()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    client = get_client()
    demo_router(client)
    demo_chain(client)
    demo_orchestrator(client)
    demo_fan_out(client)
    demo_fallback_chain(client)
    demo_rag(client)
    print("\n=== Orchestration demos complete ===")


if __name__ == "__main__":
    main()
