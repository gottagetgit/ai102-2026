"""
prompt_templates.py
===================
Demonstrates building reusable, parameterised prompt templates for
Azure OpenAI models, including Jinja2 templates used by Prompt flow.

Exam Skill: "Apply prompt engineering techniques" /
            "Use Azure AI Foundry Prompt flow"
            (Domain 2 - Implement generative AI solutions)

This file shows:
  1. Simple f-string templates with variable substitution
  2. Dataclass-backed prompt templates with validation
  3. Jinja2 templates (the format used by Prompt flow LLM nodes)
  4. Multi-turn chat templates with chat_history
  5. System/user/assistant templates for structured interactions
  6. Template library - a registry of reusable templates
  7. Rendering templates and calling Azure OpenAI
  8. Prompt versioning & A/B test helper

Required packages:
  pip install openai jinja2 python-dotenv

Required environment variables:
  AZURE_OPENAI_ENDPOINT    - https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY         - API key
  AZURE_OPENAI_DEPLOYMENT  - GPT-4o deployment name
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
import openai
from jinja2 import Environment, BaseLoader, StrictUndefined

load_dotenv()

ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def get_client() -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version="2024-12-01-preview",
    )


# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------
JINJA_ENV = Environment(
    loader=BaseLoader(),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_str: str, **kwargs: Any) -> str:
    """Render a Jinja2 template string with the given variables."""
    return JINJA_ENV.from_string(template_str).render(**kwargs)


# ---------------------------------------------------------------------------
# 1. Simple f-string templates
# ---------------------------------------------------------------------------
def demo_fstring_templates(client: openai.AzureOpenAI) -> None:
    print("\n=== 1. Simple f-string templates ===")

    def summarise_template(text: str, max_words: int = 50) -> str:
        return (
            f"Summarise the following text in at most {max_words} words. "
            f"Return only the summary, no preamble.\n\nText:\n{text}"
        )

    article = (
        "Azure AI Foundry is a unified platform for building enterprise-grade AI applications. "
        "It brings together Azure Machine Learning, Azure OpenAI Service, and various Azure "
        "Cognitive Services under a single project-based workspace. Developers can manage "
        "models, datasets, evaluations, and deployments from one place, making the path from "
        "prototype to production significantly shorter."
    )

    prompt = summarise_template(article, max_words=30)
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0,
    )
    print(f"Summary: {resp.choices[0].message.content.strip()}")


# ---------------------------------------------------------------------------
# 2. Dataclass-backed prompt template
# ---------------------------------------------------------------------------
@dataclass
class PromptTemplate:
    """A reusable prompt template with named variables."""
    name: str
    system: str
    user: str
    required_vars: list[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Render system and user strings. Raises if required vars missing."""
        missing = [v for v in self.required_vars if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables for template '{self.name}': {missing}")
        system_out = render(self.system, **kwargs)
        user_out   = render(self.user,   **kwargs)
        return system_out, user_out

    def call(self, client: openai.AzureOpenAI, temperature: float = 0.3,
             max_tokens: int = 400, **kwargs: Any) -> str:
        """Render and call Azure OpenAI in one step."""
        system_msg, user_msg = self.render(**kwargs)
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 3. Template library
# ---------------------------------------------------------------------------
TEMPLATE_LIBRARY: dict[str, PromptTemplate] = {

    "summarise": PromptTemplate(
        name="summarise",
        system="You are a concise summarisation assistant.",
        user="Summarise the following text in at most {{ max_words }} words:\n\n{{ text }}",
        required_vars=["text", "max_words"],
    ),

    "translate": PromptTemplate(
        name="translate",
        system="You are a professional translator.",
        user="Translate the following text from {{ source_lang }} to {{ target_lang }}.\n\nText:\n{{ text }}",
        required_vars=["text", "source_lang", "target_lang"],
    ),

    "classify": PromptTemplate(
        name="classify",
        system="You are a text classification model. Reply with ONLY the class label.",
        user=(
            "Classes: {{ classes | join(', ') }}\n\n"
            "{% if examples %}Examples:\n"
            "{% for ex in examples %}Input: {{ ex.text }}\nClass: {{ ex.label }}\n{% endfor %}\n{% endif %}"
            "Input: {{ text }}\nClass:"
        ),
        required_vars=["text", "classes"],
    ),

    "extract_json": PromptTemplate(
        name="extract_json",
        system="Extract information as JSON. Return ONLY valid JSON, no markdown.",
        user=(
            "Extract the following fields from the text below.\n"
            "Schema: {{ schema }}\n\n"
            "Text: {{ text }}"
        ),
        required_vars=["text", "schema"],
    ),

    "qa_with_context": PromptTemplate(
        name="qa_with_context",
        system=(
            "You are a question-answering assistant. "
            "Answer using ONLY the information in the context. "
            "If the answer is not in the context, say 'I don't know.'"
        ),
        user="Context:\n{{ context }}\n\nQuestion: {{ question }}",
        required_vars=["context", "question"],
    ),

    "code_review": PromptTemplate(
        name="code_review",
        system="You are a senior {{ language }} developer doing a code review.",
        user=(
            "Review the following {{ language }} code.\n"
            "Focus on: {{ focus | join(', ') }}.\n\n"
            "```{{ language }}\n{{ code }}\n```"
        ),
        required_vars=["code", "language"],
    ),
}


def demo_template_library(client: openai.AzureOpenAI) -> None:
    print("\n=== 2 & 3. Dataclass templates + Template library ===")

    result = TEMPLATE_LIBRARY["summarise"].call(
        client,
        text="Azure AI Foundry unifies Azure OpenAI, Machine Learning, and Cognitive Services into one platform.",
        max_words=20,
    )
    print(f"Summarise: {result}")

    result = TEMPLATE_LIBRARY["translate"].call(
        client,
        text="Hello, how are you?",
        source_lang="English",
        target_lang="Spanish",
    )
    print(f"Translate: {result}")

    result = TEMPLATE_LIBRARY["classify"].call(
        client,
        temperature=0,
        text="My order hasn't arrived and I'm very upset.",
        classes=["Billing", "Shipping", "Technical", "General"],
        examples=[
            {"text": "I was charged twice for my subscription.", "label": "Billing"},
            {"text": "The app crashes when I open it.",          "label": "Technical"},
        ],
    )
    print(f"Classify: {result}")

    result = TEMPLATE_LIBRARY["extract_json"].call(
        client,
        temperature=0,
        text="John Smith, born 15 March 1985, lives at 42 Oak Street, London.",
        schema='{"name": "", "dob": "", "address": ""}',
    )
    print(f"Extract JSON: {result}")

    result = TEMPLATE_LIBRARY["qa_with_context"].call(
        client,
        context="Azure AI Foundry was launched in 2024. It supports GPT-4o, embeddings, and fine-tuning.",
        question="When was Azure AI Foundry launched?",
    )
    print(f"QA: {result}")

    sample_code = """
def get_user(id):
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {id}")
    return cursor.fetchone()
"""
    result = TEMPLATE_LIBRARY["code_review"].call(
        client,
        code=sample_code,
        language="Python",
        focus=["security", "error handling", "best practices"],
    )
    print(f"Code review:\n{result}")


# ---------------------------------------------------------------------------
# 4. Jinja2 template with chat_history (Prompt flow style)
# ---------------------------------------------------------------------------
CHAT_FLOW_TEMPLATE = """
system:
You are a helpful AI assistant specialised in {{ domain }}.

{% for turn in chat_history %}
{{ turn.role }}:
{{ turn.content }}
{% endfor %}
user:
{{ question }}
"""


def demo_chat_history_template(client: openai.AzureOpenAI) -> None:
    print("\n=== 4. Multi-turn chat template with chat_history ===")

    history = [
        {"role": "user",      "content": "What is Azure OpenAI Service?"},
        {"role": "assistant", "content": "Azure OpenAI Service is a cloud offering that provides access to OpenAI's GPT and embedding models via Azure."},
    ]

    rendered = render(
        CHAT_FLOW_TEMPLATE,
        domain="Azure AI services",
        chat_history=history,
        question="What models does it support?",
    )
    print(f"Rendered template:\n{rendered}")

    lines = rendered.strip().split("\n")
    system_content = ""
    user_content = ""
    current = None
    for line in lines:
        if line.strip() == "system:":
            current = "system"
        elif line.strip() == "user:":
            current = "user"
        elif current == "system":
            system_content += line + "\n"
        elif current == "user":
            user_content += line + "\n"

    messages = [
        {"role": "system",    "content": system_content.strip()},
    ] + history + [
        {"role": "user",      "content": user_content.strip()},
    ]

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        max_tokens=200,
        temperature=0.3,
    )
    print(f"Response: {resp.choices[0].message.content.strip()}")


# ---------------------------------------------------------------------------
# 5. Prompt versioning helper
# ---------------------------------------------------------------------------
@dataclass
class VersionedTemplate:
    """Stores multiple versions of a prompt and allows A/B selection."""
    name: str
    versions: dict[str, str]
    default_version: str

    def get(self, version: str | None = None) -> str:
        v = version or self.default_version
        if v not in self.versions:
            raise KeyError(f"Version '{v}' not found in template '{self.name}'")
        return self.versions[v]


SUMMARISE_VERSIONED = VersionedTemplate(
    name="summarise_versioned",
    versions={
        "v1": "Summarise: {{ text }}",
        "v2": "Provide a concise {{ max_words }}-word summary of the following text:\n{{ text }}",
        "v3": (
            "You are a professional editor. Summarise the text below in at most {{ max_words }} words, "
            "preserving the key facts and maintaining a neutral tone.\n\nText:\n{{ text }}"
        ),
    },
    default_version="v3",
)


def demo_versioning(client: openai.AzureOpenAI) -> None:
    print("\n=== 5. Prompt versioning & A/B test ===")

    article = (
        "Azure AI Content Safety is a service that detects harmful content in text and images. "
        "It uses machine learning models trained on large datasets to identify content such as "
        "hate speech, violence, sexual content, and self-harm."
    )

    for version_id in ["v1", "v2", "v3"]:
        template_str = SUMMARISE_VERSIONED.get(version_id)
        rendered = render(template_str, text=article, max_words=20)
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": rendered}],
            max_tokens=60,
            temperature=0,
        )
        result = resp.choices[0].message.content.strip()
        print(f"  {version_id}: {result}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    client = get_client()

    demo_fstring_templates(client)
    demo_template_library(client)
    demo_chat_history_template(client)
    demo_versioning(client)

    print("\n=== Prompt template demos complete ===")


if __name__ == "__main__":
    main()
