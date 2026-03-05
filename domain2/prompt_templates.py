"""
prompt_templates.py
===================
Demonstrates using prompt templates with variable substitution for
building reusable, maintainable prompt patterns.

Exam Skill: "Utilize prompt templates in your generative AI solution"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - Simple string format() based templates
  - Jinja2 templates (used by PromptFlow and LangChain)
  - Multi-role templates (system + user)
  - Templates with loops and conditionals (Jinja2)
  - Template versioning and management
  - PromptFlow Jinja2 template format
  - Template loading from files
  - Practical template library for common AI tasks

Templates are essential for:
  - Keeping prompts consistent across runs
  - Enabling A/B testing different prompt versions
  - Making prompts auditable and version-controlled
  - Separating prompt logic from application logic

Required packages:
  pip install openai jinja2 python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT    - e.g. https://<name>.openai.azure.com/
  AZURE_OPENAI_KEY         - API key
  AZURE_OPENAI_DEPLOYMENT  - Deployment name e.g. "gpt-4o"
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Template, Environment, FileSystemLoader
import openai

load_dotenv()

ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

TEMPLATE_DIR = Path("./prompt_templates")


def get_client() -> openai.AzureOpenAI:
    """Return authenticated AzureOpenAI client."""
    return openai.AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version="2024-12-01-preview",
    )


def complete(client: openai.AzureOpenAI, system: str, user: str, **kwargs) -> str:
    """Helper: send a single-turn completion and return the response text."""
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=kwargs.pop("max_tokens", 300),
        temperature=kwargs.pop("temperature", 0.7),
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Method 1: Python str.format() templates
# ---------------------------------------------------------------------------

class SimplePromptTemplate:
    """
    A basic prompt template using Python's str.format().
    Simple, no dependencies, good for straightforward use cases.

    Variable syntax: {variable_name}
    """

    def __init__(self, system_template: str, user_template: str):
        self.system_template = system_template
        self.user_template   = user_template

    def render(self, **variables) -> tuple[str, str]:
        """Substitute variables and return (system_message, user_message)."""
        try:
            system = self.system_template.format(**variables)
            user   = self.user_template.format(**variables)
            return system, user
        except KeyError as e:
            raise ValueError(f"Template variable {e} not provided") from e

    def __repr__(self):
        return f"SimplePromptTemplate(vars={self._extract_vars()})"

    def _extract_vars(self) -> list:
        import re
        vars_in_system = re.findall(r"\{(\w+)\}", self.system_template)
        vars_in_user   = re.findall(r"\{(\w+)\}", self.user_template)
        return list(set(vars_in_system + vars_in_user))


# ---------------------------------------------------------------------------
# Method 2: Jinja2 templates (more powerful)
# ---------------------------------------------------------------------------

class Jinja2PromptTemplate:
    """
    A prompt template using Jinja2.
    Supports variables, loops, conditionals, and filters.

    Variable syntax: {{ variable_name }}
    Conditional:     {% if condition %}...{% endif %}
    Loop:            {% for item in items %}...{% endfor %}
    Filter:          {{ text | upper }}
    """

    def __init__(self, system_template: str, user_template: str):
        self.system_tmpl = Template(system_template)
        self.user_tmpl   = Template(user_template)

    def render(self, **variables) -> tuple[str, str]:
        """Substitute variables and return (system_message, user_message)."""
        system = self.system_tmpl.render(**variables)
        user   = self.user_tmpl.render(**variables)
        return system, user


# ---------------------------------------------------------------------------
# Template library - practical templates for common AI tasks
# ---------------------------------------------------------------------------

TEMPLATE_LIBRARY = {

    # --- Summarization template ---
    "summarize": SimplePromptTemplate(
        system_template=(
            "You are a summarization assistant. "
            "Produce summaries that are {style} and approximately {length} long."
        ),
        user_template=(
            "Summarize the following text:\n\n{text}\n\n"
            "Focus on: {focus_areas}"
        ),
    ),

    # --- Classification template ---
    "classify": SimplePromptTemplate(
        system_template=(
            "You are a text classification assistant. "
            "Classify text into exactly one of these categories: {categories}. "
            "Respond with the category name only."
        ),
        user_template="Classify: {text}",
    ),

    # --- Q&A with context (RAG template) ---
    "qa_with_context": Jinja2PromptTemplate(
        system_template=(
            "You are a helpful assistant. Answer questions using ONLY the provided context.\n"
            "If the context doesn't contain the answer, say \"I don't have enough information.\"\n"
            "{% if language %}Respond in {{ language }}.{% endif %}"
        ),
        user_template=(
            "Context:\n"
            "{% for i, doc in enumerate(documents) %}"
            "[Source {{ i+1 }}] {{ doc }}\n"
            "{% endfor %}\n"
            "Question: {{ question }}"
        ),
    ),

    # --- Jinja2 Q&A with context (corrected enumerate approach) ---
    "qa_rag": Jinja2PromptTemplate(
        system_template=(
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "Cite sources as [Source N]."
        ),
        user_template=(
            "Context:\n"
            "{% for doc in documents %}"
            "[Source {{ loop.index }}] {{ doc }}\n\n"
            "{% endfor %}"
            "\nQuestion: {{ question }}"
        ),
    ),

    # --- Code review template ---
    "code_review": SimplePromptTemplate(
        system_template=(
            "You are an expert {language} developer performing a code review. "
            "Review for: correctness, security, performance, and best practices. "
            "Be constructive and specific."
        ),
        user_template="Review this {language} code:\n\n```{language}\n{code}\n```",
    ),

    # --- Translation template ---
    "translate": SimplePromptTemplate(
        system_template=(
            "You are a professional translator. "
            "Translate from {source_language} to {target_language}. "
            "Maintain the original tone and style."
        ),
        user_template="Translate:\n\n{text}",
    ),

    # --- Entity extraction template ---
    "extract_entities": Jinja2PromptTemplate(
        system_template=(
            "You are a data extraction expert. "
            "Extract entities from text and return valid JSON only. "
            "Use null for missing fields."
        ),
        user_template=(
            "Extract the following fields from the text:\n"
            "Fields: {{ fields | join(', ') }}\n\n"
            "Text: {{ text }}\n\n"
            "Return a JSON object with exactly these keys: {{ fields | join(', ') }}"
        ),
    ),
}


def save_templates_to_files() -> None:
    """
    Save templates as Jinja2 .jinja2 files that can be loaded by PromptFlow.
    This pattern separates prompt logic from Python code.
    """
    TEMPLATE_DIR.mkdir(exist_ok=True)

    templates = {
        "summarize.jinja2": """system:
You are a summarization assistant.
Produce summaries that are {{ style | default('concise') }} and approximately {{ length | default('2-3 sentences') }} long.

user:
Summarize the following text:

{{ text }}

{% if focus_areas %}Focus on: {{ focus_areas }}{% endif %}
""",

        "classify.jinja2": """system:
You are a text classification assistant.
Classify text into exactly one of these categories: {{ categories | join(', ') }}.
Respond with the category name only, nothing else.

user:
Classify: {{ text }}
""",

        "qa_with_context.jinja2": """system:
You are a helpful assistant. Answer questions using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information."
{% if language %}Respond in {{ language }}.{% endif %}

user:
Context:
{% for doc in documents %}
[Source {{ loop.index }}] {{ doc.title | default('Document') }}:
{{ doc.content }}

{% endfor %}
Question: {{ question }}
""",

        "code_review.jinja2": """system:
You are an expert {{ language | default('Python') }} developer performing a code review.
Review for: correctness, security, performance, and best practices.
Format your review with sections: Issues, Suggestions, Positive observations.

user:
Review this {{ language | default('Python') }} code:

```{{ language | default('python') }}
{{ code }}
```
""",
    }

    for filename, content in templates.items():
        path = TEMPLATE_DIR / filename
        path.write_text(content)
        print(f"  Saved: {path}")


def load_and_render_jinja2_file(template_file: str, **variables) -> tuple[str, str]:
    """
    Load a Jinja2 template from a file and render it.
    Parses the PromptFlow-style format with system:/user: sections.
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template(template_file)
    rendered = template.render(**variables)

    # Parse system/user sections
    system_content = ""
    user_content   = ""
    current_role   = None
    current_lines  = []

    for line in rendered.splitlines():
        stripped = line.strip().lower()
        if stripped == "system:":
            current_role  = "system"
            current_lines = []
        elif stripped == "user:":
            if current_role == "system":
                system_content = "\n".join(current_lines).strip()
            current_role  = "user"
            current_lines = []
        else:
            current_lines.append(line)

    if current_role == "user":
        user_content = "\n".join(current_lines).strip()
    elif current_role == "system" and not user_content:
        system_content = "\n".join(current_lines).strip()

    return system_content, user_content


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_simple_templates(client: openai.AzureOpenAI) -> None:
    """Demo: Using simple string.format() templates."""
    print("\n--- Demo: Simple Templates (str.format) ---")

    # Summarization
    template = TEMPLATE_LIBRARY["summarize"]
    system, user = template.render(
        style="clear and professional",
        length="2-3 sentences",
        text=(
            "Azure AI Content Safety is a service that uses AI algorithms to flag potentially "
            "offensive, risky, or undesirable content. The service analyzes text and images "
            "across four categories: hate speech, violence, self-harm, and sexual content. "
            "Each category has a severity level from 0 to 6."
        ),
        focus_areas="key functionality and use cases",
    )
    print(f"\n  Template: summarize")
    print(f"  System: {system}")
    print(f"  User: {user[:100]}...")
    result = complete(client, system, user, max_tokens=100, temperature=0.3)
    print(f"  Result: {result}")

    # Classification
    template = TEMPLATE_LIBRARY["classify"]
    system, user = template.render(
        categories="Technical, Billing, General, Complaint",
        text="My API key is not working, I'm getting 401 errors.",
    )
    print(f"\n  Template: classify")
    result = complete(client, system, user, max_tokens=10, temperature=0.0)
    print(f"  Category: {result}")


def demo_jinja2_templates(client: openai.AzureOpenAI) -> None:
    """Demo: Using Jinja2 templates with loops and conditionals."""
    print("\n--- Demo: Jinja2 Templates (with loops & conditionals) ---")

    # Entity extraction with dynamic field list
    template = TEMPLATE_LIBRARY["extract_entities"]
    system, user = template.render(
        fields=["name", "company", "email", "phone", "role"],
        text=(
            "Hi, I'm James Wilson, a Senior Cloud Architect at Northwind Traders. "
            "You can reach me at j.wilson@northwind.com or 555-234-5678."
        ),
    )
    print(f"  Template: extract_entities")
    result = complete(
        client, system, user,
        max_tokens=200, temperature=0.0,
        response_format={"type": "json_object"},
    )
    print(f"  Extracted: {result}")

    # Q&A with dynamic document list
    template = TEMPLATE_LIBRARY["qa_rag"]
    system, user = template.render(
        question="What authentication methods does Azure AI support?",
        documents=[
            "Azure AI Services supports API key authentication and Microsoft Entra ID (keyless) authentication.",
            "For production, Microsoft recommends using managed identity which automatically handles credential rotation.",
            "DefaultAzureCredential from azure-identity tries multiple credential sources in order.",
        ],
    )
    print(f"\n  Template: qa_rag")
    result = complete(client, system, user, max_tokens=150, temperature=0.3)
    print(f"  Answer: {result}")


def demo_file_templates(client: openai.AzureOpenAI) -> None:
    """Demo: Loading templates from .jinja2 files (PromptFlow compatible format)."""
    print("\n--- Demo: File-Based Templates (Jinja2 files) ---")

    # Save templates first
    print("  Saving template files...")
    save_templates_to_files()

    # Load and use classify template
    print("\n  Loading classify.jinja2 from file:")
    system, user = load_and_render_jinja2_file(
        "classify.jinja2",
        categories=["Positive", "Negative", "Neutral"],
        text="The deployment went smoothly but the documentation was confusing.",
    )
    print(f"  System: {system}")
    print(f"  User: {user}")
    result = complete(client, system, user, max_tokens=10, temperature=0.0)
    print(f"  Classification: {result}")


def demo_template_versioning() -> None:
    """
    Demonstrate a simple template versioning system.
    In production, use a proper template store (database, Key Vault, or Git).
    """
    print("\n--- Demo: Template Versioning ---")

    # Template versions stored as a dict (in production: use a database or config)
    template_versions = {
        "summarize_v1": {
            "version": "1.0",
            "system": "You are a summarization assistant.",
            "user": "Summarize in 2 sentences: {text}",
            "notes": "Initial version",
        },
        "summarize_v2": {
            "version": "2.0",
            "system": "You are a summarization assistant. Focus on actionable insights.",
            "user": "Summarize in 2 sentences, highlighting key takeaways: {text}",
            "notes": "Added actionable focus based on user feedback",
        },
    }

    # Save to a JSON file (version registry)
    TEMPLATE_DIR.mkdir(exist_ok=True)
    registry_path = TEMPLATE_DIR / "template_registry.json"
    registry_path.write_text(json.dumps(template_versions, indent=2))

    print(f"  Template registry saved to {registry_path}")
    print(f"  Available versions:")
    for name, tmpl in template_versions.items():
        print(f"    {name} (v{tmpl['version']}): {tmpl['notes']}")

    print("\n  Best practice: Store templates in version control (Git)")
    print("  A/B test template versions by splitting traffic between them")
    print("  Use PromptFlow evaluation to compare template quality metrics")


def main():
    print("=" * 60)
    print("Prompt Templates Demo")
    print("=" * 60)
    print(f"Deployment: {DEPLOYMENT}")

    try:
        client = get_client()

        demo_simple_templates(client)
        demo_jinja2_templates(client)
        demo_file_templates(client)
        demo_template_versioning()

        print("\n" + "=" * 60)
        print("Summary: Prompt Template Approaches")
        print("=" * 60)
        print("""
  Approach               | Best For
  -----------------------|-----------------------------------------------
  str.format()           | Simple templates, no dependencies
  Jinja2                 | Loops, conditionals, filters
  .jinja2 files          | Version-controlled, PromptFlow-compatible
  YAML with templates    | Azure AI Foundry / PromptFlow deployments
  Database/Key Vault     | Production, audit trail, A/B testing

  Template variable tips:
  - Use descriptive variable names: {customer_name} not {n}
  - Provide defaults for optional variables: {{ var | default('value') }}
  - Validate required variables before rendering
  - Version your templates alongside your code
""")

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except openai.APIError as e:
        print(f"\n[ERROR] Azure OpenAI error: {e}")


if __name__ == "__main__":
    main()
