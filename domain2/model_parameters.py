"""
model_parameters.py
===================
Demonstrates how each generation parameter affects Azure OpenAI
chat completions output. Covers all key parameters tested in AI-102.

Exam Skill: "Configure Azure OpenAI model parameters"
            (Domain 2 - Implement generative AI solutions)

Parameters covered:
  temperature     - Controls randomness (0=deterministic, 2=very random)
  top_p           - Nucleus sampling (alternative to temperature)
  max_tokens      - Maximum output length
  frequency_penalty - Reduces repetition of frequent tokens
  presence_penalty  - Encourages introducing new topics
  stop            - Custom stop sequences
  seed            - Reproducible outputs (deterministic sampling)
  n               - Number of completions to generate
  logit_bias      - Adjust probability of specific tokens
  response_format - Enforce JSON output

Required packages:
  pip install openai python-dotenv

Required environment variables (in .env):
  AZURE_OPENAI_ENDPOINT    - your Azure OpenAI endpoint
  AZURE_OPENAI_KEY         - your API key
  AZURE_OPENAI_DEPLOYMENT  - deployment name
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-02-01",
)
DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def chat(messages: list, **kwargs) -> str:
    """Helper function for chat completions."""
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        **kwargs
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# temperature
# ---------------------------------------------------------------------------

def demo_temperature() -> None:
    """
    temperature controls output randomness.

    0.0 = Deterministic: always picks the most likely next token.
          Use for: factual Q&A, extraction, classification
    1.0 = Default: balanced creativity/consistency
    2.0 = Maximum: very creative but may be incoherent

    Note: Use EITHER temperature OR top_p, not both.
    Typical exam values: 0 (deterministic), 0.7 (balanced), 1.5+ (creative)
    """
    print("\n" + "=" * 60)
    print("PARAMETER: temperature")
    print("=" * 60)
    print("Effect: Controls output randomness (0=deterministic, 2=random)")
    print("Best for: 0 for factual tasks, 0.7-1.0 for creative tasks")

    prompt = [{"role": "user", "content": "Suggest a creative name for an AI assistant. Just the name, nothing else."}]

    temps = [0.0, 0.5, 1.0, 1.5]
    for temp in temps:
        results = []
        for _ in range(3):  # Run 3 times to show variance
            result = chat(prompt, temperature=temp, max_tokens=20)
            results.append(result.strip())
        print(f"\n  temperature={temp}: {results}")
        if temp == 0.0:
            print("    -> Note: All 3 results are the same (deterministic)")
        elif temp >= 1.5:
            print("    -> Note: High variance in results")


# ---------------------------------------------------------------------------
# top_p
# ---------------------------------------------------------------------------

def demo_top_p() -> None:
    """
    top_p (nucleus sampling) controls diversity by limiting the token
    pool to the top P probability mass.

    1.0 = Consider all tokens (default)
    0.9 = Only consider tokens that make up top 90% of probability
    0.1 = Only consider top 10% - very focused and deterministic

    Note: Don't use both temperature and top_p simultaneously.
    Azure OpenAI recommendation: change one, leave other at default.
    """
    print("\n" + "=" * 60)
    print("PARAMETER: top_p")
    print("=" * 60)
    print("Effect: Nucleus sampling - limits token pool to top P% probability")
    print("Best for: Use INSTEAD of temperature (not both)")

    prompt = [{"role": "user", "content": "Describe clouds in one sentence."}]

    for top_p in [0.1, 0.5, 0.9, 1.0]:
        result = chat(prompt, top_p=top_p, temperature=1, max_tokens=60)
        print(f"\n  top_p={top_p}: {result.strip()}")


# ---------------------------------------------------------------------------
# max_tokens
# ---------------------------------------------------------------------------

def demo_max_tokens() -> None:
    """
    max_tokens limits the maximum length of the generated output.

    Key points:
    - Does NOT guarantee that length - model may stop earlier
    - Truncates at the token limit (may cut mid-sentence)
    - Finish reason will be 'length' if truncated
    - Use to control costs and prevent runaway outputs
    - 1 token ≈ 4 characters in English
    """
    print("\n" + "=" * 60)
    print("PARAMETER: max_tokens")
    print("=" * 60)
    print("Effect: Maximum output length (may truncate response)")

    prompt = [{"role": "user", "content": "Explain what Azure AI Services is in detail."}]

    for max_tok in [20, 50, 200]:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=prompt,
            max_tokens=max_tok,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        finish = response.choices[0].finish_reason
        print(f"\n  max_tokens={max_tok} (finish_reason='{finish}'):")
        print(f"  '{content[:100]}{'...' if len(content) > 100 else ''}'")
        if finish == "length":
            print(f"  -> Response was TRUNCATED at token limit")
        else:
            print(f"  -> Response completed naturally")


# ---------------------------------------------------------------------------
# frequency_penalty
# ---------------------------------------------------------------------------

def demo_frequency_penalty() -> None:
    """
    frequency_penalty reduces repetition of tokens that have already appeared.

    Range: -2.0 to 2.0
    0.0 = No penalty (default)
    Positive values: reduce word repetition
    Negative values: increase word repetition

    Applied based on HOW MANY TIMES a token has appeared.
    Each occurrence adds to the penalty.

    Use case: Long-form content generation where you want variety.
    """
    print("\n" + "=" * 60)
    print("PARAMETER: frequency_penalty")
    print("=" * 60)
    print("Effect: Penalizes tokens based on frequency of past use")
    print("Range: -2.0 to 2.0 (positive = less repetition)")

    prompt = [{"role": "user", "content": "Write 3 sentences about cloud computing."}]

    for penalty in [0.0, 1.0, 2.0]:
        result = chat(prompt, frequency_penalty=penalty, temperature=0.7, max_tokens=150)
        print(f"\n  frequency_penalty={penalty}:")
        print(f"  {result.strip()[:200]}")


# ---------------------------------------------------------------------------
# presence_penalty
# ---------------------------------------------------------------------------

def demo_presence_penalty() -> None:
    """
    presence_penalty encourages the model to talk about new topics.

    Range: -2.0 to 2.0
    0.0 = No penalty (default)
    Positive values: encourage new topics (diversity)
    Negative values: stick to topics already mentioned

    Applied based on WHETHER a token has appeared (not how many times).
    One-time penalty on any token that has been used at all.

    frequency_penalty vs presence_penalty:
    - frequency_penalty: Penalizes EACH occurrence (grows with repetition)
    - presence_penalty: Penalizes if token has EVER appeared (one-time)
    """
    print("\n" + "=" * 60)
    print("PARAMETER: presence_penalty")
    print("=" * 60)
    print("Effect: Penalizes tokens that have appeared at all (encourages new topics)")
    print("Range: -2.0 to 2.0 (positive = more topic diversity)")

    prompt = [{"role": "user", "content": "Write a paragraph about technology."}]

    for penalty in [0.0, 1.0, 2.0]:
        result = chat(prompt, presence_penalty=penalty, temperature=0.7, max_tokens=150)
        print(f"\n  presence_penalty={penalty}:")
        print(f"  {result.strip()[:200]}")


# ---------------------------------------------------------------------------
# stop sequences
# ---------------------------------------------------------------------------

def demo_stop_sequences() -> None:
    """
    stop sequences cause the model to stop generating at specified strings.

    - Can specify up to 4 stop sequences
    - Model stops BEFORE generating the stop sequence (not included in output)
    - Useful for: structured output, preventing overrun, controlled extraction
    - Examples: stop at newline, stop at specific delimiter
    """
    print("\n" + "=" * 60)
    print("PARAMETER: stop sequences")
    print("=" * 60)
    print("Effect: Stop generation when a specified string is encountered")

    # Example: extract only the first item from a numbered list
    prompt = [{
        "role": "user",
        "content": "List 5 Azure AI services, one per line, numbered:"
    }]

    # Without stop sequence - gets all 5
    result_all = chat(prompt, max_tokens=200, temperature=0.3)
    print(f"\n  Without stop sequence (all items):")
    print(f"  {result_all.strip()[:200]}")

    # With stop sequence - stops after first item
    result_one = chat(prompt, stop=["2.", "\n2"], max_tokens=200, temperature=0.3)
    print(f"\n  With stop=['2.', '\\n2'] (stops after first item):")
    print(f"  {result_one.strip()}")


# ---------------------------------------------------------------------------
# seed for reproducibility
# ---------------------------------------------------------------------------

def demo_seed() -> None:
    """
    seed enables reproducible outputs.

    When seed is set:
    - Same prompt + same seed = same (or very similar) output
    - Useful for testing, debugging, regression testing
    - Not 100% guaranteed (system_fingerprint may change)

    Check system_fingerprint in response to verify same backend version.
    """
    print("\n" + "=" * 60)
    print("PARAMETER: seed")
    print("=" * 60)
    print("Effect: Enables reproducible outputs with same seed value")

    prompt = [{"role": "user", "content": "Give me a random number between 1 and 100."}]
    seed_value = 42

    print(f"\n  Running same prompt 3 times with seed={seed_value}:")
    for i in range(3):
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=prompt,
            seed=seed_value,
            temperature=1.0,
            max_tokens=20,
        )
        result = response.choices[0].message.content.strip()
        fingerprint = response.system_fingerprint
        print(f"  Run {i+1}: '{result}' (fingerprint: {fingerprint})")

    print("  -> Results should be identical (same seed = reproducible)")


# ---------------------------------------------------------------------------
# n - multiple completions
# ---------------------------------------------------------------------------

def demo_n_completions() -> None:
    """
    n generates multiple completion choices in a single API call.

    - Default: n=1
    - Higher n = more choices but proportionally more tokens/cost
    - Access via response.choices[i]
    - Useful for: A/B testing prompts, finding best completion
    """
    print("\n" + "=" * 60)
    print("PARAMETER: n (multiple completions)")
    print("=" * 60)
    print("Effect: Generate multiple choices in one API call")

    prompt = [{"role": "user", "content": "Complete: 'Azure AI is like...' in one sentence."}]

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=prompt,
        n=3,
        temperature=0.9,
        max_tokens=60,
    )

    print(f"\n  Generated {len(response.choices)} completions:")
    for i, choice in enumerate(response.choices):
        print(f"  [{i+1}] {choice.message.content.strip()}")

    print(f"\n  Total tokens: {response.usage.total_tokens}")
    print(f"  (Roughly {response.usage.total_tokens // 3} per completion)")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_parameter_summary() -> None:
    """
    Print a summary table of all parameters for quick reference.
    Key exam reference.
    """
    print("\n" + "=" * 70)
    print("AZURE OPENAI PARAMETERS - QUICK REFERENCE (AI-102)")
    print("=" * 70)
    print(f"{'Parameter':<22} {'Range':<18} {'Effect':<30}")
    print("-" * 70)
    params = [
        ("temperature",       "0.0 – 2.0",       "Randomness (0=deterministic)"),
        ("top_p",             "0.0 – 1.0",       "Nucleus sampling (alt to temp)"),
        ("max_tokens",        "1 – model limit", "Max output length"),
        ("frequency_penalty", "-2.0 – 2.0",     "Reduce frequent token reuse"),
        ("presence_penalty",  "-2.0 – 2.0",     "Encourage new topics"),
        ("stop",              "up to 4 strings", "Custom stop sequences"),
        ("seed",              "integer",         "Reproducible outputs"),
        ("n",                 "1–10",            "Number of completions"),
        ("response_format",   "json_object",     "Force JSON output"),
        ("stream",            "bool",            "Stream tokens as generated"),
    ]
    for name, range_, effect in params:
        print(f"  {name:<20} {range_:<18} {effect}")
    print()
    print("KEY EXAM NOTES:")
    print("  - Use temperature OR top_p, not both")
    print("  - temperature=0 is NOT fully deterministic, use seed for that")
    print("  - frequency_penalty tracks per-token count; presence_penalty is binary")
    print("  - finish_reason='length' means response was cut off by max_tokens")
    print("  - finish_reason='stop' means response completed naturally")


if __name__ == "__main__":
    print("=" * 60)
    print("Azure OpenAI Model Parameters Demo")
    print("=" * 60)

    demo_temperature()
    demo_top_p()
    demo_max_tokens()
    demo_frequency_penalty()
    demo_presence_penalty()
    demo_stop_sequences()
    demo_seed()
    demo_n_completions()
    print_parameter_summary()

    print("\n" + "=" * 60)
    print("Model Parameters Demo Complete")
    print("=" * 60)
