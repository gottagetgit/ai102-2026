"""
model_parameters.py
===================
Demonstrates how each generation parameter affects Azure OpenAI model output.
Provides concrete before/after examples showing the practical effect of each setting.

Exam Skill: "Configure parameters to control generative behavior"
            (Domain 2 - Implement generative AI solutions)

What this demo shows:
  - temperature: randomness / creativity
  - max_tokens: output length control
  - top_p: nucleus sampling
  - frequency_penalty: reducing word repetition
  - presence_penalty: encouraging topic diversity
  - stop sequences: ending generation at specific tokens
  - n: generating multiple completions
  - logit_bias: influencing token probabilities
  - seed: deterministic outputs (for reproducibility)

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

BASE_MESSAGES = [
    {"role": "system", "content": "You are a creative writing assistant."},
    {"role": "user",   "content": "Write a short sentence about a robot exploring Mars."},
]


def get_client() -> openai.AzureOpenAI:
    """Return authenticated AzureOpenAI client."""
    return openai.AzureOpenAI(
        api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        api_version="2024-12-01-preview",
    )


def complete(client: openai.AzureOpenAI, messages: list, **kwargs) -> str:
    """Helper: send messages with given params and return text."""
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 1. temperature
# ---------------------------------------------------------------------------

def demo_temperature(client: openai.AzureOpenAI) -> None:
    """
    temperature controls the randomness of token selection.

    Mechanism: At each token step, the model produces a probability distribution
    over the vocabulary. Temperature scales this distribution:
      - Low (0.0-0.3)  : Sharpen distribution → picks most likely tokens → deterministic
      - Medium (0.7)   : Balanced randomness → default for most tasks
      - High (1.0-1.5) : Flatten distribution → more varied, creative, potentially incoherent
      - 0.0            : Greedy decoding (always picks the highest probability token)

    Range: 0.0 to 2.0

    DO NOT use temperature and top_p together (they both control randomness).
    """
    print("\n--- Parameter: temperature ---")
    print("  Controls randomness. Low=deterministic, High=creative")
    print()

    prompt = "Write an opening line for a science fiction story."

    for temp in [0.0, 0.5, 1.0, 1.5]:
        result = complete(
            client,
            [
                {"role": "system", "content": "You are a creative writer."},
                {"role": "user",   "content": prompt},
            ],
            temperature=temp,
            max_tokens=50,
        )
        print(f"  temperature={temp}: {result}")


# ---------------------------------------------------------------------------
# 2. max_tokens
# ---------------------------------------------------------------------------

def demo_max_tokens(client: openai.AzureOpenAI) -> None:
    """
    max_tokens sets the maximum number of tokens the model can generate.
    Does NOT control quality - just cuts off when the limit is reached.

    Important: The total tokens (prompt + completion) must stay within the
    model's context window:
      - GPT-3.5 Turbo: 16,385 tokens
      - GPT-4:         128,000 tokens
      - GPT-4o:        128,000 tokens

    Cost is proportional to total tokens processed, so max_tokens also
    acts as a cost control mechanism.

    When finish_reason == "length", the model was cut off.
    When finish_reason == "stop",   the model finished naturally.
    """
    print("\n--- Parameter: max_tokens ---")
    print("  Maximum tokens in the response. Watch for 'length' finish reason.")
    print()

    prompt = "Explain the history of artificial intelligence in detail."

    for limit in [20, 60, 200]:
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=limit,
            temperature=0.3,
        )
        text    = resp.choices[0].message.content
        reason  = resp.choices[0].finish_reason
        tokens  = resp.usage.completion_tokens
        print(f"  max_tokens={limit:4d}: [{reason:6s}] {tokens} tokens → '{text[:70]}...'")


# ---------------------------------------------------------------------------
# 3. top_p (nucleus sampling)
# ---------------------------------------------------------------------------

def demo_top_p(client: openai.AzureOpenAI) -> None:
    """
    top_p (nucleus sampling) controls diversity by restricting the model to
    only consider the top tokens whose cumulative probability reaches top_p.

    Example: top_p=0.9 means only consider tokens until we've covered 90%
    of the probability mass. Low-probability "weird" tokens are excluded.

    Range: 0.0 to 1.0
      - 1.0 = consider all tokens (full vocabulary) - most diverse
      - 0.1 = only the top 10% probability mass - very focused

    Guideline: Tune EITHER temperature OR top_p. OpenAI recommends
    not changing both at the same time.
    """
    print("\n--- Parameter: top_p ---")
    print("  Nucleus sampling. Lower=less random (fewer candidate tokens)")
    print()

    prompt = "Suggest a creative business name for a pet store."

    for tp in [0.1, 0.5, 0.95]:
        result = complete(
            client,
            [{"role": "user", "content": prompt}],
            top_p=tp,
            temperature=1.0,  # Set temperature to 1.0 when tuning top_p
            max_tokens=30,
        )
        print(f"  top_p={tp}: {result}")


# ---------------------------------------------------------------------------
# 4. frequency_penalty
# ---------------------------------------------------------------------------

def demo_frequency_penalty(client: openai.AzureOpenAI) -> None:
    """
    frequency_penalty reduces the likelihood of repeating tokens that have
    already appeared in the response. Penalizes based on how many times
    a token has appeared.

    Range: -2.0 to 2.0
      - 0.0  = No penalty (default)
      - 0.5  = Moderate reduction in repetition
      - 1.0  = Strong reduction - good for avoiding verbose/repetitive text
      - 2.0  = Very strong - may cause incoherence
      - Negative values INCREASE repetition (rarely useful)

    Use when: The model keeps repeating the same phrases or words.
    """
    print("\n--- Parameter: frequency_penalty ---")
    print("  Reduces token repetition. Higher = more varied vocabulary.")
    print()

    prompt = "Write a paragraph about the benefits of using cloud computing services."

    for fp in [0.0, 0.8, 1.5]:
        result = complete(
            client,
            [{"role": "user", "content": prompt}],
            frequency_penalty=fp,
            temperature=0.7,
            max_tokens=100,
        )
        print(f"  frequency_penalty={fp}:")
        print(f"    {result[:200]}")
        print()


# ---------------------------------------------------------------------------
# 5. presence_penalty
# ---------------------------------------------------------------------------

def demo_presence_penalty(client: openai.AzureOpenAI) -> None:
    """
    presence_penalty penalizes tokens that have appeared at all in the response,
    regardless of how many times. Encourages the model to introduce new topics.

    Range: -2.0 to 2.0
      - 0.0  = No penalty (default)
      - 0.5  = Encourages topic diversity
      - 1.0  = Strongly encourages new topics
      - 2.0  = Very aggressive - response may jump around incoherently

    Difference from frequency_penalty:
      - frequency_penalty: penalizes in proportion to frequency (discourages repetitive words)
      - presence_penalty:  penalizes any token that appeared even once (encourages new topics)

    Use when: You want the model to cover more ground rather than staying on one topic.
    """
    print("\n--- Parameter: presence_penalty ---")
    print("  Encourages topic diversity. Higher = more likely to introduce new topics.")
    print()

    prompt = "Discuss the future of technology. Write 3 sentences."

    for pp in [0.0, 1.0, 1.8]:
        result = complete(
            client,
            [{"role": "user", "content": prompt}],
            presence_penalty=pp,
            temperature=0.7,
            max_tokens=100,
        )
        print(f"  presence_penalty={pp}:")
        print(f"    {result[:200]}")
        print()


# ---------------------------------------------------------------------------
# 6. stop sequences
# ---------------------------------------------------------------------------

def demo_stop_sequences(client: openai.AzureOpenAI) -> None:
    """
    stop sequences cause the model to stop generating when it produces
    any of the specified strings. Up to 4 stop sequences can be provided.

    Common use cases:
      - Generate exactly one item in a list: stop=["\n", "2."]
      - Parse structured output: stop=["```", "---"]
      - Limit a chatbot to one exchange: stop=["User:", "Human:"]
      - Stop at a delimiter in data extraction: stop=["END", "###"]
    """
    print("\n--- Parameter: stop sequences ---")
    print("  Generation stops when any stop sequence is produced.")
    print()

    # Without stop sequence - model generates full list
    prompt = "List 5 Azure AI services:\n1."
    result_no_stop = complete(
        client,
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100,
    )
    print(f"  Without stop: {result_no_stop}")

    # With stop sequence - stops after first item
    result_with_stop = complete(
        client,
        [{"role": "user", "content": prompt}],
        stop=["\n2."],  # Stop before the second item
        temperature=0.3,
        max_tokens=100,
    )
    print(f"\n  With stop=['\\n2.']: {result_with_stop}")

    # Stop on multiple possible sequences
    result_multi_stop = complete(
        client,
        [{"role": "user", "content": "Answer: The capital of France is"}],
        stop=[".", "\n", ","],  # Stop at end of word/sentence
        temperature=0.0,
        max_tokens=10,
    )
    print(f"\n  stop=['.','\\n',','] → '{result_multi_stop}'")


# ---------------------------------------------------------------------------
# 7. n - generating multiple responses
# ---------------------------------------------------------------------------

def demo_n_completions(client: openai.AzureOpenAI) -> None:
    """
    n specifies how many completion choices to generate for a single prompt.
    Useful for:
      - Showing users multiple options (e.g., subject line alternatives)
      - Picking the best from several candidates
      - A/B testing prompts

    Note: n > 1 multiplies your token cost.
    """
    print("\n--- Parameter: n (multiple completions) ---")

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "user", "content": "Suggest a catchy tagline for an AI productivity app. One sentence only."},
        ],
        n=3,            # Generate 3 alternatives
        temperature=1.0,  # Higher temperature for variety
        max_tokens=40,
    )

    print(f"  Generated {len(resp.choices)} alternatives:")
    for i, choice in enumerate(resp.choices, 1):
        print(f"  {i}. {choice.message.content.strip()}")

    print(f"\n  Total tokens used: {resp.usage.total_tokens}")


# ---------------------------------------------------------------------------
# 8. seed - deterministic outputs
# ---------------------------------------------------------------------------

def demo_seed(client: openai.AzureOpenAI) -> None:
    """
    seed makes outputs more reproducible. With the same seed and parameters,
    the model tends to produce the same output.

    Important caveats:
      - Not guaranteed to be identical (model can have non-deterministic operations)
      - May change with model updates
      - The response includes system_fingerprint to identify the model version

    Use for: Unit tests, debugging, demos where you need consistent output.
    """
    print("\n--- Parameter: seed (reproducibility) ---")

    prompt_msgs = [
        {"role": "user", "content": "Give me a random number between 1 and 100."},
    ]

    print("  Without seed (different each time):")
    for i in range(3):
        result = complete(client, prompt_msgs, temperature=1.0, max_tokens=10)
        print(f"  Run {i+1}: {result}")

    print("\n  With seed=42 (more consistent):")
    for i in range(3):
        resp = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=prompt_msgs,
            temperature=1.0,
            max_tokens=10,
            seed=42,
        )
        text = resp.choices[0].message.content.strip()
        fingerprint = resp.system_fingerprint
        print(f"  Run {i+1}: {text} (fingerprint: {fingerprint})")


def main():
    print("=" * 60)
    print("Azure OpenAI Generation Parameters Demo")
    print("=" * 60)
    print(f"Deployment: {DEPLOYMENT}")
    print("""
  Parameters Reference:
  | Parameter         | Range       | Effect                              |
  |-------------------|-------------|-------------------------------------|
  | temperature       | 0.0 – 2.0   | Randomness / creativity             |
  | max_tokens        | 1 – ctx_len | Max output length                   |
  | top_p             | 0.0 – 1.0   | Nucleus sampling diversity          |
  | frequency_penalty | -2.0 – 2.0  | Reduce word repetition              |
  | presence_penalty  | -2.0 – 2.0  | Encourage new topics                |
  | stop              | list[str]   | Stop generation at these strings    |
  | n                 | int ≥ 1     | Number of response alternatives     |
  | seed              | int         | Deterministic output (approx.)      |
""")

    try:
        client = get_client()

        demo_temperature(client)
        demo_max_tokens(client)
        demo_top_p(client)
        demo_frequency_penalty(client)
        demo_presence_penalty(client)
        demo_stop_sequences(client)
        demo_n_completions(client)
        demo_seed(client)

    except KeyError as e:
        print(f"\n[ERROR] Missing environment variable: {e}")
    except openai.APIError as e:
        print(f"\n[ERROR] Azure OpenAI error: {e}")


if __name__ == "__main__":
    main()
