"""
agent_evaluation.py
===================
Demonstrates testing and evaluating an Azure AI Foundry agent by:

  1. Defining a structured test suite with inputs and expected outcomes
  2. Running each test case through a live agent
  3. Using an LLM-as-judge to score response quality on multiple dimensions
  4. Computing aggregate metrics: pass rate, average quality scores
  5. Outputting a structured evaluation report (JSON + human-readable summary)

Evaluation Dimensions:
  - Relevance:   Does the response address the question?
  - Accuracy:    Is the information factually correct (compared to expected)?
  - Completeness: Does the response cover all required points?
  - Conciseness:  Is the response appropriately brief without being terse?

Exam Skill Mapping:
    - "Test, optimize and deploy an agent"
    - "Create an agent with the Microsoft Foundry Agent Service"

Required Environment Variables (.env):
    AZURE_AI_PROJECT_CONNECTION_STRING
    AZURE_OPENAI_DEPLOYMENT       - Deployment for the agent under test
    AZURE_OPENAI_JUDGE_DEPLOYMENT - (Optional) Separate deployment for the judge
                                    Falls back to AZURE_OPENAI_DEPLOYMENT if not set

Install:
    pip install azure-ai-projects azure-identity openai python-dotenv
"""

import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

load_dotenv()

CONNECTION_STRING    = os.environ.get("AZURE_AI_PROJECT_CONNECTION_STRING")
MODEL_DEPLOYMENT     = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
JUDGE_DEPLOYMENT     = os.environ.get("AZURE_OPENAI_JUDGE_DEPLOYMENT", MODEL_DEPLOYMENT)
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY      = os.environ.get("AZURE_OPENAI_KEY")


# ===========================================================================
# TEST SUITE DEFINITION
# ===========================================================================

@dataclass
class TestCase:
    """A single evaluation test case."""
    id: str
    category: str
    user_input: str
    expected_keywords: list[str]  # Words/phrases that should appear in a good answer
    expected_concepts: str        # Natural-language description of what a good answer covers
    passing_threshold: float = 0.7  # Minimum average score to pass (0-1 scale)


# Define representative test cases covering different capability areas
TEST_SUITE: list[TestCase] = [
    TestCase(
        id="tc-001",
        category="factual_knowledge",
        user_input="What is Azure AI Services and what are its main categories?",
        expected_keywords=["vision", "speech", "language", "decision", "cognitive"],
        expected_concepts=(
            "Should explain that Azure AI Services is a set of cloud-based AI APIs, "
            "mention the main categories (Vision, Speech, Language, Decision/OpenAI), "
            "and note they are accessed via REST APIs."
        ),
    ),
    TestCase(
        id="tc-002",
        category="conceptual_explanation",
        user_input="Explain the difference between classification and object detection in Custom Vision.",
        expected_keywords=["classification", "object detection", "bounding box", "label", "tag"],
        expected_concepts=(
            "Should distinguish: classification assigns a label to the whole image; "
            "object detection finds and localises multiple objects with bounding boxes. "
            "Should mention both use cases."
        ),
    ),
    TestCase(
        id="tc-003",
        category="best_practices",
        user_input="What are best practices for responsible AI deployment on Azure?",
        expected_keywords=["fairness", "transparency", "accountability", "safety", "privacy"],
        expected_concepts=(
            "Should cover Microsoft's Responsible AI principles: fairness, reliability, "
            "privacy, inclusiveness, transparency, and accountability. "
            "Practical examples are a bonus."
        ),
    ),
    TestCase(
        id="tc-004",
        category="troubleshooting",
        user_input="Why might an Azure OpenAI API call return a 429 error and how do you fix it?",
        expected_keywords=["rate limit", "throttling", "retry", "TPM", "quota", "429"],
        expected_concepts=(
            "Should explain 429 = Too Many Requests / rate limit exceeded, "
            "mention tokens-per-minute (TPM) limits, and suggest solutions: "
            "exponential back-off retry, increase quota, switch deployment, or batch requests."
        ),
    ),
    TestCase(
        id="tc-005",
        category="architecture",
        user_input=(
            "In an Azure AI agent, what is the difference between a thread and a run?"
        ),
        expected_keywords=["thread", "run", "conversation", "messages", "agent", "execute"],
        expected_concepts=(
            "Thread = a conversation session containing a sequence of messages. "
            "Run = the execution of an agent against a thread; a run processes "
            "the messages and produces a response. Multiple runs can occur on the same thread."
        ),
    ),
]


# ===========================================================================
# AGENT UNDER TEST
# ===========================================================================

def get_agent_response(
    client: AIProjectClient, agent_id: str, user_input: str
) -> str:
    """Run a single user message through the agent and return the reply."""
    thread = client.agents.create_thread()
    client.agents.create_message(thread_id=thread.id, role="user", content=user_input)
    run = client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent_id)

    if run.status == "failed":
        return f"[ERROR] Run failed: {run.last_error}"

    messages = client.agents.list_messages(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant":
            parts = [b.text.value for b in msg.content if hasattr(b, "text")]
            return "\n".join(parts)
    return "[ERROR] No assistant message returned"


# ===========================================================================
# LLM-AS-JUDGE EVALUATION
# ===========================================================================

JUDGE_SYSTEM_PROMPT = """
You are an objective AI evaluation judge. You assess AI assistant responses
based on specific quality dimensions. For each dimension, give a score from
0.0 to 1.0 (to one decimal place), where:
  1.0 = Excellent, fully meets the criterion
  0.7 = Good, mostly meets the criterion with minor gaps
  0.5 = Acceptable, partially meets the criterion
  0.3 = Poor, significant gaps or issues
  0.0 = Fails to meet the criterion

Always respond with valid JSON only - no other text.
"""

JUDGE_USER_TEMPLATE = """
Evaluate this AI assistant response:

USER QUESTION:
{question}

AI RESPONSE:
{response}

EVALUATION CRITERIA:
{criteria}

EXPECTED KEYWORDS (check if present): {keywords}

Rate on these four dimensions and provide a brief justification for each:
- relevance:    Does the response directly address the question?
- accuracy:     Is the information correct and aligned with the expected concepts?
- completeness: Does the response cover all important points from the criteria?
- conciseness:  Is the response appropriately concise (not too short, not rambling)?

Also set "passes_threshold" to true if the average score is >= {threshold}.

Respond ONLY with this JSON structure:
{{
  "relevance":          {{ "score": 0.0, "justification": "..." }},
  "accuracy":           {{ "score": 0.0, "justification": "..." }},
  "completeness":       {{ "score": 0.0, "justification": "..." }},
  "conciseness":        {{ "score": 0.0, "justification": "..." }},
  "keywords_found":     ["list", "of", "matched", "keywords"],
  "keywords_missing":   ["list", "of", "missing", "keywords"],
  "overall_score":      0.0,
  "passes_threshold":   false,
  "summary":            "One-sentence overall verdict."
}}
"""


@dataclass
class EvaluationResult:
    test_case_id: str
    category: str
    user_input: str
    agent_response: str
    scores: dict = field(default_factory=dict)
    keywords_found: list = field(default_factory=list)
    keywords_missing: list = field(default_factory=list)
    overall_score: float = 0.0
    passes_threshold: bool = False
    summary: str = ""
    judge_error: Optional[str] = None


def judge_response(
    oai_client: AzureOpenAI, test_case: TestCase, agent_response: str
) -> EvaluationResult:
    """Use an LLM judge to score the agent response against a test case."""
    result = EvaluationResult(
        test_case_id=test_case.id,
        category=test_case.category,
        user_input=test_case.user_input,
        agent_response=agent_response,
    )

    judge_prompt = JUDGE_USER_TEMPLATE.format(
        question=test_case.user_input,
        response=agent_response,
        criteria=test_case.expected_concepts,
        keywords=", ".join(test_case.expected_keywords),
        threshold=test_case.passing_threshold,
    )

    try:
        completion = oai_client.chat.completions.create(
            model=JUDGE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": judge_prompt},
            ],
            temperature=0.0,  # Deterministic for evaluation
            response_format={"type": "json_object"},
        )

        judgment = json.loads(completion.choices[0].message.content)

        result.scores = {
            dim: judgment[dim]
            for dim in ("relevance", "accuracy", "completeness", "conciseness")
            if dim in judgment
        }
        result.keywords_found   = judgment.get("keywords_found", [])
        result.keywords_missing = judgment.get("keywords_missing", [])
        result.overall_score    = judgment.get("overall_score", 0.0)
        result.passes_threshold = judgment.get("passes_threshold", False)
        result.summary          = judgment.get("summary", "")

    except Exception as exc:
        result.judge_error = str(exc)
        print(f"     [Judge error for {test_case.id}]: {exc}")

    return result


# ===========================================================================
# EVALUATION RUNNER
# ===========================================================================

def run_evaluation():
    """Run the full agent evaluation pipeline."""
    if not CONNECTION_STRING:
        raise ValueError("AZURE_AI_PROJECT_CONNECTION_STRING is not set.")
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set for the judge.")

    # Foundry project client (for the agent under test)
    project_client = AIProjectClient.from_connection_string(
        conn_str=CONNECTION_STRING,
        credential=DefaultAzureCredential(),
    )

    # Direct Azure OpenAI client for the judge
    oai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-01",
    )

    # Create the agent under test
    print("[1/4] Creating agent under test...")
    agent = project_client.agents.create_agent(
        model=MODEL_DEPLOYMENT,
        name="eval-agent-under-test",
        instructions=(
            "You are a knowledgeable Azure AI expert. "
            "Answer questions about Azure AI Services, Azure OpenAI, and related topics "
            "accurately and concisely. Focus on exam-relevant facts."
        ),
    )
    print(f"      Agent id: {agent.id}")

    results: list[EvaluationResult] = []

    try:
        print(f"\n[2/4] Running {len(TEST_SUITE)} test cases...\n")

        for i, test_case in enumerate(TEST_SUITE, 1):
            print(f"  [{i}/{len(TEST_SUITE)}] {test_case.id} ({test_case.category})")
            print(f"       Input: {test_case.user_input[:80]}...")

            # Get agent response
            response = get_agent_response(project_client, agent.id, test_case.user_input)
            print(f"       Response length: {len(response)} chars")

            # Judge the response
            result = judge_response(oai_client, test_case, response)
            results.append(result)

            status = "PASS" if result.passes_threshold else "FAIL"
            print(f"       Score: {result.overall_score:.2f} | {status} | {result.summary[:60]}...")
            time.sleep(0.5)  # Brief pause between runs

        # ------------------------------------------------------------------
        # 3. Compute aggregate metrics
        # ------------------------------------------------------------------
        print(f"\n[3/4] Computing aggregate metrics...")

        passed = sum(1 for r in results if r.passes_threshold)
        pass_rate = passed / len(results) if results else 0

        dimension_averages = {}
        for dim in ("relevance", "accuracy", "completeness", "conciseness"):
            scores = [
                r.scores[dim]["score"]
                for r in results
                if dim in r.scores
            ]
            dimension_averages[dim] = round(sum(scores) / len(scores), 3) if scores else 0

        overall_avg = sum(r.overall_score for r in results) / len(results) if results else 0

        # ------------------------------------------------------------------
        # 4. Output the evaluation report
        # ------------------------------------------------------------------
        report = {
            "evaluation_summary": {
                "total_tests":      len(results),
                "passed":           passed,
                "failed":           len(results) - passed,
                "pass_rate":        f"{pass_rate:.0%}",
                "overall_avg_score": round(overall_avg, 3),
                "dimension_averages": dimension_averages,
            },
            "test_results": [asdict(r) for r in results],
        }

        report_path = "agent_evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"      Report saved to: {report_path}")

        # Human-readable summary
        print("\n" + "=" * 60)
        print("EVALUATION REPORT SUMMARY")
        print("=" * 60)
        print(f"Total tests:        {len(results)}")
        print(f"Passed:             {passed} ({pass_rate:.0%})")
        print(f"Overall avg score:  {overall_avg:.3f}")
        print("\nDimension averages:")
        for dim, avg in dimension_averages.items():
            bar = "#" * int(avg * 20)
            print(f"  {dim:<15} {avg:.3f}  {bar}")

        print("\nPer-test results:")
        for r in results:
            status = "PASS" if r.passes_threshold else "FAIL"
            print(f"  [{status}] {r.test_case_id} ({r.category}): {r.overall_score:.2f} - {r.summary}")

        print("\n[4/4] Evaluation complete.")
        return report

    finally:
        project_client.agents.delete_agent(agent.id)
        print(f"\nCleaned up agent: {agent.id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Foundry - Agent Evaluation Demo ===\n")
    try:
        run_evaluation()
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
