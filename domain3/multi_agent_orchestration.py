"""
multi_agent_orchestration.py
=============================
Demonstrates a multi-agent workflow using Semantic Kernel's Agent Framework.
Two specialised agents collaborate in a structured group chat:

  - ResearchAgent: gathers and summarises factual information on a topic
  - WriterAgent:   takes the research and produces polished written content

The AgentGroupChat orchestrates message passing between agents with a
termination condition (writer signals DONE) and a selection strategy
(alternating between researcher and writer roles).

Workflow:
    1. Create two ChatCompletionAgent instances with different personas
    2. Set up an AgentGroupChat with KernelFunctionTerminationStrategy
       and KernelFunctionSelectionStrategy
    3. Inject a user task and iterate the chat until termination
    4. Display the full conversation transcript

Exam Skill Mapping:
    - "Implement complex workflows including orchestration for a multi-agent solution"
    - "Implement complex agents with Microsoft Agent Framework"

Required Environment Variables (.env):
    AZURE_OPENAI_ENDPOINT    - e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_KEY         - Azure OpenAI API key
    AZURE_OPENAI_DEPLOYMENT  - Model deployment name (e.g. "gpt-4o")

Install:
    pip install semantic-kernel python-dotenv
    (semantic-kernel >= 1.3.0 for AgentGroupChat support)
"""

import os
import asyncio
from dotenv import load_dotenv

# Semantic Kernel imports
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies import (
    KernelFunctionTerminationStrategy,
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.functions import KernelFunctionFromPrompt

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY    = os.environ["AZURE_OPENAI_KEY"]
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]


# ---------------------------------------------------------------------------
# Helper: build a ChatCompletionAgent backed by Azure OpenAI
# ---------------------------------------------------------------------------

def build_agent(name: str, instructions: str) -> ChatCompletionAgent:
    """Return a ChatCompletionAgent with its own Kernel + AzureChatCompletion."""
    kernel = sk.Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="azure_oai",
            deployment_name=DEPLOYMENT,
            endpoint=ENDPOINT,
            api_key=API_KEY,
        )
    )
    return ChatCompletionAgent(
        kernel=kernel,
        name=name,
        instructions=instructions,
    )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

RESEARCH_INSTRUCTIONS = """
You are ResearchAgent. When given a topic:
1. Provide 3-5 concise factual points about the topic.
2. End your message with: RESEARCH_COMPLETE
"""

WRITER_INSTRUCTIONS = """
You are WriterAgent. When you receive research points:
1. Transform them into a short, engaging 2-paragraph article.
2. End your message with: DONE
"""


# ---------------------------------------------------------------------------
# Kernel functions for termination and selection strategies
# ---------------------------------------------------------------------------

def build_strategy_kernel() -> sk.Kernel:
    """A lightweight kernel used only for strategy functions."""
    kernel = sk.Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="azure_oai",
            deployment_name=DEPLOYMENT,
            endpoint=ENDPOINT,
            api_key=API_KEY,
        )
    )
    return kernel


TERMINATION_PROMPT = """
Determine whether the conversation should end.
The conversation should end when the WriterAgent has produced the article
and included the word DONE in its last message.

Respond with a single word: YES or NO.

History:
{{$history}}
"""

SELECTION_PROMPT = """
Decide which agent should speak next based on the conversation history.
Rules:
- If the last speaker was ResearchAgent (or no one has spoken yet), choose WriterAgent.
- Otherwise choose ResearchAgent.

Respond with ONLY the agent name: ResearchAgent or WriterAgent.

History:
{{$history}}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_multi_agent_chat(topic: str) -> None:
    print(f"\n{'='*60}")
    print(f"Multi-Agent Chat: {topic}")
    print('='*60)

    # Build agents
    research_agent = build_agent("ResearchAgent", RESEARCH_INSTRUCTIONS)
    writer_agent   = build_agent("WriterAgent",   WRITER_INSTRUCTIONS)

    # Build strategy kernel
    strategy_kernel = build_strategy_kernel()

    # Termination strategy: stop when WriterAgent says DONE
    termination_fn = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=TERMINATION_PROMPT,
    )
    termination_strategy = KernelFunctionTerminationStrategy(
        agents=[writer_agent],
        function=termination_fn,
        kernel=strategy_kernel,
        result_parser=lambda result: "YES" in str(result).upper(),
        history_variable_name="history",
        maximum_iterations=10,
    )

    # Selection strategy: alternate agents
    selection_fn = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=SELECTION_PROMPT,
    )
    selection_strategy = KernelFunctionSelectionStrategy(
        function=selection_fn,
        kernel=strategy_kernel,
        result_parser=lambda result: str(result).strip(),
        history_variable_name="history",
    )

    # Create AgentGroupChat
    group_chat = AgentGroupChat(
        agents=[research_agent, writer_agent],
        termination_strategy=termination_strategy,
        selection_strategy=selection_strategy,
    )

    # Seed the conversation with the user's task
    await group_chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=f"Write a short article about: {topic}")
    )

    # Iterate until termination
    print("\n--- Conversation ---")
    async for message in group_chat.invoke():
        print(f"\n[{message.name}]:\n{message.content}")

    print("\n--- Chat complete ---")
    print(f"Total turns: {group_chat.history.__len__()}")


if __name__ == "__main__":
    topic = "Azure AI Services and their role in enterprise AI solutions"
    asyncio.run(run_multi_agent_chat(topic))
