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

AZURE_OPENAI_ENDPOINT   = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY        = os.environ.get("AZURE_OPENAI_KEY")
MODEL_DEPLOYMENT        = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# Agent names — used in strategies to route messages
RESEARCH_AGENT_NAME = "ResearchAgent"
WRITER_AGENT_NAME   = "WriterAgent"

# Termination signal the writer includes in its final message
TERMINATION_PHRASE  = "DONE"


# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------
RESEARCH_AGENT_INSTRUCTIONS = """
You are a meticulous research assistant. Your job is to:
1. Analyse the given topic or question
2. Identify the key facts, statistics, and important points
3. Organise the research into clear bullet-point notes
4. Indicate when your research is thorough and ready for the writer

Format your research as a structured list with sections where appropriate.
Be concise but comprehensive. Do not write prose paragraphs — keep it as
organised notes that a writer can use.
"""

WRITER_AGENT_INSTRUCTIONS = f"""
You are a professional content writer. Your job is to:
1. Take the research notes provided by the ResearchAgent
2. Transform them into engaging, well-structured prose content
3. Use clear language appropriate for a general audience
4. Produce a complete, polished piece

When you have produced a final, complete piece of writing, end your message
with the word "{TERMINATION_PHRASE}" on its own line so the workflow knows
you are finished.

Do not include raw bullet points in your final output — write proper paragraphs.
"""


# ---------------------------------------------------------------------------
# Selection strategy: decides which agent speaks next
# ---------------------------------------------------------------------------
SELECTION_FUNCTION_PROMPT = """
Examine the conversation history and determine which agent should respond next.

Agents:
- {research_agent}: gathers facts and research notes
- {writer_agent}: writes polished content from research

Rules:
- If the last message is from the user, {research_agent} goes first
- If the last message is from {research_agent}, {writer_agent} should respond
- If the last message is from {writer_agent} but does not contain '{termination}',
  {research_agent} should provide additional research or clarification
- Never select the same agent twice in a row

Conversation history:
{{{{$history}}}}

Respond with ONLY the agent name — either "{research_agent}" or "{writer_agent}".
""".format(
    research_agent=RESEARCH_AGENT_NAME,
    writer_agent=WRITER_AGENT_NAME,
    termination=TERMINATION_PHRASE,
)

# ---------------------------------------------------------------------------
# Termination strategy: detects when to stop the group chat
# ---------------------------------------------------------------------------
TERMINATION_FUNCTION_PROMPT = f"""
Review the most recent message in the conversation.

If the message contains the word "{TERMINATION_PHRASE}", respond with "yes".
Otherwise, respond with "no".

Most recent message:
{{{{$history}}}}

Respond with ONLY "yes" or "no".
"""


def build_kernel() -> sk.Kernel:
    """Create a Semantic Kernel instance with Azure OpenAI chat service."""
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set in .env"
        )

    kernel = sk.Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="azure-chat",
            deployment_name=MODEL_DEPLOYMENT,
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
        )
    )
    return kernel


async def run_multi_agent_workflow(topic: str) -> None:
    """Run the researcher+writer multi-agent workflow on a given topic.

    Args:
        topic: The subject the agents should research and write about.
    """
    kernel = build_kernel()

    # ------------------------------------------------------------------
    # 1. Create the two agents
    # Each ChatCompletionAgent wraps an LLM with a persona.
    # ------------------------------------------------------------------
    print("[1/4] Creating specialised agents...")

    research_agent = ChatCompletionAgent(
        service_id="azure-chat",
        kernel=kernel,
        name=RESEARCH_AGENT_NAME,
        instructions=RESEARCH_AGENT_INSTRUCTIONS,
    )

    writer_agent = ChatCompletionAgent(
        service_id="azure-chat",
        kernel=kernel,
        name=WRITER_AGENT_NAME,
        instructions=WRITER_AGENT_INSTRUCTIONS,
    )

    print(f"      Created: {RESEARCH_AGENT_NAME}")
    print(f"      Created: {WRITER_AGENT_NAME}")

    # ------------------------------------------------------------------
    # 2. Build selection and termination strategy functions
    # ------------------------------------------------------------------
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=SELECTION_FUNCTION_PROMPT,
    )

    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt=TERMINATION_FUNCTION_PROMPT,
    )

    # ------------------------------------------------------------------
    # 3. Create the AgentGroupChat
    # ------------------------------------------------------------------
    print("[2/4] Setting up AgentGroupChat...")

    group_chat = AgentGroupChat(
        agents=[research_agent, writer_agent],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip(),
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[writer_agent],   # Only check writer messages for termination
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip().lower() == "yes",
            history_variable_name="history",
            maximum_iterations=10,   # Safety limit
        ),
    )

    # ------------------------------------------------------------------
    # 4. Inject the user task and run the workflow
    # ------------------------------------------------------------------
    print(f"[3/4] Running multi-agent workflow on topic: '{topic}'\n")
    print("=" * 60)

    await group_chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=f"Topic: {topic}")
    )

    turn = 0
    async for message in group_chat.invoke():
        turn += 1
        author = message.name or message.role
        content = message.content

        print(f"\n--- Turn {turn}: {author} ---")
        print(content)

        # Check for termination signal
        if TERMINATION_PHRASE in content and message.name == WRITER_AGENT_NAME:
            print("\n[Termination signal detected — workflow complete]")
            break

    print("\n" + "=" * 60)
    print("[4/4] Multi-agent workflow finished.")

    # Display the full history
    print("\n--- Full Conversation History ---")
    async for msg in group_chat.get_chat_messages():
        print(f"\n[{msg.name or msg.role}]: {msg.content[:300]}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Semantic Kernel — Multi-Agent Orchestration Demo ===\n")

    topics = [
        "The benefits and challenges of quantum computing for cybersecurity",
    ]

    for topic in topics:
        print(f"\nProcessing topic: {topic}")
        try:
            asyncio.run(run_multi_agent_workflow(topic))
        except Exception as exc:
            print(f"Error: {exc}")
            raise
