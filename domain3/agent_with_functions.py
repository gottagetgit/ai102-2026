"""
agent_with_functions.py
=======================
Demonstrates creating an Azure AI Foundry agent with custom function tools
(also called "function calling" or "tool use"). The agent decides when to call
a function, the host application executes the actual function logic, and feeds
the result back so the agent can continue its reasoning.

Workflow:
    1. Define Python functions + JSON schemas describing their parameters
    2. Create an agent with those function tool definitions
    3. Create a thread and send a user message
    4. Start a run; detect when the run enters "requires_action" status
    5. Extract the tool calls the agent wants to make
    6. Execute the corresponding local Python functions
    7. Submit the results back to the run
    8. Repeat until the run completes, then display the final answer

Exam Skill Mapping:
    - "Implement complex agents with Microsoft Agent Framework"
    - "Create an agent with the Microsoft Foundry Agent Service"

Required Environment Variables (.env):
    AZURE_AI_PROJECT_CONNECTION_STRING
    AZURE_OPENAI_DEPLOYMENT

Install:
    pip install azure-ai-projects azure-identity python-dotenv
"""

import os
import json
import time
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FunctionTool, ToolSet, RequiredFunctionToolCall
from azure.identity import DefaultAzureCredential

load_dotenv()

CONNECTION_STRING = os.environ.get("AZURE_AI_PROJECT_CONNECTION_STRING")
MODEL_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# ===========================================================================
# LOCAL FUNCTION IMPLEMENTATIONS
# These are the actual business-logic functions that the agent can invoke.
# In production these would call real APIs, databases, or services.
# ===========================================================================

def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Return mock current weather for a given location.

    Args:
        location: City and country, e.g. "London, UK"
        unit:     Temperature unit — "celsius" or "fahrenheit"

    Returns:
        Dict with temperature, condition, humidity, and wind_speed.
    """
    # Simulated weather data (replace with a real API call in production)
    mock_data = {
        "london":   {"temp_c": 12, "condition": "Cloudy",  "humidity": 78, "wind_kmh": 20},
        "paris":    {"temp_c": 15, "condition": "Sunny",   "humidity": 55, "wind_kmh": 10},
        "new york": {"temp_c": 8,  "condition": "Rainy",   "humidity": 85, "wind_kmh": 30},
        "tokyo":    {"temp_c": 18, "condition": "Overcast","humidity": 65, "wind_kmh": 15},
    }

    city_key = location.split(",")[0].strip().lower()
    data = mock_data.get(city_key, {
        "temp_c": random.randint(5, 30),
        "condition": "Partly Cloudy",
        "humidity": random.randint(40, 80),
        "wind_kmh": random.randint(5, 40),
    })

    temp = data["temp_c"]
    if unit.lower() == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)
        unit_symbol = "°F"
    else:
        unit_symbol = "°C"

    return {
        "location": location,
        "temperature": f"{temp}{unit_symbol}",
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
        "wind_speed": f"{data['wind_kmh']} km/h",
        "retrieved_at": datetime.utcnow().isoformat() + "Z",
    }


def get_weather_forecast(location: str, days: int = 3) -> dict:
    """Return a mock multi-day weather forecast.

    Args:
        location: City name
        days:     Number of forecast days (1–7)

    Returns:
        Dict with a list of daily forecast entries.
    """
    days = min(max(days, 1), 7)  # Clamp to 1-7
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Thunderstorms", "Clear"]
    forecast = []

    for i in range(days):
        date = (datetime.utcnow() + timedelta(days=i+1)).strftime("%Y-%m-%d")
        forecast.append({
            "date": date,
            "high_c": random.randint(10, 28),
            "low_c": random.randint(2, 15),
            "condition": random.choice(conditions),
            "precipitation_chance": f"{random.randint(0, 90)}%",
        })

    return {"location": location, "forecast_days": days, "forecast": forecast}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between two currencies using fixed exchange rates.

    Args:
        amount:        The amount to convert
        from_currency: Source currency code (e.g. "USD")
        to_currency:   Target currency code (e.g. "EUR")

    Returns:
        Dict with original amount, converted amount, and rate used.
    """
    # Mock exchange rates relative to USD
    rates_to_usd = {
        "USD": 1.0, "EUR": 1.09, "GBP": 1.27, "JPY": 0.0067,
        "CAD": 0.74, "AUD": 0.66, "CHF": 1.13, "CNY": 0.14,
    }

    from_c = from_currency.upper()
    to_c   = to_currency.upper()

    if from_c not in rates_to_usd:
        return {"error": f"Unknown source currency: {from_c}"}
    if to_c not in rates_to_usd:
        return {"error": f"Unknown target currency: {to_c}"}

    usd_amount = amount / rates_to_usd[from_c]
    converted  = round(usd_amount * rates_to_usd[to_c], 2)
    rate       = round(rates_to_usd[to_c] / rates_to_usd[from_c], 6)

    return {
        "original":  f"{amount} {from_c}",
        "converted": f"{converted} {to_c}",
        "rate":      f"1 {from_c} = {rate} {to_c}",
        "note":      "Exchange rates are illustrative and not real-time.",
    }


# ---------------------------------------------------------------------------
# Function dispatcher: maps function name → callable
# ---------------------------------------------------------------------------
FUNCTION_MAP = {
    "get_current_weather":  get_current_weather,
    "get_weather_forecast": get_weather_forecast,
    "convert_currency":     convert_currency,
}


def dispatch_function_call(name: str, arguments_json: str) -> str:
    """Look up and execute a function by name; return JSON-serialised result.

    Args:
        name:           The function name from the tool call
        arguments_json: JSON string of keyword arguments

    Returns:
        JSON string result to pass back to the agent.
    """
    func = FUNCTION_MAP.get(name)
    if func is None:
        return json.dumps({"error": f"Unknown function: {name}"})

    try:
        kwargs = json.loads(arguments_json)
        result = func(**kwargs)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ===========================================================================
# JSON SCHEMAS FOR AGENT TOOL DEFINITIONS
# These tell the model what functions exist and how to call them.
# ===========================================================================

FUNCTION_DEFINITIONS = [
    {
        "name": "get_current_weather",
        "description": (
            "Get the current weather conditions for a specified location. "
            "Returns temperature, condition, humidity, and wind speed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'London, UK' or 'Tokyo, Japan'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit to use in the response.",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_weather_forecast",
        "description": "Get a multi-day weather forecast for a specified location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris'",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of forecast days (1 to 7).",
                    "minimum": 1,
                    "maximum": 7,
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "The amount to convert.",
                },
                "from_currency": {
                    "type": "string",
                    "description": "The source currency code, e.g. 'USD'.",
                },
                "to_currency": {
                    "type": "string",
                    "description": "The target currency code, e.g. 'EUR'.",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
]


# ===========================================================================
# AGENT + RUN LOOP
# ===========================================================================

def handle_tool_calls(client: AIProjectClient, thread_id: str, run) -> None:
    """Process all required tool calls and submit results back to the run.

    When a run enters 'requires_action' status, it contains one or more
    tool calls the agent wants to execute. We run them locally and submit
    the outputs so the agent can continue.
    """
    tool_outputs = []
    required_action = run.required_action

    for tool_call in required_action.submit_tool_outputs.tool_calls:
        print(f"      → Calling tool: {tool_call.function.name}")
        print(f"        Arguments: {tool_call.function.arguments}")

        output = dispatch_function_call(
            name=tool_call.function.name,
            arguments_json=tool_call.function.arguments,
        )
        print(f"        Result: {output[:200]}")

        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": output,
        })

    # Submit all results in a single call
    client.agents.submit_tool_outputs_to_run(
        thread_id=thread_id,
        run_id=run.id,
        tool_outputs=tool_outputs,
    )


def run_agent_with_polling(
    client: AIProjectClient, thread_id: str, agent_id: str
) -> str:
    """Start a run and manually poll, handling tool calls along the way.

    This low-level approach (vs create_and_process_run) gives us visibility
    into each step of the run lifecycle — important for exam understanding.

    Returns:
        The final assistant message text.
    """
    run = client.agents.create_run(thread_id=thread_id, agent_id=agent_id)
    print(f"   Run started: id={run.id}")

    # Poll loop
    while run.status in ("queued", "in_progress", "requires_action"):
        time.sleep(1)
        run = client.agents.get_run(thread_id=thread_id, run_id=run.id)
        print(f"   Run status: {run.status}")

        if run.status == "requires_action":
            print("   Agent requires tool execution:")
            handle_tool_calls(client, thread_id, run)

    if run.status == "failed":
        raise RuntimeError(f"Run failed: {run.last_error}")

    # Retrieve final messages
    messages = client.agents.list_messages(thread_id=thread_id)
    for msg in messages.data:
        if msg.role == "assistant":
            parts = [
                block.text.value
                for block in msg.content
                if hasattr(block, "text")
            ]
            return "\n".join(parts)
    return "(no response)"


def run_function_tools_demo():
    """Full demonstration of an agent with custom function tools."""
    if not CONNECTION_STRING:
        raise ValueError("AZURE_AI_PROJECT_CONNECTION_STRING is not set.")

    client = AIProjectClient.from_connection_string(
        conn_str=CONNECTION_STRING,
        credential=DefaultAzureCredential(),
    )

    # Build a FunctionTool from the schema definitions
    functions = FunctionTool(definitions=FUNCTION_DEFINITIONS)

    print("[1/5] Creating agent with function tools...")
    agent = client.agents.create_agent(
        model=MODEL_DEPLOYMENT,
        name="function-tools-demo",
        instructions=(
            "You are a helpful travel and finance assistant. "
            "Use the available tools to fetch current weather, forecasts, "
            "and currency conversions. Always use the tools rather than guessing. "
            "Provide friendly, well-structured responses."
        ),
        tools=functions.definitions,
    )
    print(f"      Agent id: {agent.id}")

    try:
        thread = client.agents.create_thread()
        print(f"[2/5] Thread: {thread.id}\n")

        # Test prompts that will trigger function calls
        prompts = [
            "What is the current weather in London and Tokyo?",
            "I'm travelling to Paris for 4 days — can you give me a forecast "
            "and also convert £500 to Euros?",
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"{'='*60}")
            print(f"User ({i}): {prompt}")
            print(f"{'='*60}")

            client.agents.create_message(
                thread_id=thread.id, role="user", content=prompt
            )

            reply = run_agent_with_polling(client, thread.id, agent.id)
            print(f"\nAssistant:\n{reply}\n")

    finally:
        client.agents.delete_agent(agent.id)
        print(f"Cleaned up agent: {agent.id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Azure AI Foundry — Agent with Function Tools Demo ===\n")
    try:
        run_function_tools_demo()
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
