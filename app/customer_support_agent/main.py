"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()

# TODO: Create the BedrockAgentCoreApp instance
app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in Part 1 of the INSTRUCTIONS.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-ub6brbogpk.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"   # TODO: Replace with your Gateway URL
KB_ID       = "UYHXUR4O7X"          # TODO: Replace with your Knowledge Base ID
REGION = "us-east-1"        # TODO: Replace with your AWS region
MEMORY_ID   = "CustomerSupportMemory-3nvQJtAkDs"        # TODO: Replace with your Memory ID


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

model_id = "global.amazon.nova-2-lite-v1:0"

# TODO: Create the BedrockModel instance
model = BedrockModel(model_id=model_id)

# TODO: Create the MemoryClient instance
memory_client = MemoryClient(region_name=REGION)

# TODO: Create the boto3 bedrock-agent-runtime client
_bedrock_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Return a dict: { strategy["type"]: strategy["namespaces"][0] for each strategy }
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)

    namespaces = {}

    for strategy in strategies:
        strategy_type = strategy.get("type")

        # Current API uses namespaceTemplates.
        # Older responses may use namespaces.
        namespace_templates = strategy.get("namespaceTemplates")
        if not namespace_templates:
            namespace_templates = strategy.get("namespaces", [])

        if strategy_type and namespace_templates:
            namespaces[strategy_type] = namespace_templates[0]

    return namespaces


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent

class MemoryHook(HookProvider):
    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)
        self._context_added = False

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant long-term memories and add them to the user message."""
        if self._context_added:
            return
        messages = event.agent.messages

        if not messages:
            return

        # Get the latest user message.
        user_message = None

        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = message.get("content", "")

            if isinstance(content, str):
                user_message = content
                break

            if isinstance(content, list):
                text_parts = []

                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])

                if text_parts:
                    user_message = "\n".join(text_parts)
                    break

        if not user_message:
            return

        memory_context = []

        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.replace(
                "{actorId}",
                self.actor_id,
            )

            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_message,
                    top_k=5,
                )

                for memory in memories:
                    memory_text = memory.get("content", {}).get("text")

                    if not memory_text:
                        memory_text = memory.get("text")

                    if memory_text:
                        memory_context.append(
                            f"[{strategy_type}] {memory_text}"
                        )

            except Exception as e:
                logging.warning(
                    "Could not retrieve memories from %s: %s",
                    strategy_type,
                    e,
                )

        if memory_context:
            context_text = "\n".join(memory_context)

            original_message = user_message

            enhanced_message = (
                "Customer Context:\n"
                f"{context_text}\n\n"
                f"{original_message}"
            )

            # Replace the latest user message with the version
            # containing retrieved customer context.
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get("content")

                    if isinstance(content, str):
                        message["content"] = enhanced_message
                    elif isinstance(content, list):
                        message["content"] = [
                            {"text": enhanced_message}
                        ]

                    break

        # Prevent this hook from repeatedly enriching the same
        # user message when additional MessageAddedEvent events fire.
        self._context_added = True

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the latest customer question and assistant response."""
        messages = event.agent.messages

        customer_query = None
        assistant_response = None

        # Walk backwards so we get the most recent interaction.
        for message in reversed(messages):
            role = message.get("role")
            content = message.get("content", "")

            if isinstance(content, list):
                text_parts = []

                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])

                content = "\n".join(text_parts)

            if not isinstance(content, str):
                continue

            if role == "assistant" and assistant_response is None:
                assistant_response = content

            elif role == "user" and customer_query is None:
                customer_query = content

            if customer_query and assistant_response:
                break

        if not customer_query or not assistant_response:
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (assistant_response, "ASSISTANT"),
                ],
            )

        except Exception as e:
            logging.warning(
                "Could not save support interaction to memory: %s",
                e,
            )

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(
            MessageAddedEvent,
            self.retrieve_customer_context,
        )

        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction,
        )


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    # TODO: Implement the Knowledge Base search
    if not KB_ID:
        return "Knowledge base not configured."

    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )

        results = response.get("retrievalResults", [])

        if not results:
            return "No relevant information found in the knowledge base."

        chunks = []

        for result in results:
            content = result.get("content", {})
            text = content.get("text")

            if text:
                chunks.append(text)

        if not chunks:
            return "No relevant information found in the knowledge base."

        return "\n---\n".join(chunks)

    except Exception as e:
        logger.warning("Knowledge base search failed: %s", e)
        return f"Knowledge base search failed: {e}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    # TODO: Build the code string (use an f-string to inject the arguments)
    code = f"""
import json
import math

loyalty_points = {int(loyalty_points)}
tier = {tier!r}
order_total = {float(order_total)}
product_category = {product_category!r}

earn_rates = {{
    "standard": 1,
    "device": 2,
    "fresh": 5,
}}

tier_rates = {{
    "Silver": 0.00,
    "Gold": 0.10,
    "Platinum": 0.15,
}}

# Points can only be redeemed in blocks of 500.
# Redemption is capped at 50% of the order total.
max_points_value = order_total * 0.50
max_redeemable_points = math.floor(max_points_value / 0.01)
points_redeemed = min(
    loyalty_points // 500 * 500,
    max_redeemable_points,
)
points_value = points_redeemed * 0.01

subtotal_after_points = order_total - points_value

tier_discount_pct = tier_rates.get(tier, 0.00)
tier_discount = subtotal_after_points * tier_discount_pct

final_total = subtotal_after_points - tier_discount
total_savings = points_value + tier_discount

points_earned = math.floor(
    final_total * earn_rates.get(product_category, 1)
)

remaining_points = loyalty_points - points_redeemed

result = {{
    "points_redeemed": points_redeemed,
    "tier_discount_pct": tier_discount_pct,
    "tier_discount": round(tier_discount, 2),
    "points_value": round(points_value, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            result = code_client.invoke(
                "executeCode",
                {
                    "language": "python",
                    "code": code,
                },
            )

        if not result:
            return json.dumps({
                "error": "Code Interpreter returned no result."
            })

        stream = result.get("stream")

        if stream is None:
            return json.dumps({
                "error": "Code Interpreter response did not contain a stream."
            })

        events = list(stream)

        if not events:
            return json.dumps({
                "error": "Code Interpreter stream returned no events."
            })

        event = events[-1]
        event_result = event.get("result", {})

        if event_result.get("isError"):
            structured = event_result.get("structuredContent", {})
            return json.dumps({
                "error": "Code Interpreter execution failed.",
                "stderr": structured.get("stderr", ""),
                "stdout": structured.get("stdout", ""),
            })

        structured = event_result.get("structuredContent", {})
        stdout = structured.get("stdout", "")

        if stdout:
            try:
                calculation = json.loads(stdout)
                return (
                    f"Points redeemed: {calculation['points_redeemed']}\n"
                    f"Tier discount: {calculation['tier_discount_pct'] * 100:.0f}% "
                    f"(${calculation['tier_discount']:.2f})\n"
                    f"Points value discount: ${calculation['points_value']:.2f}\n"
                    f"Final total: ${calculation['final_total']:.2f}\n"
                    f"Total savings: ${calculation['total_savings']:.2f}\n"
                    f"Points earned: {calculation['points_earned']}\n"
                    f"Remaining points: {calculation['remaining_points']}"
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                return stdout.strip()

        return json.dumps(structured)

    except Exception as e:
        logger.warning("Code Interpreter unavailable: %s", e)

        tier_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15,
        }

        tier_discount_pct = tier_rates.get(tier, 0.00)
        tier_discount = order_total * tier_discount_pct
        final_total = order_total - tier_discount

        return json.dumps({
            "points_redeemed": 0,
            "tier_discount_pct": tier_discount_pct,
            "tier_discount": round(tier_discount, 2),
            "points_value": 0.00,
            "final_total": round(final_total, 2),
            "total_savings": round(tier_discount, 2),
            "points_earned": 0,
            "remaining_points": loyalty_points,
            "fallback": True,
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.
    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    prompt = payload.get("prompt", "")
    customer_id = payload.get("customer_id", "anonymous")
    session_id = payload.get("session_id") or str(uuid.uuid4())

    if not prompt:
        return "Please provide a customer support question or request."

    memory_hook = MemoryHook(
        actor_id=customer_id,
        session_id=session_id,
        memory_client=memory_client,
        memory_id=MEMORY_ID,
    )

    browser = AgentCoreBrowser(region=REGION)

    mcp = MCPClient(url=GATEWAY_URL)
    gateway_tools = await mcp.load_tools()

    system_prompt = """
You are an AI customer support assistant for an e-commerce store.

Your job is to help customers with:
1. Order tracking and order information.
2. Returns and refunds.
3. Product information and recommendations.
4. Loyalty program questions and discount calculations.
5. General customer support questions.

Tool usage rules:

- For order tracking, order information, or customer order information,
  use the appropriate order-tracker Gateway tool.
- For refunds, refund status, or return labels, use the appropriate
  refund-processor Gateway tool.
- For product specifications, product information, return policies,
  warranty information, loyalty program details, and other store
  knowledge, use the search_knowledge_base tool.
- For loyalty discount calculations, use calculate_loyalty_discount exactly once.
- Do not perform the loyalty calculation yourself.
- After receiving the tool result, treat its values as authoritative.
- Do not recalculate, reinterpret, or replace any monetary value returned
  by the loyalty calculation tool.
- In the final response, use the exact values returned by the tool for
  points redeemed, points discount, tier discount, final total, total
  savings, points earned, and remaining points.
- Do not call calculate_loyalty_discount again unless the customer
  provides new or corrected calculation inputs.
- For requests involving live websites or web pages, use the browser tool.
- Use information returned by tools as the source of truth.
- Do not invent order details, refund information, product information,
  or loyalty results.
- If required information is missing, ask the customer for it.
- Keep responses clear and helpful.
"""

    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            *gateway_tools,
            search_knowledge_base,
            calculate_loyalty_discount,
            browser.browser,
        ],
        hooks=[memory_hook],
    )

    result = await agent.invoke_async(prompt)

    return str(result)


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    #main()
