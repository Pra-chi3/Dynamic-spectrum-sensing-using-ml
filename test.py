from typing import Literal, Annotated, List, Dict
from typing_extensions import TypedDict
import json

from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# =====================================================
# Router Decision Schema (MULTI-AGENT)
# =====================================================
class RouteDecision(BaseModel):
    next_agents: List[Literal["sql_agent", "wiki_agent", "clarify"]] = Field(
        description="One or more agents that should handle this query"
    )
    reasoning: str


# =====================================================
# Graph State
# =====================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    decision: RouteDecision | None
    partial_answers: Dict[str, str]
    final_answer: str | None


# =====================================================
# Router Prompt
# =====================================================
ROUTER_SYSTEM_PROMPT = """
You are a smart router that decides which specialized agents should answer the user.

Available agents:
- sql_agent  → database queries, metrics, counts, sums, filters, reports, trends
- wiki_agent → explanations, definitions, concepts, history, general knowledge
- clarify    → ambiguous or unclear questions

Rules:
- If the question needs BOTH data and explanation → select BOTH agents
- If it fits only one → select that one
- If unclear → select ["clarify"]

Respond ONLY with valid JSON in this format:

{
  "next_agents": ["sql_agent", "wiki_agent"],
  "reasoning": "short explanation"
}
"""


# =====================================================
# Router Node
# =====================================================
def router_node(state: AgentState) -> dict:
    user_question = state["messages"][0].content

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    try:
        raw = llm.invoke(messages)

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("No JSON found")

        parsed = json.loads(raw[start:end])
        decision = RouteDecision.model_validate(parsed)

    except (ValidationError, json.JSONDecodeError, Exception) as e:
        decision = RouteDecision(
            next_agents=["clarify"],
            reasoning=f"Router failure: {str(e)}"
        )

    state["messages"].append(
        AIMessage(
            content=f"[Routing]\nAgents: {decision.next_agents}\nReason: {decision.reasoning}"
        )
    )

    return {
        "decision": decision,
        "messages": state["messages"],
        "partial_answers": {}
    }


# =====================================================
# SQL Agent
# =====================================================
def sql_agent_node(state: AgentState) -> dict:
    question = state["messages"][0].content
    result = (
        f"SQL Agent Answer:\n"
        f"(simulated query result for: '{question}')"
    )

    return {
        "partial_answers": {
            **state.get("partial_answers", {}),
            "sql_agent": result
        }
    }


# =====================================================
# Wiki Agent
# =====================================================
def wiki_agent_node(state: AgentState) -> dict:
    question = state["messages"][0].content
    result = (
        f"Wiki Agent Answer:\n"
        f"(simulated explanation for: '{question}')"
    )

    return {
        "partial_answers": {
            **state.get("partial_answers", {}),
            "wiki_agent": result
        }
    }


# =====================================================
# Clarify Agent
# =====================================================
def clarify_node(state: AgentState) -> dict:
    return {
        "final_answer": (
            "Your question is a bit unclear.\n\n"
            "Is it about:\n"
            "• database / metrics / reports\n"
            "• general knowledge / explanation\n\n"
            "Please clarify or rephrase."
        )
    }


# =====================================================
# Merge Node (JOIN)
# =====================================================
def merge_node(state: AgentState) -> dict:
    answers = state.get("partial_answers", {})

    if not answers:
        return {"final_answer": "No agent returned an answer."}

    combined = "\n\n".join(
        f"### {agent.upper()}\n{answer}"
        for agent, answer in answers.items()
    )

    return {"final_answer": combined}


# =====================================================
# Router → Fan-out logic
# =====================================================
def route_after_router(state: AgentState):
    return state["decision"].next_agents


# =====================================================
# Build Graph
# =====================================================
workflow = StateGraph(state_schema=AgentState)

workflow.add_node("router", router_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("wiki_agent", wiki_agent_node)
workflow.add_node("clarify", clarify_node)
workflow.add_node("merge", merge_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_after_router,
    {
        "sql_agent": "sql_agent",
        "wiki_agent": "wiki_agent",
        "clarify": "clarify",
    }
)

workflow.add_edge("sql_agent", "merge")
workflow.add_edge("wiki_agent", "merge")
workflow.add_edge("merge", END)

workflow.add_edge("clarify", END)

graph = workflow.compile()


# =====================================================
# Runner
# =====================================================
def ask(question: str) -> str:
    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "decision": None,
        "partial_answers": {},
        "final_answer": None
    })
    return result["final_answer"]


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":
    print(ask("Explain photosynthesis and tell me how many customers signed up last month"))




















You are a Confluence Knowledge Retrieval Agent.

Your task is to:
1. Search Confluence based on the user question.
2. Identify relevant page IDs or titles from search results.
3. Fetch content for ALL relevant pages.
4. Summarize the information.
5. Return ALL page links.

You MUST follow the EXACT format below. This format is STRICT and REQUIRED.

---------------- REQUIRED FORMAT ----------------

Thought: <your reasoning>

Action: <tool name>

Action Input: "<single-line string argument>"

Observation: <tool output>

(Repeat Thought/Action/Action Input/Observation until finished)

Final Answer:
Question: <original user question>
Summary: <final summarized answer from all pages>
Links:
- <url1>
- <url2>
- <url3>

---------------- CRITICAL FORMAT RULES ----------------

1. Thought, Action, Action Input, Observation MUST each start on a NEW LINE.
2. NEVER place Action on the same line as Thought.
3. NEVER place Action Input on the same line as Action.
4. NEVER place Observation on the same line as Action Input.
5. Action Input MUST be a single-line string wrapped in double quotes.
6. Only call ONE Action per step.
7. Do NOT output JSON, markdown, bullet lists, or extra formatting in Thought/Action blocks.
8. Do NOT hallucinate page content. Always call get_page_content before summarizing.
9. If format is violated, rewrite the response in correct format before continuing.
10. If search results are irrelevant, refine the search query and search again.
11. Fetch content for ALL relevant pages found in search results.

---------------- TOOL DEFINITIONS ----------------

Available tools:

search_confluence(query: string)
get_page_content(page_id_or_title: string)

---------------- LINK CONSTRUCTION RULE ----------------

If page metadata contains "_Links" with "base" and "webui":
Construct URL as:
URL = base + webui

Collect ALL URLs and include them in Final Answer.

---------------- WORKFLOW RULES ----------------

Step 1: Search using the user question.
Step 2: Identify relevant page IDs or titles.
Step 3: Call get_page_content for EACH relevant page.
Step 4: Summarize across all pages.
Step 5: Return summary and all links.

---------------- EXAMPLE ----------------

Thought: I need to search Confluence based on the user question

Action: search_confluence

Action Input: "trade ticket system"

Observation: Found pages Trade Ticket Overview, Trade Processing Flow

Thought: These pages are relevant, I will fetch the first page content

Action: get_page_content

Action Input: "Trade Ticket Overview"

Observation: Page content retrieved

Thought: I should fetch the second relevant page

Action: get_page_content

Action Input: "Trade Processing Flow"

Observation: Page content retrieved

Thought: I have all information and links, I will summarize and return final answer

Final Answer:
Question: What is a trade ticket system?
Summary: <combined summary>
Links:
- https://confluence.company/wiki/page1
- https://confluence.company/wiki/page2

