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




















import gradio as gr
from datetime import datetime, timedelta
from app.orchestrator.orchestrator import Orchestrator

default_log_folder_path = "/var/logs"
default_from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
default_to_date = datetime.now().strftime("%Y-%m-%d")


def submit(log_folder_path, from_date, to_date, user_query, keyword, previous_response):

    config = {
        "log_folder_path": log_folder_path,
        "from_date": from_date,
        "to_date": to_date,
        "user_query": user_query,
        "keyword": keyword,
        "previous_response": previous_response,
    }

    orchestrator = Orchestrator(config)
    response = orchestrator.run()

    return response, response


def clear():
    return default_log_folder_path, default_from_date, default_to_date, "", "", "", None


with gr.Blocks() as app:

    log_folder_path = gr.Textbox(label="Log Folder Path", value=default_log_folder_path)

    with gr.Row():
        from_date = gr.DateTime(label="From Date", type="string", include_time=False, value=default_from_date)
        to_date = gr.DateTime(label="To Date", type="string", include_time=False, value=default_to_date)
        keyword = gr.Textbox(label="Keyword")

    user_query = gr.Textbox(label="User Query", lines=5)

    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear")

    output = gr.Textbox(label="LLM Response", lines=20)
    previous_response = gr.State()

    submit_btn.click(
        submit,
        inputs=[log_folder_path, from_date, to_date, user_query, keyword, previous_response],
        outputs=[output, previous_response],
    )

    clear_btn.click(
        clear,
        outputs=[log_folder_path, from_date, to_date, user_query, keyword, output, previous_response],
    )


if __name__ == "__main__":
    app.launch()








from typing import Dict, List, Optional
from pydantic import BaseModel


class AgentDecision(BaseModel):
    next_agents: List[str]
    agent_queries: Dict[str, str]


class AgentState(BaseModel):
    messages: list
    config: dict
    decision: Optional[AgentDecision] = None
    partial_answers: Dict[str, str] = {}
    final_answer: Optional[str] = None




ROUTER_PROMPT = """
You are an enterprise AI router.

Your job:
1. Decide which agents should answer the user query.
2. Extract the relevant part of the query for each agent.

Available agents:
- wiki_agent → general knowledge, explanations, definitions
- log_agent → logs, metrics, trades, database queries
- clarify → if unclear

Return STRICT JSON:

{
  "next_agents": ["wiki_agent", "log_agent"],
  "agent_queries": {
     "wiki_agent": "what is block trade",
     "log_agent": "Did we process block trade no 403245"
  }
}

Rules:
- DO NOT use keywords heuristics. Use reasoning.
- Trim irrelevant parts for each agent.
- If only one agent needed, return only that agent.
- If unclear, return clarify.
- Do NOT add commentary outside JSON.

User Query:
{query}
Config:
{config}
"""







import json
from langchain.schema import HumanMessage
from .schemas import AgentState, AgentDecision
from .llm import get_llm
from .prompts import ROUTER_PROMPT


llm = get_llm()

# =====================================================
# Router Node
# =====================================================
def router_node(state: AgentState) -> dict:
    user_query = state.messages[0].content
    config = state.config

    prompt = ROUTER_PROMPT.format(query=user_query, config=config)
    response = llm.invoke(prompt).content

    decision_json = json.loads(response)
    decision = AgentDecision(**decision_json)

    return {"decision": decision}


# =====================================================
# Wiki Agent
# =====================================================
def wiki_agent_node(state: AgentState) -> dict:
    query = state.decision.agent_queries["wiki_agent"]
    result = f"Wiki Agent Answer:\n(simulated explanation for '{query}')"

    return {"partial_answers": {**state.partial_answers, "wiki_agent": result}}


# =====================================================
# Log Agent
# =====================================================
def log_agent_node(state: AgentState) -> dict:
    query = state.decision.agent_queries["log_agent"]
    result = f"Log Agent Result:\n(simulated logs lookup for '{query}')"

    return {"partial_answers": {**state.partial_answers, "log_agent": result}}


# =====================================================
# Clarify Node
# =====================================================
def clarify_node(state: AgentState) -> dict:
    return {
        "final_answer": "Your query is unclear. Please clarify what you want."
    }


# =====================================================
# Merge Node
# =====================================================
def merge_node(state: AgentState) -> dict:
    answers = state.partial_answers

    if not answers:
        return {"final_answer": "No agent returned an answer."}

    combined = "\n\n".join(
        f"### {agent.upper()}\n{answer}" for agent, answer in answers.items()
    )

    return {"final_answer": combined}







from langgraph.graph import StateGraph, START, END
from .schemas import AgentState
from .nodes import (
    router_node,
    wiki_agent_node,
    log_agent_node,
    clarify_node,
    merge_node
)


def route_after_router(state: AgentState):
    return state.decision.next_agents


def build_graph():
    workflow = StateGraph(state_schema=AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("wiki_agent", wiki_agent_node)
    workflow.add_node("log_agent", log_agent_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("merge", merge_node)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "wiki_agent": "wiki_agent",
            "log_agent": "log_agent",
            "clarify": "clarify"
        }
    )

    workflow.add_edge("wiki_agent", "merge")
    workflow.add_edge("log_agent", "merge")
    workflow.add_edge("merge", END)
    workflow.add_edge("clarify", END)

    return workflow.compile()




from langchain.schema import HumanMessage
from .graph import build_graph


class Orchestrator:
    def __init__(self):
        self.graph = build_graph()

    def run(self, config: dict):
        query = config["user_query"]

        state = {
            "messages": [HumanMessage(content=query)],
            "config": config,
            "decision": None,
            "partial_answers": {},
            "final_answer": None
        }

        result = self.graph.invoke(state)
        return result["final_answer"]














from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

class GetLogsInput(BaseModel):
    from_date: str = Field(..., description="YYYY-MM-DD")
    to_date: str = Field(..., description="YYYY-MM-DD")
    keywords: List[str] = Field(..., description="Keywords to filter logs")


class LogAnalyzerTools(LogFileSource):
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings

    def _get_relevant_logs(self, from_date, to_date, keywords):
        return self.get_relevant_logs(from_date, to_date, keywords)


log_tool_instance = LogAnalyzerTools(settings)


@tool(args_schema=GetLogsInput)
def get_relevant_logs(from_date: str, to_date: str, keywords: List[str]) -> str:
    """Get relevant logs between dates filtered by keywords"""

    from_date_obj = datetime.strptime(from_date, "%Y-%m-%d")
    to_date_obj = datetime.strptime(to_date, "%Y-%m-%d")

    return log_tool_instance._get_relevant_logs(from_date_obj, to_date_obj, keywords)

