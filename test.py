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








import json
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from app.orchestrator.graph_runner import run_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROUTER_PROMPT = """
You are an AI routing controller in a multi-agent system.

Your job:
1. Decide which agents should run.
2. Extract the relevant sub-question for each agent.

Agents:
- wiki_agent: explanations, definitions, tutorials, general knowledge.
- log_agent: logs, trades, debugging, monitoring, system data.
- clarify: unclear or missing info.

Return STRICT JSON ONLY:

{
  "next_agents": ["wiki_agent" | "log_agent" | "clarify"],
  "agent_inputs": {
    "wiki_agent": "<trimmed question>",
    "log_agent": "<trimmed question>"
  }
}

Rules:
- No explanations or markdown.
- Do not repeat full user query unless necessary.
- Extract only relevant parts for each agent.
- If unclear return {"next_agents":["clarify"],"agent_inputs":{}}.

Request Context:
<CONFIG_JSON>
"""


class Orchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_query = config.get("user_query", "")

        # Replace with corporate model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

    # ---------- LLM Planner / Router ---------- #
    def _route(self) -> Dict[str, Any]:
        messages = [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=json.dumps(self.config, indent=2)),
        ]

        response = self.llm.invoke(messages).content
        logger.info(f"Router output: {response}")

        try:
            decision = json.loads(response)
        except Exception:
            logger.exception("Router JSON parse failed")
            return {"next_agents": ["clarify"], "agent_inputs": {}}

        # Validate output
        allowed = {"wiki_agent", "log_agent", "clarify"}
        decision["next_agents"] = [a for a in decision.get("next_agents", []) if a in allowed]
        decision.setdefault("agent_inputs", {})

        if not decision["next_agents"]:
            decision["next_agents"] = ["clarify"]

        return decision

    # ---------- Run LangGraph ---------- #
    def run(self) -> str:
        if not self.user_query:
            return "No user query provided."

        decision = self._route()
        logger.info(f"Routing decision: {decision}")

        state = {
            "messages": [{"role": "user", "content": self.user_query}],
            "decision": decision,
            "agent_inputs": decision.get("agent_inputs", {}),
            "partial_answers": {},
            "final_answer": None,
            "config": self.config,
        }

        result = run_graph(state)
        return result["final_answer"]







# Router node (no logic, orchestrator already decided)
def router_node(state):
    return state


# ---------------- Wiki Agent ---------------- #
def wiki_agent_node(state):
    query = state["agent_inputs"].get("wiki_agent")
    if not query:
        return {}

    return {
        "partial_answers": {
            **state.get("partial_answers", {}),
            "wiki_agent": f"Wiki Agent Answer: {query}"
        }
    }


# ---------------- Log Agent ---------------- #
def log_agent_node(state):
    query = state["agent_inputs"].get("log_agent")
    if not query:
        return {}

    folder = state["config"].get("log_folder_path")

    return {
        "partial_answers": {
            **state.get("partial_answers", {}),
            "log_agent": f"Log Agent processed '{query}' in folder {folder}"
        }
    }


# ---------------- Clarify ---------------- #
def clarify_node(state):
    return {
        "final_answer": "Your request is unclear. Please clarify your intent."
    }


# ---------------- Merge ---------------- #
def merge_node(state):
    answers = state.get("partial_answers", {})

    if not answers:
        return {"final_answer": "No agent returned an answer."}

    combined = "\n\n".join(
        f"### {agent.upper()}\n{answer}"
        for agent, answer in answers.items()
    )

    return {"final_answer": combined}










from langgraph.graph import StateGraph, START, END
from app.orchestrator.nodes import (
    router_node,
    wiki_agent_node,
    log_agent_node,
    clarify_node,
    merge_node,
)

AgentState = dict

workflow = StateGraph(state_schema=AgentState)

workflow.add_node("router", router_node)
workflow.add_node("wiki_agent", wiki_agent_node)
workflow.add_node("log_agent", log_agent_node)
workflow.add_node("clarify", clarify_node)
workflow.add_node("merge", merge_node)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    lambda s: s["decision"]["next_agents"],
    {
        "wiki_agent": "wiki_agent",
        "log_agent": "log_agent",
        "clarify": "clarify",
    },
)

workflow.add_edge("wiki_agent", "merge")
workflow.add_edge("log_agent", "merge")
workflow.add_edge("merge", END)
workflow.add_edge("clarify", END)

graph = workflow.compile()













