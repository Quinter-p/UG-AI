from typing import Literal
from langgraph.graph import StateGraph, START, END

from core.state import AgentState
from core.config import load_config
from core.nodes.parse_event import parse_event_node
from core.nodes.identity import identity_node
from core.nodes.relationship import relationship_load_node, relationship_save_node
from core.nodes.relationship_policy import relationship_policy_node
from core.nodes.reflex import reflex_node
from core.nodes.command import command_node
from core.nodes.speak_decision import speak_decision_node
from core.nodes.emotion import emotion_node
from core.nodes.history import history_load_node, history_save_node
from core.nodes.long_memory import long_memory_load_node
from core.nodes.reply_policy import reply_policy_node
from core.nodes.expression_style import expression_style_node
from core.nodes.prompt import prompt_node
from core.nodes.expression import expression_node
from core.nodes.output_filter import output_filter_node


def after_relationship_policy(state: AgentState) -> Literal["end", "reflex"]:
    if state.get("route") == "ignore":
        return "end"
    return "reflex"


def after_reflex(state: AgentState) -> Literal["end", "command"]:
    if state.get("route") == "auto_reply":
        return "end"
    return "command"


def after_command(state: AgentState) -> Literal["end", "speak_decision"]:
    if state.get("route") == "command_reply":
        return "end"
    return "speak_decision"


def after_speak_decision(state: AgentState) -> Literal["end", "emotion"]:
    if state.get("route") == "ignore" or not state.get("should_reply"):
        return "end"
    return "emotion"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("parse_event", parse_event_node)
    builder.add_node("identity", identity_node)
    builder.add_node("relationship_load", relationship_load_node)
    builder.add_node("relationship_policy", relationship_policy_node)
    builder.add_node("reflex", reflex_node)
    builder.add_node("command", command_node)
    builder.add_node("speak_decision", speak_decision_node)
    builder.add_node("emotion", emotion_node)
    builder.add_node("history_load", history_load_node)
    builder.add_node("long_memory_load", long_memory_load_node)
    builder.add_node("reply_policy", reply_policy_node)
    builder.add_node("expression_style", expression_style_node)
    builder.add_node("prompt", prompt_node)
    builder.add_node("expression", expression_node)
    builder.add_node("output_filter", output_filter_node)
    builder.add_node("history_save", history_save_node)
    builder.add_node("relationship_save", relationship_save_node)

    builder.add_edge(START, "parse_event")
    builder.add_edge("parse_event", "identity")
    builder.add_edge("identity", "relationship_load")
    builder.add_edge("relationship_load", "relationship_policy")
    builder.add_conditional_edges("relationship_policy", after_relationship_policy, {"end": END, "reflex": "reflex"})

    builder.add_conditional_edges("reflex", after_reflex, {"end": END, "command": "command"})
    builder.add_conditional_edges("command", after_command, {"end": END, "speak_decision": "speak_decision"})
    builder.add_conditional_edges("speak_decision", after_speak_decision, {"end": END, "emotion": "emotion"})

    builder.add_edge("emotion", "history_load")
    builder.add_edge("history_load", "long_memory_load")
    builder.add_edge("long_memory_load", "reply_policy")
    builder.add_edge("reply_policy", "expression_style")
    builder.add_edge("expression_style", "prompt")
    builder.add_edge("prompt", "expression")
    builder.add_edge("expression", "output_filter")
    builder.add_edge("output_filter", "history_save")
    builder.add_edge("history_save", "relationship_save")
    builder.add_edge("relationship_save", END)

    return builder.compile()


class AgentCore:
    def __init__(self):
        self.config = load_config()
        self.graph = build_graph()

    def handle_event(self, event):
        initial = {"raw_event": event, "config": self.config, "logs": []}
        return self.graph.invoke(initial)
