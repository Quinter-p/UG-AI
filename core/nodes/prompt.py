from core.prompts.persona import build_persona_fragment
from core.prompts.identity import build_identity_fragment
from core.prompts.relationship import build_relationship_fragment
from core.prompts.relationship_policy import build_relationship_policy_fragment
from core.prompts.emotion import build_emotion_fragment
from core.prompts.memory import build_memory_fragment
from core.prompts.long_memory import build_world_lore_fragment, build_conversation_memory_fragment
from core.prompts.fact_memory import build_fact_memory_fragment
from core.prompts.reflection_memory import build_reflection_memory_fragment
from core.prompts.task_memory import build_task_memory_fragment
from core.prompts.tool_bus import build_tool_bus_fragment
from core.prompts.style import build_style_fragment
from core.prompts.expression_style import build_expression_style_fragment
from core.prompts.speak import build_speak_decision_fragment

def prompt_node(state):
    config = state.get("config") or {}
    reply_policy = state.get("reply_policy") or {}

    persona_fragment = build_persona_fragment(config)
    identity_fragment = build_identity_fragment(
        state.get("identity") or {},
        is_master=bool(state.get("is_master", False)),
    )
    relationship_fragment = build_relationship_fragment(state.get("relationship_state") or {})
    relationship_policy_fragment = build_relationship_policy_fragment(state.get("relationship_policy") or {})
    emotion_fragment = build_emotion_fragment(state.get("emotion_state") or {})
    world_lore_fragment = build_world_lore_fragment(state.get("world_lore_text", ""))
    conversation_memory_fragment = build_conversation_memory_fragment(state.get("conversation_memory_text", ""))
    fact_memory_fragment = build_fact_memory_fragment(state.get("fact_memory_text", ""))
    reflection_memory_fragment = build_reflection_memory_fragment(state.get("reflection_memory_text", ""))
    task_memory_fragment = build_task_memory_fragment(state.get("task_memory_text", ""))
    tool_bus_fragment = build_tool_bus_fragment(state.get("tool_bus_text", ""))
    short_memory_fragment = build_memory_fragment(
        rolling_summary=state.get("rolling_summary", ""),
        turns=state.get("history_turns", []),
        last_user_message=state.get("last_user_message", ""),
    )
    style_fragment = build_style_fragment(config, reply_policy=reply_policy)
    expression_style_fragment = build_expression_style_fragment(state.get("expression_style") or {})
    speak_decision_fragment = build_speak_decision_fragment(state.get("speak_decision") or {})
    current_input = str(state.get("clean_text") or state.get("raw_message") or "").strip()

    prompt_fragments = {
        "persona": persona_fragment,
        "identity": identity_fragment,
        "relationship": relationship_fragment,
        "relationship_policy": relationship_policy_fragment,
        "emotion": emotion_fragment,
        "world_lore": world_lore_fragment,
        "conversation_memory": conversation_memory_fragment,
        "fact_memory": fact_memory_fragment,
        "reflection_memory": reflection_memory_fragment,
        "task_memory": task_memory_fragment,
        "tool_bus": tool_bus_fragment,
        "short_memory": short_memory_fragment,
        "reply_policy": str(reply_policy),
        "reply_style": style_fragment,
        "expression_style": expression_style_fragment,
        "speak_decision": speak_decision_fragment,
    }

    prompt_messages = [
        {"role": "system", "content": persona_fragment},
        {"role": "system", "content": "[身份识别]\n" + identity_fragment},
        {"role": "system", "content": "[关系状态]\n" + relationship_fragment},
        {"role": "system", "content": "[关系行为策略]\n" + relationship_policy_fragment},
        {"role": "system", "content": "[内部情绪状态]\n" + emotion_fragment},
        {"role": "system", "content": "[世界观设定]\n" + world_lore_fragment},
        {"role": "system", "content": "[长期个人记忆]\n" + conversation_memory_fragment},
        {"role": "system", "content": "[事实记忆]\n" + fact_memory_fragment},
        {"role": "system", "content": "[反思记忆]\n" + reflection_memory_fragment},
        {"role": "system", "content": "[任务登记表]\n" + task_memory_fragment},
        {"role": "system", "content": "[工具总线]\n" + tool_bus_fragment},
        {"role": "system", "content": "[短期对话]\n" + short_memory_fragment},
        {"role": "system", "content": "[发言决策]\n" + speak_decision_fragment},
        {"role": "system", "content": "[本轮回复预算]\n" + style_fragment},
        {"role": "system", "content": "[本轮表达风格]\n" + expression_style_fragment},
        {"role": "user", "content": current_input},
    ]

    rel = state.get("relationship_state") or {}
    relp = state.get("relationship_policy") or {}
    expr = state.get("expression_style") or {}
    speak = state.get("speak_decision") or {}

    return {
        "prompt_fragments": prompt_fragments,
        "prompt_messages": prompt_messages,
        "prompt_meta": {
            "builder_strategy": "fragments_with_tool_bus",
            "speak_reason": speak.get("reason", ""),
            "speak_mode": speak.get("mode", ""),
            "reply_mode": reply_policy.get("mode", "unknown"),
            "reply_reason": reply_policy.get("reason", ""),
            "expression_tone": expr.get("tone", ""),
            "expression_distance": expr.get("distance", ""),
            "history_turns_used": len(state.get("history_turns", [])),
            "world_lore_files_used": state.get("world_lore_files", []),
            "conversation_memory_items_used": len(state.get("conversation_memory_items", [])),
            "fact_memory_items_used": len(state.get("fact_memory_items", [])),
            "reflection_memory_items_used": len(state.get("reflection_memory_items", [])),
            "task_items_used": len(state.get("task_items", [])),
            "tool_call_items_used": len(state.get("tool_call_items", [])),
            "has_rolling_summary": bool(state.get("rolling_summary", "")),
            "has_last_user_message": bool(state.get("last_user_message", "")),
            "relationship_role": rel.get("role", "unknown"),
            "relationship_attitude": rel.get("attitude", "neutral"),
            "relationship_stance": relp.get("stance", "neutral"),
        },
    }
