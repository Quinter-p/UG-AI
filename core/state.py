from typing import TypedDict, Literal, Optional, Any

Route = Literal["continue", "ignore", "auto_reply", "command_reply", "llm_reply"]

class AgentState(TypedDict, total=False):
    raw_event: dict
    message_type: str
    user_id: str
    group_id: Optional[str]
    self_id: str

    raw_message: str
    clean_text: str
    should_reply: bool

    identity: dict
    is_master: bool
    relationship_state: dict
    relationship_policy: dict

    route: Route
    final_reply: str

    emotion_state: dict

    # Speak decision
    speak_decision: dict

    # Short-term memory
    session_key: str
    history_turns: list[dict]
    rolling_summary: str
    last_user_message: str

    # Long-term context
    world_lore_text: str
    world_lore_files: list[str]
    conversation_memory_items: list[dict]
    conversation_memory_text: str

    # Reply policy / budget
    reply_policy: dict

    # Affective expression style
    expression_style: dict

    # Prompt assembly
    prompt_fragments: dict
    prompt_messages: list[dict]
    prompt_meta: dict

    # Model/output
    llm_output: str
    assistant_short_summary: str
    usage_metadata: dict

    config: dict
    logs: list[str]
