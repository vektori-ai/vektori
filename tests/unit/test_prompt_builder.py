from vektori.context import LoadedAgentContext
from vektori.memory.profile import ProfilePatch
from vektori.memory.window import WindowState
from vektori.prompts import build_messages


def test_build_messages_includes_system_memory_summary_and_turns():
    context = LoadedAgentContext(
        persona="You are concise.",
        instructions=["Prefer direct answers."],
    )
    patches = [
        ProfilePatch(
            key="response_style.verbosity",
            value="short",
            reason="user asked for short answers",
            source="explicit_user_request",
            observer_id="agent-1",
            observed_id="user-1",
            confidence=0.9,
        )
    ]
    window_state = WindowState(
        recent_messages=[{"role": "user", "content": "Hello"}],
        rolling_summary="Conversation Summary\n- Active goals: demo",
        estimated_tokens=25,
        compaction_count=1,
    )

    messages = build_messages(
        context=context,
        profile_patches=patches,
        memories={"facts": [{"text": "User prefers short answers."}]},
        window_state=window_state,
    )

    assert messages[0]["role"] == "system"
    assert "You are concise." in messages[0]["content"]
    assert "response_style.verbosity = short" in messages[0]["content"]
    assert any("Retrieved Memory" in message["content"] for message in messages)
    assert any("Conversation Summary" in message["content"] for message in messages)
    assert messages[-1] == {"role": "user", "content": "Hello"}
