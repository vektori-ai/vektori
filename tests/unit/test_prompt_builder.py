from vektori.context import LoadedAgentContext
from vektori.memory.profile import ProfilePatch
from vektori.memory.window import WindowState
from vektori.prompts import build_messages, build_prompt_result


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
        max_context_tokens=1200,
        reserve_response_tokens=200,
    )

    assert messages[0]["role"] == "system"
    assert "You are concise." in messages[0]["content"]
    assert "response_style.verbosity = short" in messages[0]["content"]
    assert any("Retrieved Memory" in message["content"] for message in messages)
    assert any("Conversation Summary" in message["content"] for message in messages)
    assert messages[-1] == {"role": "user", "content": "Hello"}


def test_prompt_builder_trims_memory_and_old_messages_under_budget():
    context = LoadedAgentContext(persona="You are concise.")
    window_state = WindowState(
        recent_messages=[
            {"role": "assistant", "content": "x" * 180},
            {"role": "user", "content": "y" * 180},
            {"role": "assistant", "content": "z" * 180},
            {"role": "user", "content": "latest user turn"},
        ],
        rolling_summary="Conversation Summary\n- Active goals: " + ("s" * 150),
        estimated_tokens=400,
        compaction_count=1,
    )

    result = build_prompt_result(
        context=context,
        profile_patches=[],
        memories={
            "facts": [{"text": "fact 1 " + ("f" * 80)}, {"text": "fact 2 " + ("g" * 80)}],
            "episodes": [{"text": "episode 1 " + ("e" * 120)}],
            "sentences": [{"text": "sentence 1 " + ("s" * 120)}],
        },
        window_state=window_state,
        max_context_tokens=160,
        reserve_response_tokens=100,
    )

    assert result.prompt_debug["trimmed"]["sentences"] >= 1
    assert result.prompt_debug["trimmed"]["episodes"] >= 1
    assert result.prompt_debug["trimmed"]["facts"] >= 1
    assert result.messages[-1] == {"role": "user", "content": "latest user turn"}
    assert result.prompt_debug["estimated_prompt_tokens"] <= result.prompt_debug["budget_tokens"]
