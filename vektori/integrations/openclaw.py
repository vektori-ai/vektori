"""OpenClaw persistent memory plugin integration for Vektori."""

from typing import Any

try:
    from openclaw_sdk.plugins import OpenClawPlugin, OpenClawTool, hook
except ImportError:
    class OpenClawPlugin:
        pass

    def hook(*args, **kwargs):
        return lambda f: f

    class OpenClawTool:
        pass

from vektori import Vektori


class VektoriOpenClawPlugin(OpenClawPlugin):
    """
    Vektori integration for OpenClaw.
    Replaces OpenClaw's flat Markdown memory with Vektori's graph memory tools.
    """

    name = "vektori-memory"
    version = "1.0.0"

    def __init__(self, vektori_instance: Vektori):
        self.memory = vektori_instance

    @hook("on_message_received")
    async def capture_message(self, message: dict[str, Any], context: dict[str, Any]) -> None:
        """Passive background sync of interactions to Vektori L2 Sentences."""
        user_id = context.get("user_id", "default_openclaw_user")
        session_id = context.get("session_id", "default_session")

        await self.memory.add(
            messages=[{"role": message.get("role", "user"), "content": message.get("content", "")}],
            user_id=user_id,
            session_id=session_id
        )

    def get_tools(self) -> list:
        """Expose Vektori search natively to OpenClaw agents."""

        async def search_vektori(query: str, depth: str = "l1") -> str:
            """Search the Vektori long-term graph memory."""
            results = await self.memory.search(query=query, user_id="default_openclaw_user", depth=depth)
            output = []
            if results.get("facts"):
                output.append("Facts:\n" + "\n".join(f"- {f['text']}" for f in results["facts"]))
            if results.get("episodes"):
                output.append("Insights:\n" + "\n".join(f"- {ep['text']}" for ep in results["episodes"]))

            return "\n\n".join(output) if output else "No relevant memories found."

        return [search_vektori]

