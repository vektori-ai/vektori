"""Context file loading for the conversational harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoadedAgentContext:
    persona: str = ""
    instructions: list[str] = field(default_factory=list)
    response_style: dict[str, Any] = field(default_factory=dict)
    memory_policy: dict[str, Any] = field(default_factory=dict)
    extra_sections: dict[str, str] = field(default_factory=dict)


class AgentContextLoader:
    """Loads `agents.md` or `vektori.yaml` into a normalized structure."""

    def __init__(self, context_path: str | None = None) -> None:
        self.context_path = context_path

    def load(self) -> LoadedAgentContext:
        path = self._resolve_path()
        if path is None:
            return LoadedAgentContext()
        if path.suffix in {".yaml", ".yml"}:
            return self._load_yaml(path)
        return self._load_markdown(path)

    def _resolve_path(self) -> Path | None:
        candidates: list[Path] = []
        if self.context_path is not None:
            candidates.append(Path(self.context_path))
        else:
            cwd = Path.cwd()
            candidates.extend(
                [
                    cwd / "agents.md",
                    cwd / "vektori.yaml",
                    cwd / "vektori.yml",
                ]
            )
            candidates.extend(
                [
                    path / "agents.md"
                    for path in [*cwd.parents][:3]
                ]
            )
            candidates.extend(
                [
                    path / "vektori.yaml"
                    for path in [*cwd.parents][:3]
                ]
            )
            candidates.extend(
                [
                    path / "vektori.yml"
                    for path in [*cwd.parents][:3]
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_yaml(self, path: Path) -> LoadedAgentContext:
        raw = yaml.safe_load(path.read_text()) or {}
        agent = raw.get("agent", {})
        return LoadedAgentContext(
            persona=agent.get("persona", "") or "",
            instructions=[str(item) for item in (agent.get("instructions", []) or [])],
            response_style=agent.get("response_style", {}) or {},
            memory_policy=agent.get("memory", {}) or {},
            extra_sections={
                key: value
                for key, value in agent.items()
                if key not in {"name", "persona", "instructions", "response_style", "memory"}
            },
        )

    def _load_markdown(self, path: Path) -> LoadedAgentContext:
        sections: dict[str, list[str]] = {}
        current_section = "root"
        sections[current_section] = []

        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                current_section = stripped[3:].strip().lower()
                sections.setdefault(current_section, [])
                continue
            sections.setdefault(current_section, []).append(line)

        persona = self._normalize_text_block(sections.get("persona", []))
        instructions = self._normalize_bullets(sections.get("instructions", []))
        response_style = self._coerce_section(sections.get("response style", []))
        memory_policy = self._coerce_section(sections.get("memory policy", []))
        extras = {
            key: self._normalize_text_block(value)
            for key, value in sections.items()
            if key not in {"root", "persona", "instructions", "response style", "memory policy"}
            and self._normalize_text_block(value)
        }
        return LoadedAgentContext(
            persona=persona,
            instructions=instructions,
            response_style=response_style,
            memory_policy=memory_policy,
            extra_sections=extras,
        )

    def _normalize_bullets(self, lines: list[str]) -> list[str]:
        items: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- "):
                items.append(stripped[2:].strip())
            elif stripped:
                items.append(stripped)
        return items

    def _normalize_text_block(self, lines: list[str]) -> str:
        return "\n".join(line.strip() for line in lines if line.strip()).strip()

    def _coerce_section(self, lines: list[str]) -> dict[str, Any]:
        bullets = self._normalize_bullets(lines)
        if not bullets:
            notes = self._normalize_text_block(lines)
            return {"notes": notes} if notes else {}

        parsed: dict[str, Any] = {}
        notes: list[str] = []
        for bullet in bullets:
            if ":" in bullet:
                key, value = bullet.split(":", 1)
                parsed[self._slugify(key)] = value.strip()
            else:
                notes.append(bullet)
        if notes:
            parsed["notes"] = notes
        return parsed

    def _slugify(self, value: str) -> str:
        return value.strip().lower().replace(" ", "_")
