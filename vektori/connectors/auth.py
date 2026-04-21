"""AuthStore for managing user credentials for connectors."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuthStore:
    """
    File-based token store at ~/.vektori/tokens/{user_id}/{platform}.json.
    Holds {access_token, refresh_token, expiry, scopes}.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        if base_dir is None:
            base_dir = os.path.expanduser("~/.vektori/tokens")
        self.base_dir = Path(base_dir)

    def _get_path(self, user_id: str, platform: str) -> Path:
        return self.base_dir / user_id / f"{platform}.json"

    def get_token(self, user_id: str, platform: str) -> Optional[Dict[str, Any]]:
        """Retrieve the authentication details for a specific user and platform."""
        path = self._get_path(user_id, platform)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def set_token(
        self,
        user_id: str,
        platform: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expiry: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        """Save the authentication details securely."""
        path = self._get_path(user_id, platform)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry": expiry,
            "scopes": scopes or [],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def clear_token(self, user_id: str, platform: str) -> bool:
        """Remove the authentication details."""
        path = self._get_path(user_id, platform)
        if path.exists():
            path.unlink()
            return True
        return False
