"""GitHub Connector for Vektori.

Ingests Repository Issues, PRs, and Discussions into Vektori memory.
Maps GitHub data to Vektori documents using `Vektori.add_document()`.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from vektori.client import Vektori
from vektori.connectors.auth import AuthStore
from vektori.connectors.base import Connector

logger = logging.getLogger(__name__)


class GitHubConnector(Connector):
    """
    Ingests GitHub Issues and comments as conversational memories.
    Uses PyGithub (optional dependency).
    """

    def __init__(self, repo_name: str, auth_store: AuthStore | None = None) -> None:
        self.repo_name = repo_name
        self.auth_store = auth_store or AuthStore()

    @property
    def platform(self) -> str:
        return "github"

    async def ingest(
        self,
        user_id: str,
        vektori: Vektori,
        since: datetime | None = None,
    ) -> int:
        """
        Pulls GitHub issues and their comments for the configured repository.
        """
        try:
            from github import Github
            from github.GithubException import GithubException
        except ImportError as e:
            raise ImportError(
                "PyGithub is required for the GitHub connector. "
                "Install it with: pip install PyGithub"
            ) from e

        token_data = self.auth_store.get_token(user_id, self.platform)
        if not token_data or not token_data.get("access_token"):
            logger.error(f"No GitHub access token found for user {user_id}")
            return 0

        g = Github(token_data["access_token"])
        count = 0

        try:
            repo = g.get_repo(self.repo_name)
            
            kwargs: dict[str, Any] = {"state": "all"}
            if since:
                kwargs["since"] = since
                
            issues = repo.get_issues(**kwargs)

            for issue in issues:
                # We want to skip raw pull requests if we just want issues
                # (PyGithub returns PRs as issues too, but we can detect them)
                
                source_id = f"{self.repo_name}/issues/{issue.number}"
                
                # Combine issue body and comments into a readable chronological "document"
                content_parts = [
                    f"Title: {issue.title}",
                    f"State: {issue.state}",
                    f"Author: {issue.user.login}",
                    f"Body:\n{issue.body or 'No description provided.'}",
                ]
                
                comments = issue.get_comments()
                for comment in comments:
                    content_parts.append(
                        f"\n--- Comment by {comment.user.login} at {comment.created_at} ---\n{comment.body}"
                    )
                
                document_content = "\n".join(content_parts)
                document_time = issue.updated_at.replace(tzinfo=timezone.utc)
                
                # Insert into Vektori
                await vektori.add_document(
                    content=document_content,
                    source=self.platform,
                    source_id=source_id,
                    user_id=user_id,
                    document_time=document_time,
                    metadata={
                        "repo": self.repo_name,
                        "issue_number": issue.number,
                        "state": issue.state,
                        "author": issue.user.login,
                        "labels": [lbl.name for lbl in issue.labels],
                        "is_pr": issue.pull_request is not None,
                    }
                )
                
                count += 1
                logger.debug(f"Ingested GitHub issue: {source_id}")

        except GithubException as e:
            logger.error(f"GitHub API error during ingestion: {e}")
            return count

        return count

    async def watch(self, user_id: str, callback: Any) -> None:
        """Future webhook implementation for real-time GitHub sync."""
        raise NotImplementedError("Real-time webhooks for GitHub are not yet implemented.")
