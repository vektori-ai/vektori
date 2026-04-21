import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

from vektori.client import Vektori
from vektori.connectors.github import GitHubConnector
from vektori.connectors.auth import AuthStore

async def main():
    # Provide a test user ID
    user_id = "test_user_001"
    
    # Check if GitHub token is provided via environment
    gh_token = os.environ.get("GITHUB_TOKEN")
    if not gh_token:
        print("Please set the GITHUB_TOKEN environment variable.")
        print("Example: export GITHUB_TOKEN=ghp_your_personal_access_token")
        sys.exit(1)

    # 1. Store the token for this user
    print("Saving token to local AuthStore...")
    auth = AuthStore()
    auth.set_token(user_id=user_id, platform="github", access_token=gh_token)

    # 2. Instantiate Vektori memory engine
    print("Initializing Vektori...")
    # Using sqlite memory for demo
    v = Vektori(storage_backend="sqlite")

    # 3. Create the GitHub Connector for a specific repo
    # Using a fast/small repo as an example. Change to any repo you have access to.
    repo_name = "tiangolo/fastapi"  # Public repo Example, but you can use your own.
    print(f"Initializing GitHub connector for repo: {repo_name}...")
    github_conn = GitHubConnector(repo_name=repo_name, auth_store=auth)

    # 4. Ingest data! We use `since` to only get recent issues for a quick test
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    print("Pulling and ingesting issues...")
    try:
        ingested_count = await v.connect(
            connector=github_conn, 
            user_id=user_id,
            since=thirty_days_ago
        )
        print(f"Successfully ingested {ingested_count} GitHub issues/PRs into memory!")
        
        # 5. Let's test the memory by searching it!
        print("\nSearching Vektori memory for facts about these issues...")
        results = await v.search(
            query="What are the recent bug fixes or feature requests?", 
            user_id=user_id
        )
        
        print("\nExtracted Facts:")
        for fact in results.get("facts", []):
            print(f"- {fact['content']}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await v.close()

if __name__ == "__main__":
    asyncio.run(main())
