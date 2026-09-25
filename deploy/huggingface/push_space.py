"""Upload the assembled demo folder to a Hugging Face Space (creating it if needed).

HF_TOKEN=hf_xxx [HF_SPACE=user/escalator-monitor] python deploy/huggingface/push_space.py space/
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi


def main(folder: str) -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    space = os.environ.get("HF_SPACE") or f"{api.whoami()['name']}/escalator-monitor"
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    revision = os.environ.get("GITHUB_SHA", "local")[:7]
    api.upload_folder(
        folder_path=folder,
        repo_id=space,
        repo_type="space",
        commit_message=f"Deploy {revision}",
        delete_patterns=["*.py", "escalator_monitor/**", "examples/**"],  # drop files removed upstream
    )
    print(f"Deployed to https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "space")
