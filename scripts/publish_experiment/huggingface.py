from pathlib import Path

from huggingface_hub import HfApi


def upload_space(
    local_dir: Path,
    repo_id: str,
    private: bool,
) -> None:

    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=private,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message=(
            f"Publish {local_dir.name} experiment"
        ),
    )

    print()
    print("Published:")
    print(
        f"https://huggingface.co/spaces/{repo_id}"
    )