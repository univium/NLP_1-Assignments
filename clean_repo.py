#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def clean_repo():
    """
    Removes common clutter from the repository to make it cleaner for public viewing.
    """
    patterns_to_remove = [
        "__pycache__",
        ".ipynb_checkpoints",
        ".DS_Store",
        "db.sqlite3",
        "db.sqlite3-shm",
        "db.sqlite3-wal",
    ]
    
    repo_root = Path(__file__).parent.resolve()
    removed_count = 0

    print(f"🧹 Starting cleanup in: {repo_root}\n")

    for root, dirs, files in os.walk(repo_root):
        # Clean directories
        for d in list(dirs):
            if d in patterns_to_remove:
                dir_path = Path(root) / d
                print(f"Removing directory: {dir_path.relative_to(repo_root)}")
                shutil.rmtree(dir_path)
                dirs.remove(d)
                removed_count += 1

        # Clean files
        for f in files:
            if any(f.endswith(suffix) for suffix in [".pyc", ".pyo", ".pyd"]) or f in patterns_to_remove:
                file_path = Path(root) / f
                print(f"Removing file: {file_path.relative_to(repo_root)}")
                os.remove(file_path)
                removed_count += 1

    print(f"\n✨ Cleanup complete. Removed {removed_count} items.")

if __name__ == "__main__":
    clean_repo()
