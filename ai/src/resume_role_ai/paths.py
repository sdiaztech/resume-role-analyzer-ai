from __future__ import annotations

from pathlib import Path


def find_repository_root(start: str | Path | None = None) -> Path:
    """Find the project checkout from a caller-provided path or the current directory."""
    current = Path(start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        has_project_marker = (candidate / "ai" / "pyproject.toml").is_file() or (
            candidate / "web" / "index.html"
        ).is_file()
        if (candidate / "datasets").is_dir() and has_project_marker:
            return candidate
    raise FileNotFoundError(
        "Could not find the repository root; run the command from inside the project"
    )
