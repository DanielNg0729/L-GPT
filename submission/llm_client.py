"""Small shared Groq configuration used by the optional recovery helpers."""
from __future__ import annotations

import os
from pathlib import Path


ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def load_project_env() -> None:
    """Read a local ignored .env file without replacing real environment variables."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass
