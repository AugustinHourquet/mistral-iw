"""Settings, read from the environment and a local `.env` file."""

import os
from dataclasses import dataclass
from pathlib import Path

MAX_LLM_CALLS = 6


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal `.env` file (KEY=VALUE lines, # comments, optional quotes)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"
    traces_dir: Path = Path("traces")
    max_llm_calls: int = MAX_LLM_CALLS
    max_retries: int = 3
    retry_base_delay: float = 1.0

    @classmethod
    def load(cls, env_file: Path = Path(".env")) -> "Settings":
        # Real environment variables win over the .env file.
        env = load_dotenv(env_file) | dict(os.environ)
        return cls(
            mistral_api_key=env.get("MISTRAL_API_KEY", ""),
            mistral_model=env.get("MISTRAL_MODEL") or cls.mistral_model,
            traces_dir=Path(env.get("TRACES_DIR") or cls.traces_dir),
        )
