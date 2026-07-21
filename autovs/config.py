from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    config_path: Path

    @property
    def database_path(self) -> Path:
        return _resolve_project_path(os.getenv("AUTOVS_DATABASE", self.raw["service"]["database"]))

    @property
    def task_root(self) -> Path:
        return _resolve_project_path(os.getenv("AUTOVS_TASK_ROOT", self.raw["service"]["task_root"]))

    @property
    def host(self) -> str:
        return os.getenv("AUTOVS_MCP_HOST", self.raw["service"]["host"])

    @property
    def port(self) -> int:
        return int(os.getenv("AUTOVS_MCP_PORT", self.raw["service"]["port"]))

    def executable(self, name: str) -> Path | None:
        value = os.getenv(f"AUTOVS_{name.upper()}_BIN", self.raw.get("executables", {}).get(name, ""))
        return _resolve_project_path(value) if value else None

    def environment(self, name: str) -> str:
        return str(self.raw.get("environments", {}).get(name, ""))

    def container(self, name: str) -> Path | None:
        value = self.raw.get("containers", {}).get(name, "")
        return _resolve_project_path(value) if value else None

    def limit(self, name: str, default: Any = None) -> Any:
        return self.raw.get("limits", {}).get(name, default)

    def library(self, name: str = "default") -> dict[str, Any]:
        return dict(self.raw.get("libraries", {}).get(name, {}))

    @property
    def default_library_path(self) -> Path:
        value = os.getenv("AUTOVS_DEFAULT_LIBRARY", self.library().get("path", ""))
        if not value:
            raise ValueError("default molecular library is not configured")
        return _resolve_project_path(value)


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("AUTOVS_CONFIG", PROJECT_ROOT / "config/tools.toml")).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    settings = Settings(raw=raw, config_path=config_path)
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    settings.task_root.mkdir(parents=True, exist_ok=True)
    return settings
