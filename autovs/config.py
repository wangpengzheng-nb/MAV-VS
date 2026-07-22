from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autovs.schemas import ExecutorConfig, ExecutorType


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

    def executor_config(self, name: str) -> ExecutorConfig | None:
        """返回结构化工具执行器配置（v2 注册表）。

        优先读取 [executors.<name>]，回退到旧 [executables] 键值并自动推断 executor 类型。
        """
        raw_exec = self.raw.get("executors", {}).get(name)
        if raw_exec:
            return ExecutorConfig.model_validate(raw_exec)
        # 回退：旧格式 [executables]
        legacy = self.raw.get("executables", {}).get(name, "")
        if not legacy:
            return None
        # 推断 executor 类型
        if name in {"gromacs"} or str(legacy).endswith(".sif"):
            ex_type = ExecutorType.APPTAINER
        elif name in {"admet_ai"}:
            ex_type = ExecutorType.PYTHON_MODULE
        else:
            ex_type = ExecutorType.SUBPROCESS
        return ExecutorConfig(
            name=name, executor=ex_type, path=str(legacy),
            env_hint=self.environment(name) if name in self.raw.get("environments", {}) else "",
        )

    @property
    def executors(self) -> dict[str, ExecutorConfig]:
        """返回所有已注册的工具执行器配置。"""
        result: dict[str, ExecutorConfig] = {}
        # 新格式
        for name in self.raw.get("executors", {}):
            cfg = self.executor_config(name)
            if cfg:
                result[name] = cfg
        # 旧格式补充（未被新格式覆盖的）
        for name in self.raw.get("executables", {}):
            if name not in result:
                cfg = self.executor_config(name)
                if cfg:
                    result[name] = cfg
        return result

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
