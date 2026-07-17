"""标准工具接口 — 统一封装所有底层工具。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ToolSpec:
    """工具规格说明 — Orchestrator 据此做绑定和拆分决策。"""
    name: str = ""
    action_type: str = ""
    description: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)   # {"library": "sdf", "protein": "pdb"}
    outputs: Dict[str, str] = field(default_factory=dict)  # {"docked": "sdf", "scores": "csv"}
    batching_strategy: str = "none"  # "none" | "file_split" | "smiles_split" | "custom"
    batch_size: int = 1000           # file_split/smiles_split 时每批数量
    input_aliases: Dict[str, List[str]] = field(default_factory=dict)
    # 例: {"library": ["docked_ligands","compounds"], "protein": ["target_pdb","receptor"]}


class BaseTool(ABC):
    """所有工具的基类。"""

    spec: ToolSpec

    @abstractmethod
    def execute(self, inputs: Dict[str, str], ctx) -> Dict[str, str]:
        """统一入口: {param: path} → {param: path}。ctx is ResourceContext."""
        ...

    def supports_batching(self) -> bool:
        return self.spec.batching_strategy in ("file_split", "smiles_split")

    def __repr__(self):
        return f"<{self.spec.name} [{self.spec.action_type}]>"
