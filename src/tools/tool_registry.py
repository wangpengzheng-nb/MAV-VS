"""工具注册表 — 按 action_type 查找和注册工具。"""
from __future__ import annotations
from typing import Dict, List, Optional
from src.tools.tool_interface import BaseTool


class ToolRegistry:
    """管理所有可用工具, 按 action_type 索引。"""

    def __init__(self):
        self._tools: Dict[str, List[BaseTool]] = {}

    def register(self, tool: BaseTool):
        at = tool.spec.action_type
        self._tools.setdefault(at, []).append(tool)

    def register_all(self, tools: List[BaseTool]):
        for t in tools:
            self.register(t)

    def find(self, action_type: str, constraints: Optional[dict] = None) -> Optional[BaseTool]:
        """查找匹配 action_type 的工具。如有多个, 按 constraints 筛选。"""
        candidates = self._tools.get(action_type, [])
        if not candidates:
            return None
        if not constraints:
            return candidates[0]  # 默认返回第一个
        for t in candidates:
            if self._match(t, constraints):
                return t
        return candidates[0]  # fallback

    def list_actions(self) -> List[str]:
        return sorted(self._tools.keys())

    def count(self) -> int:
        return sum(len(v) for v in self._tools.values())

    @staticmethod
    def _match(tool: BaseTool, constraints: dict) -> bool:
        for k, v in constraints.items():
            if getattr(tool.spec, k, None) != v:
                return False
        return True
