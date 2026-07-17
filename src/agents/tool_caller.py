"""
工具调用智能体 — 策略(JSON)→工具注册→DAG编排→执行→结果输出。
"""
from __future__ import annotations
import os, json
from typing import Dict, List, Optional
from pathlib import Path

from src.tools.tool_interface import BaseTool
from src.tools.tool_registry import ToolRegistry
from src.tools.data_bus import ResourceContext
from src.tools.orchestrator import DAGWorkflow


class ToolCallerAgent:
    """接收策略和用户上传文件, 编排并执行工作流。"""

    def __init__(self, registry: ToolRegistry = None):
        self.registry = registry or ToolRegistry()

    def run(self, strategy: dict, uploaded_files: Dict[str, str],
            work_dir: str, max_workers: int = 4) -> Dict[str, dict]:
        """执行一个策略。

        Args:
            strategy: 策略JSON (含pipeline)
            uploaded_files: {"target_protein": "/path/to/protein.pdb",
                             "compound_library": "/path/to/lib.smi"}
            work_dir: 工作目录(中间文件和checkpoint存放位置)
            max_workers: 最大并发数

        Returns:
            {step_id: {param: path}} 每个节点的输出
        """
        ctx = ResourceContext(work_dir)
        for name, path in uploaded_files.items():
            ctx.add_file(name, path)

        wf = DAGWorkflow(self.registry, ctx, max_workers=max_workers)
        wf.build_from_strategy(strategy)

        # 注入用户上传文件作为初始输入
        initial = {}
        for name, path in uploaded_files.items():
            key = name.replace("target_", "").replace("compound_", "")
            initial[key] = path
        wf.set_initial_inputs(initial)

        result = wf.execute(resume=True)

        # 清理
        ctx.cleanup()
        return result

    def run_evolved(self, evolved_strategies: list, uploaded_files: dict,
                    work_dir: str) -> dict:
        """执行进化后的Top策略。只跑v2版本。"""
        results = {}
        for s in evolved_strategies:
            if "(v2" not in s.get("strategy_name", ""):
                continue
            name = s["strategy_name"][:40]
            subdir = os.path.join(work_dir, name.replace("/", "_").replace(" ", "_"))
            print(f"\n  🚀 执行: {name}", flush=True)
            results[name] = self.run(s, uploaded_files, subdir)
        return results
