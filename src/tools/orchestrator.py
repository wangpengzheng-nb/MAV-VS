"""DAG 工作流编排器 — 策略→DAG→执行→Checkpoint。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.tools.data_bus import ResourceContext, DataBus, ResourceMonitor
from src.tools.tool_interface import BaseTool, ToolSpec
from src.tools.tool_registry import ToolRegistry


@dataclass
class DAGNode:
    step_id: str
    action_type: str
    tool: Optional[BaseTool] = None
    params: dict = field(default_factory=dict)
    inputs: Dict[str, str] = field(default_factory=dict)
    deps: List[str] = field(default_factory=list)  # 依赖的 step_id


class DAGWorkflow:
    """解析策略 pipeline → 构建 DAG → 执行。"""

    def __init__(self, registry: ToolRegistry, ctx: ResourceContext, max_workers: int = 4):
        self.registry = registry
        self.ctx = ctx
        self.bus = DataBus(ctx)
        self.nodes: Dict[str, DAGNode] = {}
        self.max_workers = max_workers
        self.node_outputs: Dict[str, dict] = {}  # step_id → {param: path}

    # ═══════════════════════════════════════════
    # 构建
    # ═══════════════════════════════════════════

    def build_from_strategy(self, strategy: dict):
        """从策略 JSON 的 pipeline 构建 DAG。"""
        pipeline = strategy.get("pipeline", strategy.get("pipeline_steps", []))
        prev_outputs: Optional[dict] = None
        prev_step_id: Optional[str] = None

        for step in pipeline:
            sid = step.get("step_id", f"step_{step.get('step_number','?')}")
            at = step.get("action_type", "?")
            tool = self.registry.find(at)
            if not tool:
                print(f"  ⚠️ 未找到 tool: {at}, 跳过节点 {sid}", flush=True)
                continue

            # 绑定输入: 如果上一步有输出, 自动连接到本步输入
            node_inputs = {}
            if prev_outputs and tool.spec.inputs:
                node_inputs = _auto_bind(prev_outputs, tool.spec, self.bus)

            deps = [prev_step_id] if prev_step_id else []
            self.nodes[sid] = DAGNode(
                step_id=sid, action_type=at, tool=tool,
                params=step.get("parameters", {}),
                inputs=node_inputs, deps=deps,
            )
            prev_step_id = sid
            prev_outputs = None  # 运行时才填充

    def set_initial_inputs(self, inputs: Dict[str, str]):
        """设置第一个节点的输入(来自用户上传文件)。"""
        first_sid = next(iter(self.nodes), None)
        if first_sid:
            self.nodes[first_sid].inputs = inputs

    # ═══════════════════════════════════════════
    # 执行
    # ═══════════════════════════════════════════

    def execute(self, resume: bool = True) -> Dict[str, dict]:
        """拓扑执行所有节点, 支持 checkpoint 恢复。"""
        monitor = ResourceMonitor(str(self.ctx.work_dir))
        monitor.start()

        try:
            for sid in _topo_order(self.nodes):
                node = self.nodes[sid]

                # Checkpoint: 跳过已完成且参数未变的步骤
                if resume and self.bus.should_skip(sid, node.params):
                    ckpt = self.bus.checkpoint_load(sid)
                    self.node_outputs[sid] = {f["key"]: f["path"] for f in ckpt["output_files"]}
                    # 传递输出给下一个节点
                    _pass_to_next(self.nodes, sid, self.node_outputs[sid])
                    print(f"  ⏩ [{sid}] checkpoint命中, 跳过", flush=True)
                    continue

                print(f"  🔧 [{sid}] {node.action_type} ...", flush=True)
                try:
                    result = node.tool.execute(node.inputs, self.ctx)
                except Exception as e:
                    print(f"  ❌ [{sid}] 执行失败: {e}", flush=True)
                    raise

                self.node_outputs[sid] = result
                # 保存 checkpoint
                self.bus.checkpoint_save(sid, node.params, result,
                                         monitor.last_snapshot())
                # 传递给下一个节点
                _pass_to_next(self.nodes, sid, result)
                print(f"  ✅ [{sid}] 完成 → {list(result.keys())}", flush=True)
        finally:
            monitor.stop()

        return self.node_outputs


# ═══════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════

def _auto_bind(prev_outputs: dict, next_spec: ToolSpec, bus: DataBus) -> dict:
    """自动将上游输出绑定到下游输入 (4级匹配)。"""
    bound = {}
    for in_key, in_fmt in next_spec.inputs.items():
        aliases = next_spec.input_aliases.get(in_key, []) + [in_key]
        matched = None
        for out_key, out_path in prev_outputs.items():
            if out_key in aliases:  # 规则1+2: 精确+别名匹配
                matched = out_path
                break
        if not matched and len(prev_outputs) == 1 and len(next_spec.inputs) == 1:
            # 规则3: 唯一输入输出, 直接连接
            matched = list(prev_outputs.values())[0]
        if matched:
            out_fmt = _guess_format(matched, prev_outputs)
            if out_fmt and in_fmt and out_fmt.upper() != in_fmt.upper():
                try:
                    matched = bus.convert(Path(matched), out_fmt, in_fmt)
                except ValueError:
                    pass  # 转换失败, 直接用原文件
            bound[in_key] = str(matched)
    return bound


def _pass_to_next(nodes: dict, current_sid: str, outputs: dict):
    """将当前节点的输出作为下一个节点的输入。"""
    next_sids = [sid for sid, n in nodes.items() if current_sid in n.deps]
    for nsid in next_sids:
        if not nodes[nsid].inputs:
            nodes[nsid].inputs = dict(outputs)


def _topo_order(nodes: dict) -> List[str]:
    """拓扑排序。线性 pipeline 直接按插入顺序即可。"""
    return list(nodes.keys())


def _guess_format(path: str, outputs: dict) -> str:
    """从文件后缀猜测格式。"""
    ext = Path(path).suffix.lower().lstrip(".")
    format_map = {"sdf": "SDF", "pdb": "PDB", "pdbqt": "PDBQT",
                  "smi": "SMILES", "csv": "CSV", "txt": "TXT"}
    return format_map.get(ext, ext.upper())
