"""工具部署管理 — 根据 ToolSpec.tier 选择执行方式。"""
from __future__ import annotations
import subprocess as sp
from typing import Dict
from src.tools.tool_interface import ToolSpec


class CondaExecutor:
    """在 conda 环境中执行 Tier 1 工具 (Python/轻量级二进制)。"""

    def __init__(self, env_name: str = "base_screening"):
        self.env = env_name

    def run(self, spec: ToolSpec, command: list, inputs: Dict[str, str],
            ctx) -> Dict[str, str]:
        """在 conda 环境内执行命令。"""
        cmd = ["conda", "run", "-n", self.env, "--no-capture-output"] + command
        r = sp.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"[{spec.name}] conda执行失败: {r.stderr[:500]}")
        # 子类需覆盖此方法以返回实际的输出文件映射
        return {}

    def run_python(self, spec: ToolSpec, func, inputs: Dict[str, str],
                   ctx) -> Dict[str, str]:
        """在 conda 环境中执行 Python 函数。"""
        # 直接在当前进程调用 (假设已有该 conda 环境的 Python)
        return func(inputs, ctx)


class ApptainerExecutor:
    """在 Apptainer 容器中执行 Tier 2/3 工具。"""

    def __init__(self, bind_paths: list = None):
        self.bind_paths = bind_paths or ["/tmp", "/users_home"]

    def run(self, spec: ToolSpec, command: list, inputs: Dict[str, str],
            ctx) -> Dict[str, str]:
        """启动容器并执行命令。"""
        if not spec.image:
            raise ValueError(f"[{spec.name}] 缺少 apptainer 镜像路径")
        binds = " ".join(f"--bind {p}:{p}" for p in self.bind_paths)
        gpu_flag = "--nv" if spec.gpu_required else ""
        cmd = f"apptainer exec {gpu_flag} {binds} {spec.image} {' '.join(command)}"
        r = sp.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"[{spec.name}] apptainer执行失败: {r.stderr[:500]}")
        return {}


class ToolDeploymentManager:
    """根据 ToolSpec.tier 自动选择执行器。"""

    def __init__(self):
        self.conda = CondaExecutor(env_name="base_screening")
        self.apptainer = ApptainerExecutor()

    def execute(self, spec: ToolSpec, command: list, inputs: Dict[str, str],
                ctx) -> Dict[str, str]:
        if spec.tier == "conda":
            return self.conda.run(spec, command, inputs, ctx)
        elif spec.tier == "apptainer":
            return self.apptainer.run(spec, command, inputs, ctx)
        else:
            raise ValueError(f"未知 tier: {spec.tier}")
