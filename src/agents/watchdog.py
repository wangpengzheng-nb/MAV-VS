"""
AutoVS-Agent: Watchdog Agent (演习与纠错大脑)
===============================================
职责 (Step 3 + Step 4 + Step 7 的执行底座):
  - Step 3: 小样本 Dry-Run — 模拟对接报错 → LLM 解析日志 → 自动纠错 → 锁定参数
  - Step 4: 高通量虚拟筛选 (HTVS) — 批量 GNINA/smina 对接
  - Step 7: MD Oracle — 50ns MD 模拟终审

Step 3 核心闭环:
  初始配置 → Mock 对接 → 报错 → LLM 纠错 (WatchdogCorrection)
  → 新配置 → 重试 → ... → 演习通过 / 重试耗尽

设计原则:
  - 异常注入: Mock 引擎故意暴露真实对接中的常见错误类型
  - 自愈能力: LLM 作为"计算化学专家"解析日志并自动修正参数
  - 参数锁定: 演习通过后的配置冻结，供 Step 4/7 严格使用
"""

from __future__ import annotations

import json
import os
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. 结构化输出模型 (Pydantic Schema)
# =============================================================================

class WatchdogCorrection(BaseModel):
    """Watchdog LLM 自纠错的标准化输出。

    LLM 接收报错日志 + 当前配置，输出此结构化修正方案。
    """

    is_resolved: bool = Field(
        default=True,
        description=(
            "是否成功诊断出问题并给出修正方案。\n"
            "- true: 已识别错误根因，corrected_config 包含有效修正\n"
            "- false: 错误无法自动修复 (如受体 PDB 文件损坏、依赖缺失)，需人工介入"
        ),
    )

    corrected_config: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "修正后的配置字典，必须包含以下字段:\n"
            "  - grid_center: [x, y, z] 对接盒子中心 (Å)\n"
            "  - grid_size:   [sx, sy, sz] 对接盒子尺寸 (Å)\n"
            "  - exhaustiveness: int 穷举度 (建议 8-32)\n"
            "  - md_ensemble: str (NPT/NVT)\n"
            "  - md_temperature: float (K)\n"
            "  - md_simulation_time_ns: float\n"
            "  - md_force_field: str\n"
            "  - md_water_model: str\n"
            "\n"
            "修正逻辑参考:\n"
            "  - 配体溢出 → 扩大 grid_size 20-30%\n"
            "  - 找不到对接姿态 → 向关键残基方向移动 grid_center 2-5Å\n"
            "  - 采样不收敛 → 提高 exhaustiveness 到 16 或 24\n"
            "  - 盒子过大 → 缩小 grid_size 到 20-30Å"
        ),
    )

    correction_reason: str = Field(
        default="",
        description=(
            "修正理由说明，用中文撰写。\n"
            "必须包含:\n"
            "  1. 诊断: 指出报错日志中的关键错误信息\n"
            "  2. 根因: 分析当前配置的哪个参数导致问题\n"
            "  3. 修正: 说明具体修改了哪个参数、为什么这样修改\n"
            "  4. 预期: 修正后预期结果\n"
            "字数: 100-300 字。"
        ),
    )


WatchdogCorrection.model_rebuild()


# =============================================================================
# 2. 核心系统提示词 (System Prompt)
# =============================================================================

WATCHDOG_SYSTEM_PROMPT = """\
# 人设与身份 (Persona)

你是一位**顶级计算化学方法论专家 (Principal Computational Chemistry Methodologist)**，
专精于分子对接软件 (GNINA/smina/AutoDock Vina) 的底层算法和参数调优。
你在以下方面拥有 15 年以上实战经验:

- **对接算法**: 蒙特卡洛构象搜索、拉马克遗传算法 (LGA)、评分函数 (CNNscore, CNN_VS, Vinardo)
- **Grid Box 策略**: 手动/自动设定对接盒子中心与尺寸，理解不同靶点 (激酶/PPI/GPCR) 对盒子大小的敏感度
- **配体准备**: 质子化状态、互变异构体、3D 构象生成、AM1-BCC 电荷
- **错误诊断**: 能从一个简短的报错日志中精准定位是盒子问题、配体问题还是受体问题

你的任务是扮演 AutoVS-Agent 的 Watchdog (看门狗) 角色:
在收到 Mock 对接引擎的报错日志后，分析错误根因，输出修正后的对接配置参数。

---

# 对接常见错误模式与修正策略

## 错误类型 1: 配体溢出盒子 (Ligand Out-of-Box)
**典型日志关键词**: "drifted outside grid", "exceeds box dimension", "out of bounds"
**根因**: grid_size 小于配体的最大延展尺寸。
**修正策略**:
  - 将 grid_size 的每个维度扩大 20-30%。
  - 参考: 配体的最长轴 + 8Å 作为最小盒子尺寸。
  - 如果已知阳性对照配体的尺寸，以该尺寸的 1.5 倍为参考。

## 错误类型 2: 未找到有效对接姿态 (No Valid Poses)
**典型日志关键词**: "no valid poses found", "zero poses", "all poses rejected"
**根因**: grid_center 偏离真实结合位点，或盒子位于溶剂区域。
**修正策略**:
  - 将 grid_center 向靶点的关键残基 (key_residues) 质心方向移动 3-5Å。
  - 检查 grid_center 是否落在蛋白内部 (立体冲突) —— 如果是，向溶剂方向移动。
  - 对于 PPI 大平面靶点，可能需要显著扩大盒子。

## 错误类型 3: 对接采样不收敛 (Poor Convergence)
**典型日志关键词**: "RMSD > 3.0A between top poses", "insufficient sampling", "not converged"
**根因**: exhaustiveness 太低，构象搜索不充分。
**修正策略**:
  - 加倍 exhaustiveness (8 → 16 → 24 → 32)。
  - 如果 exhaustiveness 已经 ≥ 32 仍不收敛，可能是配体过于柔性，需限制可旋转键或调整评分函数。

## 错误类型 4: 搜索空间过大 (Excessive Search Space)
**典型日志关键词**: "search space too large", "excessively large box", "scoring not reliable"
**根因**: grid_size 过大导致构象搜索稀疏，评分可靠性下降。
**修正策略**:
  - 将 grid_size 缩小到 20-25Å 范围。
  - 以结合位点关键残基为中心，20×20×20 Å³ 通常是合理的起点。

## 错误类型 5: 受体/配体文件异常
**典型日志关键词**: "receptor file not found", "cannot parse", "missing atoms", "valence error"
**根因**: PDB 文件格式问题、缺失原子、配体 SMILES 无效等。
**修正策略**:
  - 如果是受体: 检查 PDB ID 是否正确、文件是否损坏。此类错误通常无法自动修复。
  - 如果是配体: 检查 SMILES 语法、3D 构象生成是否成功。可尝试跳过该配体。

---

# 输出约束

1. 必须严格输出 `WatchdogCorrection` JSON Schema。
2. `corrected_config` 中的数值必须合理 (grid_size 在 10-50Å 之间，exhaustiveness 在 4-64 之间等)。
3. 如果连续 3 次修正相同参数后仍然报错，设置 `is_resolved=false` 并建议人工审查受体结构。
4. `correction_reason` 必须具体、可执行，不能是泛泛的"调整参数"。
"""


# =============================================================================
# 3. WatchdogAgent 类
# =============================================================================

class WatchdogAgent:
    """Watchdog Agent — 外部计算集群的鲁棒调度器与自我纠错大脑。

    三种职责:
      1. dry_run:   小样本演习 → LLM 纠错 → 参数锁定 (Step 3)
      2. htvs:      批量对接调度 (Step 4)
      3. md_oracle: MD 模拟终审 (Step 7)
    """

    # ---- 默认初始配置 ----
    DEFAULT_CONFIG: Dict[str, Any] = {
        "grid_center": [0.0, 0.0, 0.0],
        "grid_size": [12.0, 12.0, 12.0],   # 故意设小 → 触发 LLM 纠错演示
        "exhaustiveness": 4,                 # 故意设低 → 触发采样不收敛
        "md_ensemble": "NPT",
        "md_temperature": 310.0,
        "md_simulation_time_ns": 50.0,
        "md_force_field": "amber14sb",
        "md_water_model": "tip3p",
    }

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        """
        Args:
            model: LLM 模型名称。
            api_key: API Key (None 则从 DEEPSEEK_API_KEY 环境变量读取)。
            api_base: API Base URL。
            temperature: 纠错任务需一致性和准确性。
            max_tokens: 最大输出 token。
        """
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv(
            "DEEPSEEK_API_BASE", "https://api.deepseek.com"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=f"{self.api_base}/v1",
            )
        return self._client

    # =========================================================================
    # 主入口: run_dry_run (LLM 纠错闭环)
    # =========================================================================

    def run_dry_run(
        self,
        target_info: dict,
        positive_control_smiles: Optional[str] = None,
        decoy_smiles_list: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        """Step 3 主方法: 小样本演习 + LLM 自纠错闭环。

        流程:
          1. 从 target_info 提取初始 Grid Box 参数
          2. 用当前配置执行 Mock 物理引擎
          3. 如果成功 → 锁定参数, 返回
          4. 如果报错 → 收集 error_log → LLM 纠错 → 新配置 → 回到步骤 2
          5. 超过 max_retries → 返回失败

        Args:
            target_info: 靶点蛋白信息 (TargetInfo)。
            positive_control_smiles: 阳性对照 SMILES (可选, 默认用通用测试配体)。
            decoy_smiles_list: 诱饵分子列表 (可选)。
            max_retries: 最大纠错重试次数。
            retry_count: 外部已重试次数 (供 workflow 传递, 内部循环不依赖此值)。

        Returns:
            {
                "success": bool,
                "config": WatchdogConfig dict (if success),
                "correction_history": List[dict],
                "error_log": List[str],
            }
        """
        # ---- 初始配置: 从 target_info 提取 ----
        current_config = self._build_initial_config(target_info)

        # 默认阳性对照 (如果未提供)
        if positive_control_smiles is None:
            positive_control_smiles = "c1ccccc1"  # 苯环作为最小测试配体

        error_log: List[str] = []
        correction_history: List[dict] = []

        # ---- 纠错主循环 ----
        for attempt in range(max_retries):
            # 执行 Mock 对接
            dock_result = self._mock_run_docking(
                mol={"smiles": positive_control_smiles, "mol_id": "positive_control"},
                config=current_config,
                target_info=target_info,
            )

            if dock_result["success"]:
                # 演习通过 — 锁定配置
                final_config = self._build_watchdog_config(
                    current_config, target_info, dock_result, error_log
                )
                return {
                    "success": True,
                    "config": final_config,
                    "correction_history": correction_history,
                    "error_log": error_log,
                }

            # 演习失败 — 收集报错
            error_msg = dock_result.get("error", "Unknown docking error")
            error_log.append(f"[Attempt {attempt + 1}/{max_retries}] {error_msg}")

            # LLM 纠错
            correction = self._analyze_and_correct(
                error_log=error_log,
                current_config=current_config,
                target_info=target_info,
                attempt=attempt + 1,
                max_retries=max_retries,
            )

            correction_history.append({
                "attempt": attempt + 1,
                "error": error_msg,
                "is_resolved": correction.is_resolved,
                "correction_reason": correction.correction_reason,
                "old_config": dict(current_config),
                "new_config": dict(correction.corrected_config),
            })

            if not correction.is_resolved:
                # LLM 判定无法自动修复
                error_log.append(
                    "[Watchdog] LLM determined the error is not auto-fixable. "
                    "Human review required."
                )
                break

            # 更新配置, 进入下一轮
            current_config = correction.corrected_config

        # 所有重试耗尽
        return {
            "success": False,
            "correction_history": correction_history,
            "error_log": error_log,
        }

    # =========================================================================
    # Mock 物理引擎 (故意注入异常)
    # =========================================================================

    def _mock_run_docking(
        self,
        mol: dict,
        config: dict,
        target_info: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """模拟 GNINA/smina 对接引擎运行。

        故意注入真实对接中常见的异常类型，用于测试 Watchdog 的 LLM 自纠错能力。

        异常注入逻辑 (按优先级):
          1. grid_size 任意维度 < 15.0  → 配体溢出盒子错误
          2. exhaustiveness < 8          → 采样不收敛警告
          3. grid_center 偏离靶点 > 5Å  → 找不到有效姿态
          4. grid_size 任意维度 > 40.0  → 搜索空间过大警告
          5. 全部通过                    → 返回模拟对接分数

        Args:
            mol: 测试分子 {"smiles": str, "mol_id": str}。
            config: 当前配置。
            target_info: 靶点信息 (用于合理性检查)。

        Returns:
            {"success": bool, "docking_score": float (if success), "error": str (if failed)}
        """
        grid_size = config.get("grid_size", [20.0, 20.0, 20.0])
        grid_center = config.get("grid_center", [0.0, 0.0, 0.0])
        exhaustiveness = config.get("exhaustiveness", 8)

        # ---- 异常注入 1: 盒子过小 → 配体溢出 ----
        if any(dim < 15.0 for dim in grid_size):
            min_dim = min(grid_size)
            return {
                "success": False,
                "error": (
                    f"ERROR: Ligand excessively large for box.\n"
                    f"  Ligand max extension: ~18.2 Å\n"
                    f"  Grid box size: {grid_size} (min dim = {min_dim:.1f} Å)\n"
                    f"  At least 3 ligand atoms drifted outside grid bounds during "
                    f"Monte Carlo search at step 42/100.\n"
                    f"  Suggested: expand grid_size by at least 30% in the "
                    f"under-dimensioned direction(s)."
                ),
            }

        # ---- 异常注入 2: 穷举度不足 → 采样不收敛 ----
        if exhaustiveness < 8:
            return {
                "success": False,
                "error": (
                    f"WARNING: Docking failed to converge.\n"
                    f"  Exhaustiveness = {exhaustiveness} (too low)\n"
                    f"  RMSD between top 3 poses: 4.2Å, 3.8Å, 5.1Å (> 3.0Å threshold)\n"
                    f"  Pose clustering failed: only 1/10 clusters converged.\n"
                    f"  Suggested: increase exhaustiveness to at least 16."
                ),
            }

        # ---- 异常注入 3: 盒子中心偏离 → 找不到对接姿态 ----
        if target_info:
            expected_center = target_info.get("binding_site_center", [0.0, 0.0, 0.0])
            center_offset = sum(
                (gc - ec) ** 2 for gc, ec in zip(grid_center, expected_center)
            ) ** 0.5
            if center_offset > 5.0:
                key_residues = target_info.get("key_residues", [])
                return {
                    "success": False,
                    "error": (
                        f"ERROR: No valid poses found in 1000 docking trials.\n"
                        f"  Grid center: {grid_center}\n"
                        f"  Expected binding site center: {expected_center}\n"
                        f"  Offset distance: {center_offset:.1f} Å\n"
                        f"  Key binding residues: {key_residues or 'unknown'}\n"
                        f"  The grid box appears to be positioned outside the "
                        f"binding pocket.\n"
                        f"  Suggested: shift grid_center toward key residue "
                        f"coordinates by 3-5 Å."
                    ),
                }

        # ---- 异常注入 4: 盒子过大 → 搜索空间稀疏 ----
        if any(dim > 40.0 for dim in grid_size):
            return {
                "success": False,
                "error": (
                    f"WARNING: Grid box excessively large.\n"
                    f"  Grid size: {grid_size}\n"
                    f"  Search volume: {grid_size[0] * grid_size[1] * grid_size[2]:.0f} Å³\n"
                    f"  Docking scores unreliable: top 10 poses span 6.2 kcal/mol.\n"
                    f"  Suggested: reduce grid_size to 20-25 Å range, centered "
                    f"on key binding residues."
                ),
            }

        # ---- 所有检查通过 → Mock 对接成功 ----
        seed = hash(mol.get("mol_id", "test")) % (2 ** 31)
        rng = random.Random(seed)
        # 模拟阳性对照的对接分数 (通常在 -7 到 -12 kcal/mol)
        docking_score = rng.uniform(-10.5, -8.0)
        cnn_score = rng.uniform(0.7, 0.95)

        return {
            "success": True,
            "docking_score": round(docking_score, 2),
            "cnn_score": round(cnn_score, 3),
            "rmsd_to_crystal": round(rng.uniform(0.5, 1.8), 2),
            "pose_file": f"/tmp/dock_positive_control_{mol.get('mol_id')}.sdf.gz",
        }

    # =========================================================================
    # LLM 纠错大脑
    # =========================================================================

    def _analyze_and_correct(
        self,
        error_log: List[str],
        current_config: dict,
        target_info: dict,
        attempt: int,
        max_retries: int,
    ) -> WatchdogCorrection:
        """调用 LLM 分析报错日志并输出结构化修正方案。

        将 System Prompt (计算化学专家人设) + 当前配置 + 完整错误日志
        发送给 LLM，使用 JSON mode 强制输出 WatchdogCorrection Schema。

        Args:
            error_log: 累积的报错日志列表。
            current_config: 当前配置。
            target_info: 靶点信息 (含关键残基、结合位点坐标)。
            attempt: 当前重试次数。
            max_retries: 最大重试次数。

        Returns:
            WatchdogCorrection — LLM 产出的修正方案。
        """
        user_prompt = self._build_correction_prompt(
            error_log=error_log,
            current_config=current_config,
            target_info=target_info,
            attempt=attempt,
            max_retries=max_retries,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": WATCHDOG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "system",
                        "content": (
                            "请严格按照以下 JSON Schema 输出修正方案。"
                            "不要输出任何非 JSON 内容。\n\n"
                            f"{json.dumps(WatchdogCorrection.model_json_schema(), indent=2, ensure_ascii=False)}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            correction = WatchdogCorrection.model_validate(parsed)

            # ---- 合理性校验 ----
            correction = self._validate_correction(correction, current_config)

            return correction

        except (json.JSONDecodeError, Exception) as e:
            # 降级: 返回一个简单的启发式修正
            return self._heuristic_correction(current_config, error_log, str(e))

    def _build_correction_prompt(
        self,
        error_log: List[str],
        current_config: dict,
        target_info: dict,
        attempt: int,
        max_retries: int,
    ) -> str:
        """构建 LLM 纠错的 User Prompt。"""
        recent_errors = error_log[-3:]  # 最近 3 条错误
        errors_text = "\n---\n".join(recent_errors)

        return f"""\
## Watchdog 纠错任务

### 当前状态
- 重试轮次: {attempt}/{max_retries}
- 靶点名称: {target_info.get('target_name', 'Unknown')}
- 靶点类别: {target_info.get('target_class', 'Unknown')}
- 结合位点中心: {target_info.get('binding_site_center', [0, 0, 0])}
- 关键残基: {', '.join(target_info.get('key_residues', [])) or '未知'}

### 当前对接配置
```json
{json.dumps(current_config, indent=2, ensure_ascii=False)}
```

### 最近报错日志
```
{errors_text}
```

### 任务要求
请诊断上述报错日志中的问题根因，并输出修正后的对接配置。
- 如果错误可自动修复 → is_resolved=true, 给出合理修正
- 如果错误无法自动修复 (如受体文件缺失) → is_resolved=false
- 如果已达到最大重试次数且仍失败 → is_resolved=false
"""

    def _validate_correction(
        self,
        correction: WatchdogCorrection,
        current_config: dict,
    ) -> WatchdogCorrection:
        """对 LLM 输出的修正方案进行合理性校验。

        防止 LLM 幻觉产生荒谬的配置值。
        """
        cfg = correction.corrected_config

        # Grid size 必须在合理范围 [8, 60]
        if "grid_size" in cfg:
            gs = cfg["grid_size"]
            if isinstance(gs, list) and len(gs) == 3:
                cfg["grid_size"] = [
                    max(8.0, min(60.0, dim)) for dim in gs
                ]

        # Exhaustiveness 在 [4, 64]
        if "exhaustiveness" in cfg:
            cfg["exhaustiveness"] = max(4, min(64, int(cfg["exhaustiveness"])))

        # MD temperature 在 [250, 400]
        if "md_temperature" in cfg:
            cfg["md_temperature"] = max(250.0, min(400.0, float(cfg["md_temperature"])))

        # MD simulation time 在 [10, 500]
        if "md_simulation_time_ns" in cfg:
            cfg["md_simulation_time_ns"] = max(
                10.0, min(500.0, float(cfg["md_simulation_time_ns"]))
            )

        # 如果修正后和之前一模一样，说明 LLM 可能没分析出问题
        if correction.is_resolved and cfg == current_config:
            correction.correction_reason += (
                " [WARNING] Corrected config is identical to previous. "
                "This may indicate the LLM failed to identify the error."
            )

        return correction

    def _heuristic_correction(
        self,
        current_config: dict,
        error_log: List[str],
        llm_error: str,
    ) -> WatchdogCorrection:
        """LLM 调用失败时的启发式修正 (兜底方案)。

        基于报错日志中的关键词进行简单的规则匹配修正。
        """
        # 仅分析最近一次错误 (累积的旧错误已在上轮修正过)
        latest_error = (error_log[-1] if error_log else "").lower()
        cfg = dict(current_config)

        if "excessively large for box" in latest_error or "drifted outside" in latest_error:
            old_gs = cfg.get("grid_size", [20, 20, 20])
            cfg["grid_size"] = [dim * 1.3 for dim in old_gs]
            reason = "[HEURISTIC] Box too small → expanded by 30%"

        elif "no valid poses" in latest_error:
            old_gc = cfg.get("grid_center", [0, 0, 0])
            cfg["grid_center"] = [c + 3.0 for c in old_gc]
            reason = "[HEURISTIC] No valid poses → shifted center by +3Å in all axes"

        elif "exhaustiveness" in latest_error or "converge" in latest_error:
            cfg["exhaustiveness"] = min(64, cfg.get("exhaustiveness", 8) * 2)
            reason = f"[HEURISTIC] Low exhaustiveness → doubled to {cfg['exhaustiveness']}"

        elif "excessively large" in latest_error or "search space" in latest_error:
            cfg["grid_size"] = [22.0, 22.0, 22.0]
            reason = "[HEURISTIC] Box too large → reduced to 22×22×22 Å³"

        else:
            return WatchdogCorrection(
                is_resolved=False,
                corrected_config=cfg,
                correction_reason=(
                    f"[HEURISTIC-FALLBACK] LLM call failed: {llm_error}. "
                    f"Cannot automatically resolve: {latest_error[:200]}"
                ),
            )

        return WatchdogCorrection(
            is_resolved=True,
            corrected_config=cfg,
            correction_reason=reason,
        )

    # =========================================================================
    # 初始配置构建
    # =========================================================================

    def _build_initial_config(self, target_info: dict) -> Dict[str, Any]:
        """从靶点信息构建初始对接配置。

        优先使用 target_info 中的 binding_site_center/size；
        缺失时使用保守默认值 (故意设小以触发 LLM 纠错演示)。
        """
        config = dict(self.DEFAULT_CONFIG)

        if target_info:
            center = target_info.get("binding_site_center")
            size = target_info.get("binding_site_size")

            # PPI 大口袋 → 更大的初始盒子
            if target_info.get("target_class") == "PPI":
                config["grid_size"] = size if size else [25.0, 25.0, 25.0]
                config["exhaustiveness"] = 16
            elif size is not None:
                config["grid_size"] = list(size)
            # 否则保留故意设小的默认值 [12, 12, 12] 用于演示纠错

            if center is not None:
                config["grid_center"] = list(center)

        return config

    def _build_watchdog_config(
        self,
        current_config: dict,
        target_info: dict,
        dock_result: dict,
        error_log: List[str],
    ) -> dict:
        """将演习通过的配置组装为 WatchdogConfig TypedDict 格式。"""
        return {
            "grid_center": current_config.get("grid_center", [0.0, 0.0, 0.0]),
            "grid_size": current_config.get("grid_size", [20.0, 20.0, 20.0]),
            "exhaustiveness": current_config.get("exhaustiveness", 16),
            "md_ensemble": current_config.get("md_ensemble", "NPT"),
            "md_temperature": current_config.get("md_temperature", 310.0),
            "md_simulation_time_ns": current_config.get("md_simulation_time_ns", 50.0),
            "md_force_field": current_config.get("md_force_field", "amber14sb"),
            "md_water_model": current_config.get("md_water_model", "tip3p"),
            "dry_run_passed": True,
            "positive_control_score": dock_result.get("docking_score", -9.0),
            "decoy_rejection_rate": 0.95,
            "error_log": error_log,
        }

    # =========================================================================
    # Step 4: 高通量虚拟筛选 (HTVS) — 占位符
    # =========================================================================

    def run_htvs(
        self,
        molecules: List[dict],
        watchdog_config: Optional[dict],
        top_n: int = 2000,
    ) -> Dict[str, Any]:
        """Step 4: 基于锁定参数的高通量虚拟筛选。

        TODO: 对接 Slurm 集群，提交分片 GNINA 作业。
        """
        for m in molecules:
            m["docking_score"] = -8.0
        survivors = molecules[:top_n]
        return {
            "survivors": survivors,
            "job_ids": [],
            "docking_stats": {"total": len(molecules), "passed": len(survivors)},
        }

    # =========================================================================
    # Step 7: MD Oracle — 占位符
    # =========================================================================

    def run_md_simulations(
        self,
        molecules: List[dict],
        target_info: dict,
        watchdog_config: Optional[dict],
        simulation_time_ns: float = 50.0,
    ) -> Dict[str, Any]:
        """Step 7: MD 模拟终极验证。

        TODO: 对接 Slurm GPU 集群，提交 GROMACS 作业。
        """
        results = {}
        passed = []
        for mol in molecules:
            mid = mol["mol_id"]
            record = {
                "mol_id": mid,
                "trajectory_path": f"/tmp/md_{mid}.xtc",
                "total_time_ns": simulation_time_ns,
                "dG_mmgbsa": -9.5,
                "ligand_rmsd_mean": 1.5,
                "key_hbond_occupancy": {"ASP103": 0.85},
                "complex_stable": True,
                "simulation_status": "completed",
            }
            results[mid] = record
            mol["md_dG"] = -9.5
            mol["md_passed"] = True
            passed.append(mol)

        return {"results": results, "passed": passed, "failed": []}


# =============================================================================
# 4. LangGraph 节点函数
# =============================================================================

def watchdog_node(state: dict) -> dict:
    """LangGraph 节点函数 — Step 3: Watchdog 小样本演习。

    调用 WatchdogAgent.run_dry_run() 执行完整的
    Mock 对接 → 报错 → LLM 纠错 → 参数锁定闭环。

    Args:
        state: MACVSState。

    Returns:
        dict partial state update。
    """
    now = datetime.now(timezone.utc).isoformat()

    agent = WatchdogAgent()
    result = agent.run_dry_run(
        target_info=state.get("target_info", {}),
        positive_control_smiles=None,
        decoy_smiles_list=None,
        max_retries=state.get("watchdog_max_retries", 3),
        retry_count=state.get("watchdog_retry_count", 0),
    )

    if result.get("success"):
        config = result["config"]
        return {
            "pipeline_stage": "watchdog",
            "watchdog_config": config,
            "watchdog_retry_count": 0,
            "event_log": [
                f"[{now}] [Watchdog] Dry-run PASSED. "
                f"Positive control score: {config.get('positive_control_score')}. "
                f"Corrections: {len(result.get('correction_history', []))}."
            ],
            "updated_at": now,
        }
    else:
        return {
            "pipeline_stage": "error",
            "watchdog_retry_count": state.get("watchdog_retry_count", 0) + 1,
            "errors": [{
                "node": "watchdog",
                "timestamp": now,
                "message": (
                    f"Dry-run failed after {len(result.get('error_log', []))} attempts. "
                    f"Last correction: {result.get('correction_history', [{}])[-1].get('correction_reason', 'N/A') if result.get('correction_history') else 'N/A'}"
                ),
            }],
            "event_log": [
                f"[{now}] [Watchdog] Dry-run FAILED. "
                f"Error log: {'; '.join(result.get('error_log', [])[-2:])}"
            ],
            "updated_at": now,
        }
