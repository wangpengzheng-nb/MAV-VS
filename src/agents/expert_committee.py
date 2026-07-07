"""
AutoVS-Agent: MedChem Committee (多专家委员会) — 三级联漏斗
=============================================================
职责 (Step 5: 动态靶向过滤):
  对 HTVS 粗筛后的数千分子执行分层级联过滤，逐级淘汰不达标分子。

三级联漏斗 (Tiered Cascade Funnel):
  Tier 0: 动态对接阈值截断 — 阳性对照导向的动态阈值，硬性上限 5000
  Tier 1: 2D 物理化学一票否决 — RDKit 硬计算 MW/LogP/TPSA/... + SMARTS 黑名单
  Tier 2: 3D 与 AI 多维深度图谱 — PLIP/ML 重打分/ADMET AI (占位符)

设计原则:
  - 鲁棒性第一: 无效 SMILES 绝不崩溃，捕获并记入 vetoed
  - 可追溯: 每个被淘汰分子都有具体的 veto_reason
  - 算力分层: 便宜的计算先跑 (Tier 0/1)，昂贵的 AI 只跑幸存者 (Tier 2)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from src.graph.state import (
    MoleculeRecord,
    MoleculeID,
    DynamicFilterProtocol,
    WatchdogConfig,
)


# ---------------------------------------------------------------------------
# RDKit 延迟导入 — 生产环境已安装，开发/测试环境优雅降级
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit import RDLogger

    # 禁用 RDKit 的冗长警告 (如无效 SMILES 的 WARNING 日志)
    RDLogger.logger().setLevel(RDLogger.ERROR)
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


# =============================================================================
# 常量
# =============================================================================

# Tier 0 硬性上限: 防止进入 RDKit 层的分子数过多导致内存爆炸
MAX_TIER0_SURVIVORS = 5000

# Tier 2 硬性上限: 重型工具 (PLIP/ML) 只在最多这么多幸存者上运行
MAX_TIER2_PROFILING = 500

# Tier 0 的对接分数缓冲区: 阳性对照分数 + BUFFER 以内的分子都保留
POSITIVE_CONTROL_BUFFER = 1.0  # kcal/mol


# =============================================================================
# MedChem Committee — 主类
# =============================================================================

class MedChemCommittee:
    """MedChem Committee — 三级联漏斗过滤引擎。

    取代了简单的 Top-N 截断和固定的单一工具调用，
    采用算力分层的级联策略逐步淘汰不达标分子。
    """

    def __init__(
        self,
        members: Optional[List[Dict]] = None,
        veto_mode: str = "hard",
        veto_threshold: float = 0.5,
    ):
        """
        Args:
            members: 专家委员会成员配置 (保留接口兼容性)。
            veto_mode: "hard" (一级否决则淘汰) 或 "soft"。
            veto_threshold: soft 模式下的加权阈值。
        """
        self.members = members or []
        self.veto_mode = veto_mode
        self.veto_threshold = veto_threshold

    # =========================================================================
    # 主入口: evaluate_batch
    # =========================================================================

    def evaluate_batch(
        self,
        molecules: List[MoleculeRecord],
        filter_protocol: Optional[DynamicFilterProtocol] = None,
        watchdog_config: Optional[WatchdogConfig] = None,
    ) -> Dict[str, Any]:
        """Step 5 主入口: 执行三级联漏斗过滤。

        Args:
            molecules: HTVS 对接后的存活分子列表。
            filter_protocol: Strategy Agent 产出的动态过滤规则。
            watchdog_config: Watchdog 锁定的参数 (含 positive_control_score)。

        Returns:
            {
                "passed": List[MoleculeRecord],
                "vetoed": List[MoleculeRecord],
                "veto_reasons": Dict[MoleculeID, List[str]],
                "reports": List[dict],            # 各 Tier 统计报告
                "tier_stats": Dict[str, int],     # 各 Tier 存活/淘汰计数
            }
        """
        # ---- 防御: 空输入 ----
        if not molecules:
            return {
                "passed": [],
                "vetoed": [],
                "veto_reasons": {},
                "reports": [],
                "tier_stats": {"tier0_in": 0, "tier0_out": 0, "tier0_vetoed": 0,
                               "tier1_in": 0, "tier1_out": 0, "tier1_vetoed": 0,
                               "tier2_in": 0, "tier2_out": 0},
            }

        # 全局否决原因追踪
        veto_reasons: Dict[MoleculeID, List[str]] = {}
        tier_stats: Dict[str, int] = {}

        # =====================================================================
        # Tier 0: 动态对接阈值截断
        # =====================================================================
        tier0_in = len(molecules)
        passed_t0, vetoed_t0 = self._tier0_dynamic_docking_filter(
            molecules, filter_protocol, watchdog_config, veto_reasons,
        )
        tier_stats["tier0_in"] = tier0_in
        tier_stats["tier0_out"] = len(passed_t0)
        tier_stats["tier0_vetoed"] = len(vetoed_t0)

        # =====================================================================
        # Tier 1: 2D 物理化学一票否决 (RDKit)
        # =====================================================================
        tier1_in = len(passed_t0)
        if passed_t0 and _RDKIT_AVAILABLE:
            passed_t1, vetoed_t1 = self._tier1_rdkit_physchem_filter(
                passed_t0, filter_protocol, veto_reasons,
            )
        elif passed_t0:
            # RDKit 不可用 → 跳过 Tier 1，所有分子原样通过
            passed_t1 = passed_t0
            vetoed_t1 = []
        else:
            passed_t1, vetoed_t1 = [], []

        tier_stats["tier1_in"] = tier1_in
        tier_stats["tier1_out"] = len(passed_t1)
        tier_stats["tier1_vetoed"] = len(vetoed_t1)

        # =====================================================================
        # Tier 2: 3D 与 AI 多维深度图谱 (占位符)
        # =====================================================================
        tier2_in = len(passed_t1)
        if passed_t1:
            passed_t2 = self._tier2_deep_profiling(
                passed_t1[:MAX_TIER2_PROFILING], filter_protocol,
            )
        else:
            passed_t2 = []

        tier_stats["tier2_in"] = tier2_in
        tier_stats["tier2_out"] = len(passed_t2)

        # ---- 汇总 ----
        all_vetoed = vetoed_t0 + vetoed_t1

        # 构建轻量级统计报告
        reports = [{
            "tier": "tier0_docking",
            "input_count": tier0_in,
            "survived": len(passed_t0),
            "vetoed": len(vetoed_t0),
            "threshold_used": getattr(self, "_last_tier0_threshold", None),
        }, {
            "tier": "tier1_physchem",
            "input_count": tier1_in,
            "survived": len(passed_t1),
            "vetoed": len(vetoed_t1),
            "rdkit_available": _RDKIT_AVAILABLE,
        }, {
            "tier": "tier2_deep_profiling",
            "input_count": tier2_in,
            "survived": len(passed_t2),
            "profiled": min(tier2_in, MAX_TIER2_PROFILING),
        }]

        return {
            "passed": passed_t2,
            "vetoed": all_vetoed,
            "veto_reasons": veto_reasons,
            "reports": reports,
            "tier_stats": tier_stats,
        }

    # =========================================================================
    # Tier 0: 动态对接阈值截断
    # =========================================================================

    def _tier0_dynamic_docking_filter(
        self,
        molecules: List[MoleculeRecord],
        filter_protocol: Optional[DynamicFilterProtocol],
        watchdog_config: Optional[WatchdogConfig],
        veto_reasons: Dict[MoleculeID, List[str]],
    ) -> Tuple[List[MoleculeRecord], List[MoleculeRecord]]:
        """Tier 0: 基于阳性对照的动态对接阈值截断。

        策略:
          threshold = min(positive_control_score + BUFFER, docking_score_min)
          只保留对接分数 <= threshold 的分子 (越负越好)。
          最后按对接分数升序排序，硬截断至 MAX_TIER0_SURVIVORS。

        Args:
            molecules: 输入分子列表。
            filter_protocol: 动态过滤协议。
            watchdog_config: 含 positive_control_score。
            veto_reasons: 否决原因字典 (in-place 修改)。

        Returns:
            (passed, vetoed)
        """
        # ---- 解析阳性对照分数 ----
        positive_score: Optional[float] = None
        if watchdog_config:
            positive_score = watchdog_config.get("positive_control_score")
            # 如果未设置则尝试 None
            if positive_score == 0.0:
                positive_score = None

        # ---- 解析协议最低分数 ----
        protocol_min: Optional[float] = None
        if filter_protocol:
            protocol_min = filter_protocol.get("docking_score_min")

        # ---- 计算动态阈值 ----
        threshold = self._compute_dynamic_threshold(positive_score, protocol_min)
        self._last_tier0_threshold = threshold

        # ---- 分离 passed / vetoed ----
        passed: List[MoleculeRecord] = []
        vetoed: List[MoleculeRecord] = []

        for mol in molecules:
            score = mol.get("docking_score")
            mol_id = mol.get("mol_id", "unknown")

            # 缺分数的分子: 保守处理 → veto
            if score is None:
                vetoed.append(mol)
                veto_reasons.setdefault(mol_id, []).append(
                    "Tier0: missing docking_score"
                )
                continue

            # 对接分数 <= 阈值 → 通过 (越负越优)
            if threshold is None or score <= threshold:
                passed.append(mol)
            else:
                vetoed.append(mol)
                veto_reasons.setdefault(mol_id, []).append(
                    f"Tier0: docking_score={score:.2f} > threshold={threshold:.2f}"
                )

        # ---- 硬性上限截断: 按分数升序 (越负越靠前) 保留前 N ----
        if len(passed) > MAX_TIER0_SURVIVORS:
            # 排序: 越负越靠前
            passed.sort(key=lambda m: m.get("docking_score") or 0.0)
            overflow = passed[MAX_TIER0_SURVIVORS:]
            passed = passed[:MAX_TIER0_SURVIVORS]
            for mol in overflow:
                mol_id = mol.get("mol_id", "unknown")
                vetoed.append(mol)
                veto_reasons.setdefault(mol_id, []).append(
                    f"Tier0: exceeded hard cap of {MAX_TIER0_SURVIVORS}"
                )

        return passed, vetoed

    @staticmethod
    def _compute_dynamic_threshold(
        positive_control_score: Optional[float],
        docking_score_min: Optional[float],
    ) -> Optional[float]:
        """计算动态对接阈值。

        逻辑:
          - 如果有阳性对照分数: threshold = min(positive + 1.0, docking_min)
            取更严格的那个 (更负的那个)
          - 如果只有协议阈值: threshold = docking_min
          - 如果两者都无: threshold = None (不过滤)

        对接分数约定: 越负代表结合越强 (如 smina/GNINA affinity)。
        threshold 本身也是负值，min() 返回更负的那个 (更严格)。

        Args:
            positive_control_score: 阳性对照分子对接分数 (如 -9.5 kcal/mol)。
            docking_score_min: 协议设定的最低门槛 (如 -7.0 kcal/mol)。

        Returns:
            动态阈值 (float) 或 None。
        """
        candidates: List[float] = []

        if positive_control_score is not None:
            candidates.append(positive_control_score + POSITIVE_CONTROL_BUFFER)

        if docking_score_min is not None:
            candidates.append(docking_score_min)

        if not candidates:
            return None

        # min() 返回最负/最严格的那个
        return min(candidates)

    # =========================================================================
    # Tier 1: 2D 物理化学一票否决 (RDKit)
    # =========================================================================

    def _tier1_rdkit_physchem_filter(
        self,
        molecules: List[MoleculeRecord],
        filter_protocol: Optional[DynamicFilterProtocol],
        veto_reasons: Dict[MoleculeID, List[str]],
    ) -> Tuple[List[MoleculeRecord], List[MoleculeRecord]]:
        """Tier 1: RDKit 2D 物理化学硬计算 + SMARTS 黑名单扫描。

        对每个分子执行:
          1. SMILES → RDKit Mol (失败则直接 veto)
          2. 计算理化性质: MW, LogP, TPSA, HBD, HBA, RotBonds
          3. 与 filter_protocol 阈值逐一比对，不达标则 veto
          4. SMARTS 子结构匹配: excluded_substructures + toxic_groups
          5. 更新分子的理化性质字段 (供 Step 6 锦标赛引用)

        Args:
            molecules: Tier 0 幸存分子。
            filter_protocol: 动态过滤协议。
            veto_reasons: 否决原因字典。

        Returns:
            (passed, vetoed)
        """
        if not _RDKIT_AVAILABLE:
            return molecules, []

        protocol = filter_protocol or {}

        # 从协议中提取阈值; 如果协议缺少某字段则使用宽松默认值 (不做该维度过滤)
        mw_range = protocol.get("mw_range", [0, 9999])
        logp_range = protocol.get("logp_range", [-20, 20])
        tpsa_range = protocol.get("tpsa_range", [0, 9999])
        hbd_max = protocol.get("hbd_max", 999)
        hba_max = protocol.get("hba_max", 999)
        rot_bonds_max = protocol.get("rotatable_bonds_max", 999)
        excluded_smarts = protocol.get("excluded_substructures", [])
        toxic_smarts = protocol.get("toxic_groups", [])

        # 预编译 SMARTS (有错误的 SMARTS 跳过不崩溃)
        excluded_patterns = self._compile_smarts_list(excluded_smarts)
        toxic_patterns = self._compile_smarts_list(toxic_smarts)

        passed: List[MoleculeRecord] = []
        vetoed: List[MoleculeRecord] = []

        for mol in molecules:
            mol_id = mol.get("mol_id", "unknown")
            smiles = mol.get("smiles", "")
            reasons: List[str] = []

            # ---- 1. SMILES → Mol ----
            rdmol = self._safe_mol_from_smiles(smiles)
            if rdmol is None:
                vetoed.append(mol)
                veto_reasons.setdefault(mol_id, []).append(
                    "Tier1: Invalid SMILES — RDKit cannot parse"
                )
                continue

            # ---- 2. 理化性质计算 ----
            try:
                mw = Descriptors.MolWt(rdmol)
                logp = Descriptors.MolLogP(rdmol)
                tpsa = rdMolDescriptors.CalcTPSA(rdmol)
                hbd = rdMolDescriptors.CalcNumHBD(rdmol)
                hba = rdMolDescriptors.CalcNumHBA(rdmol)
                rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(rdmol)
            except Exception:
                vetoed.append(mol)
                veto_reasons.setdefault(mol_id, []).append(
                    "Tier1: RDKit descriptor calculation failed"
                )
                continue

            # ---- 3. 阈值比对 ----
            if not (mw_range[0] <= mw <= mw_range[1]):
                reasons.append(
                    f"Tier1: MW={mw:.1f} out of range [{mw_range[0]}, {mw_range[1]}]"
                )
            if not (logp_range[0] <= logp <= logp_range[1]):
                reasons.append(
                    f"Tier1: LogP={logp:.2f} out of range [{logp_range[0]}, {logp_range[1]}]"
                )
            if not (tpsa_range[0] <= tpsa <= tpsa_range[1]):
                reasons.append(
                    f"Tier1: TPSA={tpsa:.1f} out of range [{tpsa_range[0]}, {tpsa_range[1]}]"
                )
            if hbd > hbd_max:
                reasons.append(f"Tier1: HBD={hbd} > max={hbd_max}")
            if hba > hba_max:
                reasons.append(f"Tier1: HBA={hba} > max={hba_max}")
            if rot_bonds > rot_bonds_max:
                reasons.append(f"Tier1: RotBonds={rot_bonds} > max={rot_bonds_max}")

            # ---- 4. SMARTS 黑名单扫描 ----
            smarts_hits = self._check_smarts_patterns(rdmol, excluded_patterns, "excluded")
            smarts_hits += self._check_smarts_patterns(rdmol, toxic_patterns, "toxic")
            reasons.extend(smarts_hits)

            # ---- 5. 更新分子属性 ----
            # 无论 pass 还是 veto，都填充计算值 (供后续调试/审计)
            mol["admet_flags"] = mol.get("admet_flags") or {}
            if isinstance(mol["admet_flags"], dict):
                mol["admet_flags"]["MW"] = mw
                mol["admet_flags"]["LogP"] = logp
                mol["admet_flags"]["TPSA"] = tpsa
                mol["admet_flags"]["HBD"] = hbd
                mol["admet_flags"]["HBA"] = hba
                mol["admet_flags"]["RotBonds"] = rot_bonds
                mol["admet_flags"]["PAINS_alert"] = len(smarts_hits) > 0

            # ---- 6. 裁决 ----
            if reasons:
                vetoed.append(mol)
                veto_reasons.setdefault(mol_id, []).extend(reasons)
                mol["medchem_passed"] = False
            else:
                passed.append(mol)
                mol["medchem_passed"] = True

        return passed, vetoed

    # =========================================================================
    # Tier 2: 3D 与 AI 多维深度图谱 (占位符)
    # =========================================================================

    def _tier2_deep_profiling(
        self,
        molecules: List[MoleculeRecord],
        filter_protocol: Optional[DynamicFilterProtocol],
    ) -> List[MoleculeRecord]:
        """Tier 2: 3D 与 AI 深度分析 — 为锦标赛准备多维数据。

        对 Tier 1 幸存者 (≤500) 调用重型工具:
          1. PLIP 分析 → 氢键数 + structural_score
          2. ML 重打分 → mlp_pred_dG
          3. ADMET AI → 肝毒性标志 + 额外 ADMET 属性

        注意: Tier 2 不执行淘汰，只填充数据供 Step 6 Ranking 使用。
        返回所有输入分子 (in-place 更新)。

        Args:
            molecules: Tier 1 幸存分子。
            filter_protocol: 动态过滤协议。

        Returns:
            List[MoleculeRecord] — 所有分子 (in-place 更新了深度属性)。
        """
        for mol in molecules:
            # ---- 2a. PLIP 分析 (Mock) ----
            plip_result = self._run_plip_analysis(mol)
            mol["structural_score"] = plip_result["structural_score"]
            if isinstance(mol.get("admet_flags"), dict):
                mol["admet_flags"]["PLIP_hbond_count"] = plip_result["hbond_count"]
                mol["admet_flags"]["PLIP_hydrophobic_count"] = plip_result["hydrophobic_count"]

            # ---- 2b. ML 重打分 (Mock) ----
            ml_result = self._run_ml_rescoring(mol)
            mol["mlp_pred_dG"] = ml_result["mlp_pred_dG"]
            mol["mlp_uncertainty"] = ml_result["mlp_uncertainty"]

            # ---- 2c. ADMET AI (Mock) ----
            admet_ai = self._run_admet_ai(mol)
            if isinstance(mol.get("admet_flags"), dict):
                mol["admet_flags"]["hepatotoxicity_risk"] = admet_ai["hepatotoxicity_risk"]
                mol["admet_flags"]["hERG_blocker_risk"] = admet_ai["hERG_blocker_risk"]
                mol["admet_flags"]["BBB_penetration"] = admet_ai["BBB_penetration"]
                mol["admet_flags"]["CYP2D6_inhibitor"] = admet_ai["CYP2D6_inhibitor"]

        return molecules

    # =========================================================================
    # Mock 函数: PLIP 分析
    # =========================================================================

    @staticmethod
    def _run_plip_analysis(mol: MoleculeRecord) -> Dict[str, Any]:
        """Mock: 模拟 PLIP (Protein-Ligand Interaction Profiler) 分析。

        在生产环境中，此函数将:
          1. 调用 PLIP 命令行或 Python API
          2. 解析 report.xml
          3. 提取氢键/疏水/盐桥/π-堆积的数量和距离
          4. 基于关键残基匹配度计算 structural_score

        Returns:
            {
                "hbond_count": int,         # 氢键数量
                "hydrophobic_count": int,   # 疏水接触数量
                "salt_bridge_count": int,   # 盐桥数量
                "pi_stack_count": int,      # π-π 堆积数量
                "key_residue_hbond_count": int,  # 与关键残基的氢键数
                "structural_score": float,  # 结构互补性评分 (0-1)
            }
        """
        # 用 mol_id 的哈希做确定性种子，保证同一分子每次结果一致
        seed = hash(mol.get("mol_id", "")) % (2 ** 31)
        rng = random.Random(seed)

        hbond_count = rng.randint(0, 6)
        hydrophobic_count = rng.randint(2, 12)
        salt_bridge_count = rng.randint(0, 2)
        pi_stack_count = rng.randint(0, 3)
        key_residue_hbond = rng.randint(0, min(hbond_count, 3))

        # structural_score: 基于相互作用丰度计算
        structural_score = min(1.0, (
            hbond_count * 0.15 +
            hydrophobic_count * 0.05 +
            salt_bridge_count * 0.10 +
            pi_stack_count * 0.12 +
            key_residue_hbond * 0.20
        ))

        return {
            "hbond_count": hbond_count,
            "hydrophobic_count": hydrophobic_count,
            "salt_bridge_count": salt_bridge_count,
            "pi_stack_count": pi_stack_count,
            "key_residue_hbond_count": key_residue_hbond,
            "structural_score": round(structural_score, 3),
        }

    # =========================================================================
    # Mock 函数: ML 重打分
    # =========================================================================

    @staticmethod
    def _run_ml_rescoring(mol: MoleculeRecord) -> Dict[str, Any]:
        """Mock: 模拟 ML 代理模型的重打分预测。

        在生产环境中，此函数将:
          1. 加载训练好的 ProxyMLP 模型权重
          2. 从 mol["smiles"] 计算 ECFP4 指纹
          3. 前向传播得到 ΔG 预测值
          4. MC Dropout (30 次) 计算不确定性

        Returns:
            {
                "mlp_pred_dG": float,       # 预测结合自由能 (kcal/mol)
                "mlp_uncertainty": float,   # 预测不确定性 (0-1)
            }
        """
        seed = hash(mol.get("mol_id", "") + "_mlp") % (2 ** 31)
        rng = random.Random(seed)

        # 以 docking_score 为基准，加入随机扰动模拟 ML 修正
        dock_score = mol.get("docking_score") or -7.0
        mlp_pred_dG = dock_score + rng.uniform(-1.5, 1.5)
        mlp_uncertainty = rng.uniform(0.05, 0.40)

        return {
            "mlp_pred_dG": round(mlp_pred_dG, 3),
            "mlp_uncertainty": round(mlp_uncertainty, 3),
        }

    # =========================================================================
    # Mock 函数: ADMET AI
    # =========================================================================

    @staticmethod
    def _run_admet_ai(mol: MoleculeRecord) -> Dict[str, Any]:
        """Mock: 模拟 AI 驱动的 ADMET 深度预测。

        在生产环境中，此函数将:
          1. 调用 ADMETlab 3.0 API 或本地模型
          2. 预测肝毒性 (hepatotoxicity)
          3. 预测 hERG 抑制
          4. 预测血脑屏障穿透
          5. 预测 CYP450 酶抑制谱

        Returns:
            {
                "hepatotoxicity_risk": str,   # "low" / "medium" / "high"
                "hERG_blocker_risk": str,
                "BBB_penetration": str,       # "yes" / "no" / "borderline"
                "CYP2D6_inhibitor": str,
                "CYP3A4_inhibitor": str,
            }
        """
        seed = hash(mol.get("mol_id", "") + "_admet") % (2 ** 31)
        rng = random.Random(seed)

        return {
            "hepatotoxicity_risk": rng.choice(["low", "low", "low", "medium", "high"]),
            "hERG_blocker_risk": rng.choice(["low", "low", "medium", "high"]),
            "BBB_penetration": rng.choice(["no", "no", "borderline", "yes"]),
            "CYP2D6_inhibitor": rng.choice(["no", "no", "no", "yes"]),
            "CYP3A4_inhibitor": rng.choice(["no", "no", "yes", "yes"]),
        }

    # =========================================================================
    # 辅助方法
    # =========================================================================

    @staticmethod
    def _safe_mol_from_smiles(smiles: str) -> Optional[Any]:
        """将 SMILES 字符串安全转换为 RDKit Mol 对象。

        Args:
            smiles: SMILES 字符串。

        Returns:
            rdkit.Chem.rdchem.Mol 或 None (解析失败)。
        """
        if not smiles or not isinstance(smiles, str):
            return None
        if not _RDKIT_AVAILABLE:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None
            # 尝试获取基本属性来验证分子的有效性
            _ = mol.GetNumAtoms()
            return mol
        except Exception:
            return None

    @staticmethod
    def _compile_smarts_list(smarts_list: List[str]) -> List[Tuple[str, Any]]:
        """编译 SMARTS 字符串列表为 RDKit 模式对象列表。

        对每条 SMARTS: 编译失败时跳过并继续 (不崩溃)，
        保证黑名单的整体鲁棒性。

        Args:
            smarts_list: SMARTS 字符串列表。

        Returns:
            [(原始SMARTS字符串, RDKit Mol对象), ...]
            编译失败的项目被跳过。
        """
        if not _RDKIT_AVAILABLE:
            return []
        patterns = []
        for smarts in smarts_list:
            if not smarts or not isinstance(smarts, str):
                continue
            try:
                pat = Chem.MolFromSmarts(smarts.strip())
                if pat is not None:
                    patterns.append((smarts, pat))
            except Exception:
                # 无效 SMARTS: 跳过，不阻塞整个管道
                pass
        return patterns

    @staticmethod
    def _check_smarts_patterns(
        rdmol: Any,
        patterns: List[Tuple[str, Any]],
        category: str,
    ) -> List[str]:
        """检查分子是否匹配任意 SMARTS 模式。

        Args:
            rdmol: RDKit Mol 对象。
            patterns: 预编译的 (SMARTS, Mol) 列表。
            category: 模式类别标签 (如 "excluded" / "toxic")。

        Returns:
            匹配命中的否决原因列表。
        """
        hits = []
        for smarts_str, pattern in patterns:
            try:
                if rdmol.HasSubstructMatch(pattern):
                    hits.append(f"Tier1: SMARTS match [{category}]: {smarts_str}")
            except Exception:
                # 单条 SMARTS 匹配异常不中断其他检测
                pass
        return hits


# =============================================================================
# LangGraph 节点函数
# =============================================================================

def medchem_filter_node(state: dict) -> dict:
    """LangGraph 节点函数 — Step 5: MedChem Committee 绝对值淘汰。

    从 MACVSState 中提取 surviving_pool、filter_protocol、watchdog_config，
    执行三级联漏斗过滤，返回 partial state update。

    调用方式:
      workflow.add_node("medchem_filter", medchem_filter_node)

    Args:
        state: MACVSState (LangGraph 状态字典)。

    Returns:
        dict partial state update。
    """
    from datetime import datetime, timezone

    molecules = state.get("surviving_pool", [])
    filter_protocol = state.get("filter_protocol")
    watchdog_config = state.get("watchdog_config")

    committee = MedChemCommittee()
    result = committee.evaluate_batch(
        molecules=molecules,
        filter_protocol=filter_protocol,
        watchdog_config=watchdog_config,
    )

    survivors = result.get("passed", [])
    vetoed = result.get("vetoed", [])
    veto_reasons = result.get("veto_reasons", {})
    tier_stats = result.get("tier_stats", {})
    reports = result.get("reports", [])

    now = datetime.now(timezone.utc).isoformat()

    # 存活分子过少 → 回退 Step 1 放宽策略
    if len(survivors) < 10:
        return {
            "pipeline_stage": "strategy",
            "event_log": [
                f"[{now}] [MedChem] Only {len(survivors)} survived "
                f"(Tier0: {tier_stats.get('tier0_out', '?')} → "
                f"Tier1: {tier_stats.get('tier1_out', '?')} → "
                f"Tier2: {tier_stats.get('tier2_out', '?')}). "
                f"Threshold too strict. Relaxing protocol."
            ],
            "errors": [{
                "node": "medchem_filter",
                "timestamp": now,
                "message": (
                    f"Insufficient survivors ({len(survivors)}). "
                    f"Veto reasons distribution: {_summarize_veto_reasons(veto_reasons)}"
                ),
            }],
        }

    return {
        "pipeline_stage": "medchem_filter",
        "surviving_pool": survivors,
        "committee_reports": reports,
        "screened_records": {
            **state.get("screened_records", {}),
            **{m["mol_id"]: m for m in vetoed},
        },
        "event_log": [
            f"[{now}] [MedChem] Cascade complete: "
            f"Tier0 {tier_stats.get('tier0_in',0)}→{tier_stats.get('tier0_out',0)}, "
            f"Tier1 {tier_stats.get('tier1_in',0)}→{tier_stats.get('tier1_out',0)} "
            f"({tier_stats.get('tier1_vetoed',0)} vetoed), "
            f"Tier2 profiled {tier_stats.get('tier2_out',0)}. "
            f"Final survivors: {len(survivors)}."
        ],
        "updated_at": now,
    }


def _summarize_veto_reasons(veto_reasons: Dict[MoleculeID, List[str]]) -> str:
    """汇总否决原因统计 (用于日志)。"""
    from collections import Counter

    all_reasons = []
    for reasons in veto_reasons.values():
        # 只取原因的关键词 (去掉具体的数值)
        for r in reasons:
            short = r.split(":")[0] if ":" in r else r
            all_reasons.append(short)

    counter = Counter(all_reasons)
    return ", ".join(f"{k}={v}" for k, v in counter.most_common(8))
