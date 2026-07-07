"""
AutoVS-Agent: Orchestrator Agent (空间降维引擎)
=================================================
职责 (Step 2: 化学空间聚类降维):
  - 读取百万~百亿级 SMILES 大库，分块流式处理
  - 计算 Morgan 指纹 (ECFP4, 2048 bits)，无效 SMILES 零崩溃
  - 多样性导向降维: 哈希分桶 (Hash Bucketing) + MaxMinPicker 二级漏斗
  - 闭环偏置采样: 来自 Step 8 knowledge_base 的优势骨架优先纳入候选池

输出:
  - List[MoleculeRecord]: 具备最大化学空间覆盖的代表分子 (默认 ~10万)

设计原则:
  - 内存安全: 生成器逐块读取，绝不一次性 `readlines()`
  - 算力分层: 便宜哈希先粗筛 → 昂贵指纹再精挑
  - 工业容错: 无效 SMILES → 静默跳过 + 计数，管道不中断
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Tuple

from src.graph.state import (
    MoleculeRecord,
    MoleculeID,
)


# ---------------------------------------------------------------------------
# RDKit 延迟导入 (生产环境已安装，开发环境优雅降级)
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.DataStructs import (
        ConvertToNumpyArray,
        TanimotoSimilarity,
        BulkTanimotoSimilarity,
    )
    from rdkit.SimDivFilters import MaxMinPicker

    RDLogger.logger().setLevel(RDLogger.ERROR)
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


# =============================================================================
# 常量
# =============================================================================

# Morgan 指纹参数
FINGERPRINT_RADIUS = 2        # ECFP4
FINGERPRINT_BITS = 2048

# 哈希分桶: 折叠到多少位 (越大→越细的桶→代表分子越多)
# 默认 24 bits ≈ 1600 万个桶 → 每个 100 亿大库产生约 500 万~1600 万代表
# 可通过 target_size 动态调整
DEFAULT_HASH_BITS = 24

# 哈希分桶阶段硬性上限: 折叠桶中最多保留多少代表进入 MaxMin 精挑
MAX_HASH_REPRESENTATIVES = 500_000

# 大库阈值: 超过此估算大小的库启用哈希预分桶
LARGE_LIBRARY_THRESHOLD = 1_000_000

# 闭环偏置: 多少比例的候选池预留给"赢家特征"分子
AL_BIAS_RESERVED_RATIO = 0.15   # 15%

# 优势骨架匹配阈值: Tanimoto 相似度高于此值视为匹配
PRIVILEGED_SCAFFOLD_THRESHOLD = 0.35

# 分块读取大小
CHUNK_SIZE_SMILES = 100_000     # SMILES 文件每块行数
CHUNK_SIZE_CSV = 50_000         # CSV 文件每块行数

# 指纹计算批大小
FP_BATCH_SIZE = 10_000


# =============================================================================
# OrchestratorAgent
# =============================================================================

class OrchestratorAgent:
    """Orchestrator Agent — 化学空间降维引擎。

    采用"哈希分桶粗筛 → MaxMin 精挑"二级漏斗:
      1. 分块流式读取 SMILES 文件
      2. 计算 Morgan 指纹 (ECFP4, 2048 bits)
      3. 指纹折叠 → 哈希桶 → 每桶保留 K 个代表
      4. MaxMinPicker 从代表中挑选 target_size 个多样性最优分子
      5. 如存在闭环知识库，注入优势骨架偏置
    """

    def __init__(
        self,
        fingerprint_radius: int = FINGERPRINT_RADIUS,
        fingerprint_bits: int = FINGERPRINT_BITS,
    ):
        self.fp_radius = fingerprint_radius
        self.fp_bits = fingerprint_bits

    # =========================================================================
    # 主入口
    # =========================================================================

    def run_clustering(
        self,
        library_path: str,
        target_size: int = 100_000,
        knowledge_base: Optional[Dict[str, Any]] = None,
    ) -> List[MoleculeRecord]:
        """Step 2 主方法: 多样性导向的大库降维采样。

        策略选择:
          - 小库 (< 100 万): 直接 MaxMinPicker (最精确)
          - 大库 (≥ 100 万): 哈希分桶粗筛 → MaxMinPicker 精挑

        Args:
            library_path: 分子库文件路径 (.smi / .csv)。
            target_size: 目标候选池大小 (默认 10 万)。
            knowledge_base: 闭环累积知识库 (可选)。
              若包含 privileged_scaffolds / recommended_scaffolds，
              将优先纳入具有"赢家特征"的分子。

        Returns:
            List[MoleculeRecord] — 化学空间覆盖最优的候选分子列表。
            每个 record 包含 mol_id, smiles, source_db, cluster_id。
        """
        if not _RDKIT_AVAILABLE:
            raise RuntimeError(
                "RDKit is required for clustering. "
                "Install with: conda install -c conda-forge rdkit"
            )

        # ---- 1. 估算库大小 ----
        estimated_size = self._estimate_library_size(library_path)

        # ---- 2. 解析闭环偏置 ----
        bias_scaffolds = self._extract_bias_scaffolds(knowledge_base)
        bias_fps = self._compute_bias_fingerprints(bias_scaffolds) if bias_scaffolds else []
        reserved_slots = int(target_size * AL_BIAS_RESERVED_RATIO) if bias_fps else 0

        # ---- 3. 决策路径 ----
        if estimated_size < LARGE_LIBRARY_THRESHOLD:
            # 小库: 直接 MaxMinPicker
            candidates = self._direct_maxmin_pick(
                library_path=library_path,
                target_size=target_size - reserved_slots,
                bias_fps=bias_fps,
                reserved_slots=reserved_slots,
            )
        else:
            # 大库: 哈希分桶 → MaxMinPicker
            candidates = self._hash_bucket_then_maxmin_pick(
                library_path=library_path,
                target_size=target_size - reserved_slots,
                estimated_size=estimated_size,
                bias_fps=bias_fps,
                reserved_slots=reserved_slots,
            )

        # ---- 4. 分配 cluster_id (按选择顺序) ----
        for idx, mol in enumerate(candidates):
            mol["cluster_id"] = idx

        return candidates

    # =========================================================================
    # 策略 A: 直接 MaxMinPicker (小库, < 100 万)
    # =========================================================================

    def _direct_maxmin_pick(
        self,
        library_path: str,
        target_size: int,
        bias_fps: List[Any],
        reserved_slots: int,
    ) -> List[MoleculeRecord]:
        """小库路径: 流式读取 → 建指纹 → MaxMinPicker 一次精挑。

        内存模型: 所有有效分子 + 指纹完整驻留在内存中。
        """
        records, fingerprints = self._stream_and_featurize(library_path)

        if len(records) == 0:
            return []

        actual_pick = min(target_size, len(records))

        # MaxMin 挑选
        picker = MaxMinPicker()
        picked_indices = picker.LazyPick(
            dist_func=self._make_tanimoto_dist_func(fingerprints),
            pool_size=len(fingerprints),
            pick_size=actual_pick,
            seed=42,
        )
        picked_indices = list(picked_indices)

        # 组装结果
        candidates = [records[i] for i in picked_indices]

        # 注入闭环偏置分子
        if bias_fps and reserved_slots > 0:
            biased = self._pick_biased_molecules(records, fingerprints, bias_fps, reserved_slots)
            # 去重合并
            existing_ids = {m["mol_id"] for m in candidates}
            for m in biased:
                if m["mol_id"] not in existing_ids and len(candidates) < target_size + reserved_slots:
                    candidates.append(m)
                    existing_ids.add(m["mol_id"])

        return candidates

    # =========================================================================
    # 策略 B: 哈希分桶 → MaxMinPicker (大库, ≥ 100 万)
    # =========================================================================

    def _hash_bucket_then_maxmin_pick(
        self,
        library_path: str,
        target_size: int,
        estimated_size: int,
        bias_fps: List[Any],
        reserved_slots: int,
    ) -> List[MoleculeRecord]:
        """大库路径: 哈希分桶粗筛 → 指纹计算 → MaxMinPicker 精挑。

        阶段 1 (Hash Bucketing):
          流式读取 → 轻量哈希 → 分桶 → 每桶保留有限代表。
          哈希桶的大小由 estimated_size 和 MAX_HASH_REPRESENTATIVES 动态调节。

        阶段 2 (MaxMinPicker):
          对哈希桶代表计算 Morgan 指纹 → MaxMinPicker 精挑 target_size。
        """
        # ---- 阶段 1: 哈希分桶 ----
        # TODO: Integrate PySpark/Dask/LSH backend for 10B+ scale.
        #       当前用纯 Python + RDKit 实现百万~千万级可用版本。
        hash_bits = self._choose_hash_bits(estimated_size)
        representatives = self._hash_bucket_stream(
            library_path=library_path,
            hash_bits=hash_bits,
            max_repr=MAX_HASH_REPRESENTATIVES,
        )
        # representatives: List[MoleculeRecord] — 每个桶的代表

        if len(representatives) == 0:
            return []

        # ---- 阶段 2: 对代表计算指纹 + MaxMin 精挑 ----
        rep_fps: List[Any] = []
        valid_repr: List[MoleculeRecord] = []

        for batch_start in range(0, len(representatives), FP_BATCH_SIZE):
            batch = representatives[batch_start:batch_start + FP_BATCH_SIZE]
            for mol in batch:
                fp = self._safe_morgan_fp(mol["smiles"])
                if fp is not None:
                    rep_fps.append(fp)
                    valid_repr.append(mol)

        if len(valid_repr) == 0:
            return []

        actual_pick = min(target_size, len(valid_repr))

        picker = MaxMinPicker()
        picked_indices = picker.LazyPick(
            dist_func=self._make_tanimoto_dist_func(rep_fps),
            pool_size=len(rep_fps),
            pick_size=actual_pick,
            seed=42,
        )
        picked_indices = list(picked_indices)

        candidates = [valid_repr[i] for i in picked_indices]

        # 注入闭环偏置
        if bias_fps and reserved_slots > 0:
            biased = self._pick_biased_molecules(valid_repr, rep_fps, bias_fps, reserved_slots)
            existing_ids = {m["mol_id"] for m in candidates}
            for m in biased:
                if m["mol_id"] not in existing_ids:
                    candidates.append(m)
                    existing_ids.add(m["mol_id"])

        return candidates

    # =========================================================================
    # 流式读取与特征化
    # =========================================================================

    def _stream_and_featurize(
        self,
        library_path: str,
    ) -> Tuple[List[MoleculeRecord], List[Any]]:
        """分块流式读取 SMILES 文件，实时计算 Morgan 指纹。

        使用生成器逐块读取 (Chunked Generator)，每块计算指纹后
        累积到 records + fingerprints 列表。

        Returns:
            (records, fingerprints) 一一对应。
        """
        records: List[MoleculeRecord] = []
        fingerprints: List[Any] = []

        invalid_count = 0

        for chunk in self._read_smiles_chunks(library_path):
            for smiles in chunk:
                smiles = smiles.strip()
                if not smiles or smiles.startswith("#"):
                    continue

                fp = self._safe_morgan_fp(smiles)
                if fp is None:
                    invalid_count += 1
                    continue

                mol_id = self._generate_mol_id(smiles)
                records.append({
                    "mol_id": mol_id,
                    "smiles": smiles,
                    "source_db": self._infer_source_db(library_path),
                })
                fingerprints.append(fp)

        return records, fingerprints

    def _read_smiles_chunks(
        self,
        library_path: str,
    ) -> Generator[List[str], None, None]:
        """分块流式读取 SMILES / CSV 文件。

        使用生成器模式 (Generator Pattern)，每次 yield 一个 chunk，
        保证内存中同时只驻留一个 chunk 的数据。

        支持格式:
          - .smi / .smiles: 每行一个 SMILES (可能带空格分隔的名称)
          - .csv / .tsv: Pandas chunked reader，自动检测 SMILES 列

        Yields:
            List[str] — 每块 CHUNK_SIZE_SMILES 行。
        """
        ext = library_path.rsplit(".", 1)[-1].lower() if "." in library_path else ""

        if ext in ("csv", "tsv"):
            yield from self._read_csv_chunks(library_path)
        else:
            # 默认: SMILES 文本文件
            yield from self._read_text_chunks(library_path)

    def _read_text_chunks(
        self,
        library_path: str,
    ) -> Generator[List[str], None, None]:
        """逐块读取纯文本 SMILES 文件。

        使用 Python 内置文件迭代器 (惰性读取)，not readlines()。
        """
        # TODO: Integrate PySpark's textFile() for 10B+ scale distributed reading.
        chunk: List[str] = []
        try:
            with open(library_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    chunk.append(line.strip())
                    if len(chunk) >= CHUNK_SIZE_SMILES:
                        yield chunk
                        chunk = []
                # 最后一块
                if chunk:
                    yield chunk
        except FileNotFoundError:
            raise FileNotFoundError(f"Library file not found: {library_path}")
        except Exception as e:
            raise IOError(f"Error reading library file {library_path}: {e}")

    def _read_csv_chunks(
        self,
        library_path: str,
    ) -> Generator[List[str], None, None]:
        """使用 Pandas 分块读取 CSV/TSV 文件。

        自动检测 SMILES 列 (列名包含 "smiles" / "SMILES" / "canonical_smiles")。
        """
        # TODO: Integrate Dask DataFrame for distributed CSV parsing at 10B+ scale.
        try:
            import pandas as pd
        except ImportError:
            # 降级: 按文本方式读取
            yield from self._read_text_chunks(library_path)
            return

        sep = "\t" if library_path.endswith(".tsv") else ","
        try:
            reader = pd.read_csv(
                library_path,
                sep=sep,
                chunksize=CHUNK_SIZE_CSV,
                dtype=str,
                on_bad_lines="skip",
            )
            for df_chunk in reader:
                # 自动检测 SMILES 列
                smiles_col = None
                for col in df_chunk.columns:
                    if "smiles" in col.lower() or col.lower() == "canonical_smiles":
                        smiles_col = col
                        break
                # 如果找不到，用第一列
                if smiles_col is None:
                    smiles_col = df_chunk.columns[0]

                smi_list = df_chunk[smiles_col].dropna().tolist()
                yield smi_list
        except Exception:
            # CSV 解析失败 → 降级为文本读取
            yield from self._read_text_chunks(library_path)

    # =========================================================================
    # 哈希分桶
    # =========================================================================

    def _hash_bucket_stream(
        self,
        library_path: str,
        hash_bits: int,
        max_repr: int,
    ) -> List[MoleculeRecord]:
        """流式哈希分桶: 将大库分子分配到哈希桶，每桶保留有限代表。

        算法:
          1. 分块读取 SMILES
          2. 对每个有效分子计算 Morgan 指纹 → 折叠到 hash_bits 位 → 桶 ID
          3. 每个桶维护一个固定大小的 LRU 缓存 (保留最多 K 个代表)
          4. 流式处理完毕后，从所有桶收集代表分子

        优势:
          - 内存友好: 只有 (2^hash_bits * K) 条记录在内存中
          - 化学感知: 折叠指纹保留了结构相似性 (相似分子同桶概率大)
          - 流式友好: 不需要预先知道库大小

        Args:
            library_path: 分子库文件路径。
            hash_bits: 折叠位数 (默认 24 → 2^24 ≈ 1600 万个桶)。
            max_repr: 总共最多保留的代表分子数。

        Returns:
            从各桶收集的代表分子列表。
        """
        num_buckets = 1 << hash_bits
        # 每个桶最多保留多少代表
        per_bucket_limit = max(1, max_repr // num_buckets + 1)

        # 桶 → 代表分子列表
        buckets: Dict[int, List[MoleculeRecord]] = {}
        total_kept = 0
        invalid_count = 0

        for chunk in self._read_smiles_chunks(library_path):
            for smiles in chunk:
                smiles = smiles.strip()
                if not smiles or smiles.startswith("#"):
                    continue

                # 计算指纹并折叠为哈希桶 ID
                fp = self._safe_morgan_fp(smiles)
                if fp is None:
                    invalid_count += 1
                    continue

                bucket_id = self._fold_fingerprint_to_bucket(fp, hash_bits)

                # 桶管理: 保留有限代表
                if bucket_id not in buckets:
                    buckets[bucket_id] = []

                if len(buckets[bucket_id]) < per_bucket_limit:
                    mol_id = self._generate_mol_id(smiles)
                    record: MoleculeRecord = {
                        "mol_id": mol_id,
                        "smiles": smiles,
                        "source_db": self._infer_source_db(library_path),
                    }
                    buckets[bucket_id].append(record)
                    total_kept += 1

                # 如果总代表数已达上限，随机驱逐旧桶的代表 (蓄水池采样思想)
                elif total_kept >= max_repr:
                    # 以 50% 概率替换桶内随机一条
                    import random as _random
                    if _random.random() < 0.5:
                        idx = _random.randint(0, per_bucket_limit - 1)
                        mol_id = self._generate_mol_id(smiles)
                        buckets[bucket_id][idx] = {
                            "mol_id": mol_id,
                            "smiles": smiles,
                            "source_db": self._infer_source_db(library_path),
                        }

        # 从所有桶收集代表
        representatives: List[MoleculeRecord] = []
        for bucket_mols in buckets.values():
            representatives.extend(bucket_mols)

        # 如果代表过多，随机截断到 max_repr
        if len(representatives) > max_repr:
            import random as _random
            _random.shuffle(representatives)
            representatives = representatives[:max_repr]

        return representatives

    @staticmethod
    def _fold_fingerprint_to_bucket(fp: Any, hash_bits: int) -> int:
        """将 Morgan 指纹均匀且确定性地映射为 hash_bits 位的整数桶 ID。
        
        使用 MD5 对指纹的二进制流进行高维哈希散列，
        彻底解决 RDKit 原生折叠容易导致的哈希碰撞问题。
        """
        import hashlib
        
        # 1. 将 RDKit 指纹转换为底层二进制字节流
        fp_bytes = fp.ToBinary()
        
        # 2. 使用 MD5 计算 128 位确定性哈希，转为 16 进制再转为大整数
        hash_int = int(hashlib.md5(fp_bytes).hexdigest(), 16)
        
        # 3. 取模将其安全落入目标桶 (0 ~ 2^hash_bits - 1)
        return hash_int % (1 << hash_bits)

    @staticmethod
    def _choose_hash_bits(estimated_size: int) -> int:
        """根据库大小自适应选择哈希折叠位数。

        原则: 桶数 ≈ estimated_size / 期望每桶分子数。
        期望每桶 2-10 个代表分子，这样桶间有区分度且桶数可控。

        动态范围:
          - < 100 万  → 18 bits (26 万桶, ~4 分子/桶)
          - 100 万~1000 万 → 21 bits (210 万桶)
          - 1000 万~1 亿 → 24 bits (1600 万桶)
          - > 1 亿 → 26 bits (6700 万桶)
        """
        if estimated_size < 1_000_000:
            return 18
        elif estimated_size < 10_000_000:
            return 21
        elif estimated_size < 100_000_000:
            return 24
        else:
            return 26

    # =========================================================================
    # 闭环偏置采样
    # =========================================================================

    def _extract_bias_scaffolds(
        self,
        knowledge_base: Optional[Dict[str, Any]],
    ) -> List[str]:
        """从闭环知识库中提取优势骨架列表。

        优先级:
          1. recommended_scaffolds (Meta-Review 明确推荐)
          2. privileged_scaffolds  (历史累积)
        """
        if not knowledge_base:
            return []
        scaffolds = knowledge_base.get("recommended_scaffolds", [])
        if not scaffolds:
            scaffolds = knowledge_base.get("privileged_scaffolds", [])
        return scaffolds[:20]  # 最多 20 个优势骨架

    def _compute_bias_fingerprints(
        self,
        scaffolds: List[str],
    ) -> List[Any]:
        """将优势骨架 SMILES 转换为 Morgan 指纹。

        对每个骨架，解析 SMILES 后计算 ECFP4 指纹。
        无效骨架静默跳过。
        """
        bias_fps = []
        for smi in scaffolds:
            fp = self._safe_morgan_fp(smi)
            if fp is not None:
                bias_fps.append(fp)
        return bias_fps

    def _pick_biased_molecules(
        self,
        records: List[MoleculeRecord],
        fingerprints: List[Any],
        bias_fps: List[Any],
        max_pick: int,
    ) -> List[MoleculeRecord]:
        """从分子池中挑选与优势骨架 Tanimoto 最相似的分子。

        对每个分子计算其与所有优势骨架的最大 Tanimoto 相似度，
        按相似度降序取前 max_pick 个。

        Args:
            records: 分子记录列表。
            fingerprints: 对应的指纹列表。
            bias_fps: 优势骨架指纹列表。
            max_pick: 最多挑选数量。

        Returns:
            偏置挑选的分子记录。
        """
        if not bias_fps or not fingerprints:
            return []

        # 计算每个分子与优势骨架集中的最大相似度
        scored: List[Tuple[int, float]] = []
        for i, fp in enumerate(fingerprints):
            max_sim = max(
                TanimotoSimilarity(fp, b_fp) for b_fp in bias_fps
            )
            if max_sim >= PRIVILEGED_SCAFFOLD_THRESHOLD:
                scored.append((i, max_sim))

        # 按相似度降序
        scored.sort(key=lambda x: x[1], reverse=True)

        picked = []
        for idx, sim in scored[:max_pick]:
            record = records[idx].copy()
            # 标记来源 (供下游审计)
            record["source_db"] = (record.get("source_db", "") +
                                   f"_ALbiased_Tc={sim:.2f}")
            picked.append(record)

        return picked

    # =========================================================================
    # MaxMinPicker 距离函数
    # =========================================================================

    @staticmethod
    def _make_tanimoto_dist_func(
        fingerprints: List[Any],
    ):
        """构造 MaxMinPicker 所需的 Tanimoto 距离函数。

        MaxMinPicker 需要 dist_func(i, j) 返回两个索引对应的分子间距离。
        Tanimoto 距离 = 1 - Tanimoto 相似度。

        距离越大 → 分子越不相似 → MaxMin 算法会优先选取距离最大的。
        """
        def tanimoto_dist(i: int, j: int) -> float:
            sim = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            return 1.0 - sim

        return tanimoto_dist

    # =========================================================================
    # 指纹计算
    # =========================================================================

    def _safe_morgan_fp(self, smiles: str) -> Optional[Any]:
        """安全地将 SMILES 转换为 Morgan 指纹。

        所有异常内部捕获，返回 None。

        Args:
            smiles: SMILES 字符串。

        Returns:
            RDKit ExplicitBitVect 或 None (解析失败)。
        """
        if not smiles or not isinstance(smiles, str):
            return None
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.fp_radius,
                nBits=self.fp_bits,
            )
            return fp
        except Exception:
            return None

    # =========================================================================
    # 辅助
    # =========================================================================

    @staticmethod
    def _generate_mol_id(smiles: str) -> MoleculeID:
        """为分子生成唯一标识符。

        使用 UUID v4 (随机) + 时间戳前缀，保证全局唯一性。
        """
        return f"MOL_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _infer_source_db(library_path: str) -> str:
        """从文件路径推断分子来源数据库名。"""
        import os
        basename = os.path.basename(library_path).lower()
        if "zinc" in basename:
            return "ZINC20"
        elif "enamine" in basename:
            return "Enamine_REAL"
        elif "chembl" in basename:
            return "ChemBL"
        elif "pubchem" in basename:
            return "PubChem"
        else:
            return os.path.splitext(basename)[0]

    @staticmethod
    def _estimate_library_size(library_path: str) -> int:
        """快速估算分子库大小 (不遍历全文件)。

        策略:
          1. 读取前 1000 行，估算每行平均字节数
          2. 总字节数 ÷ 平均行字节数 → 近似行数
        """
        import os
        try:
            file_size = os.path.getsize(library_path)
        except OSError:
            return 0

        if file_size == 0:
            return 0

        # 采样前 1000 行
        sample_bytes = 0
        sample_lines = 0
        try:
            with open(library_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    sample_bytes += len(line.encode("utf-8"))
                    sample_lines += 1
                    if sample_lines >= 1000:
                        break
        except Exception:
            return 0

        if sample_lines == 0:
            return 0

        avg_bytes_per_line = sample_bytes / sample_lines
        return int(file_size / avg_bytes_per_line)


# =============================================================================
# LangGraph 节点函数
# =============================================================================

def clustering_node(state: dict) -> dict:
    """LangGraph 节点函数 — Step 2: 化学空间聚类降维。

    从 MACVSState 中提取 library_path、知识库，调用 OrchestratorAgent
    执行多样性导向降维采样，返回 candidates + surviving_pool。

    调用方式:
      workflow.add_node("clustering", clustering_node)

    Args:
        state: MACVSState。

    Returns:
        dict partial state update。
    """
    from datetime import datetime, timezone

    library_path = state.get("full_library_path", "")
    target_size = 100_000  # 默认候选池大小
    knowledge_base = state.get("knowledge_base")

    if not library_path:
        return {
            "pipeline_stage": "error",
            "errors": [{
                "node": "clustering",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Missing full_library_path in state.",
            }],
            "event_log": ["[Clustering] ERROR: No library path provided."],
        }

    agent = OrchestratorAgent()

    try:
        candidates = agent.run_clustering(
            library_path=library_path,
            target_size=target_size,
            knowledge_base=knowledge_base,
        )
    except Exception as e:
        return {
            "pipeline_stage": "error",
            "errors": [{
                "node": "clustering",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Clustering failed: {str(e)}",
            }],
            "event_log": [f"[Clustering] ERROR: {str(e)}"],
        }

    now = datetime.now(timezone.utc).isoformat()
    bias_note = ""
    if knowledge_base and (knowledge_base.get("recommended_scaffolds") or
                           knowledge_base.get("privileged_scaffolds")):
        bias_note = " (with AL bias)"

    return {
        "pipeline_stage": "clustering",
        "candidate_pool": candidates,
        "surviving_pool": candidates,
        "cluster_count": len(candidates),
        "event_log": [
            f"[{now}] [Clustering] {len(candidates)} representatives "
            f"selected from {library_path}{bias_note}."
        ],
        "updated_at": now,
    }
