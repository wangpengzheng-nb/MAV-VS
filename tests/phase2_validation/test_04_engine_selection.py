"""Test 04: 引擎自动选择逻辑验证 — 登录节点直接运行."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from autovs.docking import detect_target_type, select_docking_engine

REPORT_DIR = Path(__file__).resolve().parent

# ── BCL-2 PPI 靶点 research (模拟) ──
BCL2_RESEARCH = {
    "identity": {
        "gene_symbol": "BCL2",
        "function": "Apoptosis regulator Bcl-2, protein-protein interaction inhibitor",
        "uniprot_id": "P10415",
    },
    "target_uniprot_id": "P10415",
    "gene_symbol": "BCL2",
}

# ── EGFR 激酶靶点 research (模拟) ──
EGFR_RESEARCH = {
    "identity": {
        "gene_symbol": "EGFR",
        "function": "Epidermal growth factor receptor tyrosine kinase",
        "uniprot_id": "P00533",
    },
    "target_uniprot_id": "P00533",
    "gene_symbol": "EGFR",
}


TEST_CASES = [
    # (target_name, research, cpu_only, gpu_available, strategy_params, expected_engine)
    ("BCL-2 PPI + GPU", BCL2_RESEARCH, False, True, None, "diffdock"),
    ("BCL-2 PPI + CPU only", BCL2_RESEARCH, True, True, None, "smina"),
    ("BCL-2 PPI + no GPU", BCL2_RESEARCH, False, False, None, "smina"),
    ("EGFR kinase + GPU", EGFR_RESEARCH, False, True, None, "gnina"),
    ("EGFR kinase + CPU only", EGFR_RESEARCH, True, True, None, "smina"),
    ("EGFR kinase + no GPU", EGFR_RESEARCH, False, False, None, "smina"),
    ("General target + GPU", None, False, True, None, "gnina"),
    ("General target + no GPU", None, False, False, None, "smina"),
    # 显式指定引擎（覆盖自动选择）
    ("BCL-2 + explicit smina", BCL2_RESEARCH, False, True, {"engine": "smina"}, "smina"),
    ("BCL-2 + explicit gnina", BCL2_RESEARCH, False, True, {"engine": "gnina"}, "gnina"),
    ("BCL-2 + explicit diffdock", BCL2_RESEARCH, False, True, {"engine": "diffdock"}, "diffdock"),
    ("EGFR + explicit diffdock", EGFR_RESEARCH, False, True, {"engine": "diffdock"}, "diffdock"),
    # 无效engine参数 → 自动fallback
    ("BCL-2 + invalid engine", BCL2_RESEARCH, False, True, {"engine": "invalid"}, "diffdock"),
    ("EGFR + invalid engine", EGFR_RESEARCH, False, True, {"engine": "invalid"}, "gnina"),
    # PPI关键词检测
    ("MDM2-p53 PPI", {"identity": {"function": "MDM2-p53 protein-protein interaction"}}, False, True, None, "diffdock"),
    ("XIAP PPI", {"identity": {"gene_symbol": "XIAP", "function": "inhibitor of apoptosis"}}, False, True, None, "diffdock"),
    ("GPCR receptor", {"identity": {"function": "G protein-coupled receptor"}}, False, True, None, "gnina"),
    ("COX-2 enzyme", {"identity": {"function": "cyclooxygenase-2 enzyme"}}, False, True, None, "gnina"),
    # detect_target_type 验证
    ("PPI detection", BCL2_RESEARCH, False, True, None, "diffdock"),
    ("enzyme detection", EGFR_RESEARCH, False, True, None, "gnina"),
]


def run_validation() -> int:
    checks = []
    passed = 0

    print(f"\n{'='*60}")
    print(f"  引擎自动选择逻辑验证")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    for i, (name, research, cpu_only, gpu_available, params, expected) in enumerate(TEST_CASES, 1):
        engine = select_docking_engine(
            strategy_params=params,
            research=research,
            gpu_available=gpu_available,
            cpu_only=cpu_only,
        )
        ok = engine == expected
        item = {
            "case": name,
            "expected": expected,
            "actual": engine,
            "pass": ok,
            "params": {
                "cpu_only": cpu_only,
                "gpu_available": gpu_available,
                "explicit_engine": params.get("engine") if params else None,
            },
        }
        checks.append(item)
        if ok:
            passed += 1
        icon = "✅" if ok else "❌"
        target_type = detect_target_type(research)
        print(f"  {icon} {name:<35} → {engine:<10} (expected: {expected}, target={target_type})")

    total = len(checks)
    print(f"  {'='*60}")
    print(f"  通过: {passed}/{total}")
    all_ok = passed == total
    print(f"  {'✅ 引擎选择逻辑验证通过' if all_ok else '❌ 存在失败用例'}")

    report = {
        "test": "engine_selection",
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "total": total,
        "all_ok": all_ok,
        "cases": checks,
    }
    report_path = REPORT_DIR / "engine_selection.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告: {report_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run_validation())
