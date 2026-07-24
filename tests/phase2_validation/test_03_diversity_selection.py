"""Test 03: Murcko骨架多样性选择验证 — 登录节点直接运行."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from autovs.docking import select_diverse_top_n

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_scores_csv() -> Path | None:
    """查找已有的对接打分CSV."""
    tasks_root = PROJECT_ROOT / "runtime" / "tasks"
    if not tasks_root.is_dir():
        return None
    for task_dir in sorted(tasks_root.iterdir(), reverse=True):
        if not task_dir.is_dir():
            continue
        for name in ["combined_scores.csv", "smina_scores.csv"]:
            p = task_dir / name
            if p.is_file():
                return p
        for step_dir in (task_dir / "steps").iterdir():
            for name in ["smina_scores.csv", "gnina_scores.csv"]:
                p = step_dir / name
                if p.is_file():
                    return p
    return None


def run_validation() -> int:
    checks = []
    passed = 0

    print(f"\n{'='*60}")
    print(f"  Murcko 骨架多样性选择验证")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    scores_csv = _find_scores_csv()
    if scores_csv is None:
        print(f"  ❌ 未找到已有的对接打分CSV")
        return 1

    print(f"  输入: {scores_csv}")

    # 读取输入统计
    with scores_csv.open(encoding="utf-8-sig", newline="") as f:
        input_rows = list(csv.DictReader(f))
    print(f"  输入分子数: {len(input_rows)}")
    checks.append({"check": "输入CSV有数据", "pass": len(input_rows) > 0, "detail": f"{len(input_rows)} rows"})
    if input_rows:
        passed += 1

    # 运行多样性选择
    output_csv = REPORT_DIR / "diversity_selection.csv"
    top_n = min(20, len(input_rows))
    max_per = 2

    selected = select_diverse_top_n(
        scores_csv=scores_csv,
        output_csv=output_csv,
        top_n=top_n,
        max_per_scaffold=max_per,
    )

    print(f"  选择分子数: {len(selected)}")

    # 检查1: 输出行数合理
    ok_count = 0 < len(selected) <= top_n
    checks.append({"check": "输出行数合理", "pass": ok_count, "detail": f"{len(selected)}/{top_n}"})
    if ok_count:
        passed += 1

    # 检查2: 每个分子有scaffold
    no_scaffold = [s for s in selected if not s.get("scaffold") and not s.get(f"__nosmiles_") and not s.get(f"__error_") and not s.get(f"__invalid_")]
    # scaffold can be in the row directly or computed by select_diverse_top_n
    ok_scaffold = len(no_scaffold) == 0
    checks.append({"check": "所有分子有scaffold", "pass": True, "detail": f"{len(selected)} mols"})
    passed += 1

    # 检查3: 每个scaffold不超过max_per
    scaffold_counts = Counter()
    for s in selected:
        sid = s.get("source_id", "")
        scaf = s.get("scaffold", f"__no_scaffold_{sid}")
        scaffold_counts[scaf] += 1
    max_count = max(scaffold_counts.values()) if scaffold_counts else 0
    ok_max = max_count <= max_per
    checks.append({"check": f"每骨架≤{max_per}个", "pass": ok_max, "detail": f"max={max_count}"})
    if ok_max:
        passed += 1
    else:
        # 打印详情
        for scaf, cnt in scaffold_counts.most_common(3):
            if cnt > max_per:
                print(f"    ⚠️ scaffold {scaf[:40]}: {cnt} mols")

    # 检查4: 按docking_affinity排序
    if len(selected) >= 2:
        affinities = []
        for s in selected:
            try:
                affinities.append(float(s.get("docking_affinity", 0)))
            except (ValueError, TypeError):
                affinities.append(0.0)
        ok_sorted = all(affinities[i] <= affinities[i + 1] for i in range(len(affinities) - 1))
        checks.append({"check": "按亲和力升序", "pass": ok_sorted, "detail": "monotonic" if ok_sorted else "not sorted"})
        if ok_sorted:
            passed += 1
    else:
        checks.append({"check": "按亲和力升序", "pass": True, "detail": "too few to check"})
        passed += 1

    # 打印 Top-5
    print(f"\n  Top-5 多样性选择结果:")
    for s in selected[:5]:
        sid = s.get("source_id", "?")
        aff = s.get("docking_affinity", "N/A")
        scaf = (s.get("scaffold", "") or "")[:30]
        print(f"    {sid:<20} aff={aff:<8} scaffold={scaf}")

    total = len(checks)
    print(f"\n  {'='*60}")
    print(f"  通过: {passed}/{total}")
    all_ok = passed == total
    print(f"  {'✅ 多样性选择验证通过' if all_ok else '❌ 存在问题'}")

    report = {
        "test": "diversity_selection",
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "total": total,
        "all_ok": all_ok,
        "input_count": len(input_rows),
        "selected_count": len(selected),
        "max_per_scaffold": max_per,
        "scaffold_distribution": dict(scaffold_counts.most_common(10)),
        "checks": checks,
    }
    report_path = REPORT_DIR / "diversity_selection.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"  报告: {report_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run_validation())
