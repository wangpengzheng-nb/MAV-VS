"""Phase 2 验证报告生成器 — 聚合所有测试结果."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent


def load_result(name: str) -> dict | None:
    """加载单个测试结果JSON."""
    path = REPORT_DIR / name
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return None


def run() -> int:
    results = {}
    for name, label in [
        ("config_validation.json", "配置验证"),
        ("engine_selection.json", "引擎自动选择"),
        ("diversity_selection.json", "多样性选择"),
        ("gnina_result.json", "GNINA GPU对接"),
        ("diffdock_result.json", "DiffDock GPU对接"),
    ]:
        data = load_result(name)
        if data:
            results[label] = data
        else:
            results[label] = {"all_ok": False, "status": "not_run"}

    total_passed = sum(1 for r in results.values() if r.get("all_ok", False))
    total_tests = len(results)

    # 生成Markdown报告
    report = []
    report.append("# AutoVS Phase 2 GPU验证报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**测试靶点**: BCL-2 / 6O0K (PPI)\n")
    report.append(f"**总体通过**: {total_passed}/{total_tests}\n")

    report.append("\n## 测试结果总览\n")
    report.append("| 测试 | 状态 | 详情 |")
    report.append("|------|------|------|")
    for label, data in results.items():
        ok = data.get("all_ok", False)
        status = "✅ PASS" if ok else ("❌ FAIL" if data.get("status") != "not_run" else "⏭️ SKIP")
        detail = ""
        if data.get("status") == "not_run":
            detail = "未运行"
        else:
            p = data.get("passed", 0)
            t = data.get("total", 0)
            detail = f"{p}/{t}"
        report.append(f"| {label} | {status} | {detail} |")

    report.append("\n## 详细结果\n")

    # 配置验证
    if "config_validation.json":
        data = load_result("config_validation.json")
        if data:
            report.append("### 配置验证\n")
            for c in data.get("checks", []):
                icon = "✅" if c["pass"] else "❌"
                report.append(f"- {icon} {c['check']}: {c.get('detail', '')}")

    # 引擎选择
    if "engine_selection.json":
        data = load_result("engine_selection.json")
        if data:
            report.append("\n### 引擎自动选择逻辑\n")
            report.append("| 场景 | 预期 | 实际 | 结果 |")
            report.append("|------|------|------|------|")
            for c in data.get("cases", [])[:10]:
                icon = "✅" if c["pass"] else "❌"
                report.append(f"| {c['case']} | {c['expected']} | {c['actual']} | {icon} |")

    # 多样性选择
    if "diversity_selection.json":
        data = load_result("diversity_selection.json")
        if data:
            report.append("\n### 多样性选择\n")
            report.append(f"- 输入分子: {data.get('input_count', 0)}")
            report.append(f"- 选择分子: {data.get('selected_count', 0)}")
            report.append(f"- 每骨架上限: {data.get('max_per_scaffold', 0)}")
            for c in data.get("checks", []):
                icon = "✅" if c["pass"] else "❌"
                report.append(f"- {icon} {c['check']}: {c.get('detail', '')}")

    # GNINA
    if "gnina_result.json":
        data = load_result("gnina_result.json")
        if data:
            report.append("\n### GNINA GPU对接\n")
            if data.get("status") == "not_run":
                report.append("⏭️ 未运行")
            else:
                report.append(f"- 对接姿态数: {data.get('poses_count', 0)}")
                report.append(f"- 打分分子数: {data.get('mols_in_csv', 0)}")
                report.append(f"- Top CNN_VS: {data.get('top_cnn_vs', 'N/A')}")
                for c in data.get("checks", []):
                    icon = "✅" if c["pass"] else "❌"
                    report.append(f"- {icon} {c['check']}: {c.get('detail', '')}")

    # DiffDock
    if "diffdock_result.json":
        data = load_result("diffdock_result.json")
        if data:
            report.append("\n### DiffDock PPI对接\n")
            if data.get("status") == "not_run":
                report.append("⏭️ 未运行")
            else:
                report.append(f"- 测试配体数: {data.get('ligands_tested', 0)}")
                for r in data.get("results", []):
                    icon = "✅" if r.get("success") else "❌"
                    report.append(f"- {icon} {r.get('name', '?')}: conf={r.get('top_confidence', 'N/A')}, poses={r.get('poses_count', 0)}")

    report_content = "\n".join(report) + "\n"

    # 写入Markdown文件
    report_path = REPORT_DIR / "phase2_report.md"
    report_path.write_text(report_content, encoding="utf-8")

    # 也输出到stdout
    print(report_content)
    print(f"报告已保存: {report_path}")

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(run())
