from __future__ import annotations

import html
import json
from pathlib import Path


def generate_report(task_id: str, task_dir: Path, *, request: dict, plan: dict,
                    results: list[dict], rejected_strategies: list[dict],
                    health: dict, jobs: list[dict], artifacts: list[dict]) -> dict[str, str]:
    report_dir = task_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path, html_path = report_dir / "report.md", report_dir / "report.html"
    manifest_path = report_dir / "artifact_manifest.json"
    lines = [
        f"# AutoVS-Agent 可复现虚拟筛选报告", "", f"- Task ID: `{task_id}`",
        f"- Query: {request.get('query', '')}", f"- Protein: `{request.get('protein_path', '')}`",
        f"- Library: `{request.get('library_path', '')}`", f"- pH: {request.get('ph', 7.4)}", "",
        "## 执行策略", "", f"- Strategy: `{plan.get('strategy_id', 'baseline')}`",
        f"- Plan version: `{plan.get('plan_version', '1.0')}`", "",
        "## 最终候选", "",
        "| Rank | Source ID | Docking affinity | CNN_VS | PLIP | ADMET risk | MMGBSA | Final score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append("| {rank} | {source_id} | {docking_affinity} | {cnn_vs} | {plip_score} | {admet_risk} | {mmgbsa_delta_total} | {final_score} |".format(
            rank=row.get("rank", ""), source_id=row.get("source_id", ""),
            docking_affinity=row.get("docking_affinity", ""), cnn_vs=row.get("cnn_vs", ""),
            plip_score=row.get("plip_score", ""), admet_risk=row.get("admet_risk", ""),
            mmgbsa_delta_total=row.get("mmgbsa_delta_total", ""), final_score=row.get("final_score", "")))
    lines.extend(["", "## 被拒绝策略", ""])
    lines.extend([f"- `{item['strategy_name']}`: {item['reason']}" for item in rejected_strategies] or ["- 无"])
    lines.extend(["", "## 运行状态", "", f"- Environment status: `{health.get('status')}`",
                  f"- Jobs: {len(jobs)}", f"- Artifacts indexed: {len(artifacts)}", "",
                  "> 计算评分是候选优先级证据，不等同于实验活性证明。", ""])
    markdown = "\n".join(lines)
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>AutoVS Report</title>"
        "<style>body{font:16px/1.55 sans-serif;max-width:1100px;margin:40px auto;padding:0 20px}"
        "pre{white-space:pre-wrap}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:6px}</style>"
        f"<pre>{html.escape(markdown)}</pre>", encoding="utf-8")
    manifest_path.write_text(json.dumps({"task_id": task_id, "artifacts": artifacts}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report_md": str(md_path), "report_html": str(html_path), "artifact_manifest": str(manifest_path)}

