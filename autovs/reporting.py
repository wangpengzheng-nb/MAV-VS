from __future__ import annotations

import html
import json
from pathlib import Path


def generate_report(task_id: str, task_dir: Path, *, request: dict, plan: dict,
                    results: list[dict], rejected_strategies: list[dict],
                    health: dict, jobs: list[dict], artifacts: list[dict],
                    pocket_resolution: dict | None = None,
                    input_manifest: dict | None = None,
                    candidate_strategies: list[dict] | None = None) -> dict[str, str]:
    report_dir = task_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path, html_path = report_dir / "report.md", report_dir / "report.html"
    manifest_path = report_dir / "artifact_manifest.json"
    lines = [
        f"# AutoVS-Agent 可复现虚拟筛选报告", "", f"- Task ID: `{task_id}`",
        f"- Query: {request.get('query', '')}", f"- pH: {request.get('ph', 7.4)}", "",
        "## 输入绑定", "",
    ]
    library_asset = (input_manifest or {}).get("library_asset", {})
    target_asset = (input_manifest or {}).get("target_asset", {})
    lines.extend([
        f"- Library source: `{library_asset.get('source', 'unknown')}`",
        f"- Library version: `{library_asset.get('version') or 'user-upload'}`",
        f"- Library SHA256: `{library_asset.get('sha256', '')}`",
        f"- Accepted / quarantined: {library_asset.get('accepted_records', '?')} / {library_asset.get('quarantined_records', '?')}",
        f"- Target source: `{target_asset.get('source', 'unknown')}`",
        f"- Target PDB ID: `{target_asset.get('pdb_id') or 'user-upload'}`",
        f"- Target SHA256: `{target_asset.get('sha256', '')}`", "",
    ])
    warnings = (input_manifest or {}).get("warnings", [])
    if warnings:
        lines.extend(["### 输入提示", "", *[f"- {warning}" for warning in warnings], ""])
    lines.extend(["## 执行策略", "", f"- Strategy: `{plan.get('strategy_id', 'baseline')}`",
                  f"- Plan version: `{plan.get('plan_version', '1.0')}`", ""])
    if candidate_strategies:
        lines.extend(["## 候选策略组合", ""])
        for item in candidate_strategies[:12]:
            missing = item.get("missing_capabilities") or []
            lines.append(
                f"- `{item.get('strategy_name', item.get('strategy_id', '?'))}` "
                f"[{item.get('execution_status', 'unknown')}] "
                f"axis={item.get('diversity_axis', '')}; focus={item.get('problem_focus', '')}"
            )
            if missing:
                lines.append(f"  - capability gaps: {', '.join(str(x) for x in missing[:4])}")
        lines.append("")
    lines.extend(["## 口袋预检", ""])
    pocket = (pocket_resolution or {}).get("selected_pocket", {})
    if pocket:
        lines.extend([
            f"- Pocket ID: `{pocket.get('pocket_id', '')}`",
            f"- Source: `{pocket.get('source', '')}`",
            f"- Confidence: `{pocket.get('confidence', '')}`",
            f"- Center: `{pocket.get('center', '')}`",
            f"- Size: `{pocket.get('size', '')}`",
            f"- Alternate pockets: {len((pocket_resolution or {}).get('alternate_pockets', []))}", "",
        ])
    else:
        lines.extend(["- Pocket resolution unavailable", ""])
    lines.extend([
        "## 最终候选", "",
        "| Rank | Source ID | Docking affinity | CNN_VS | PLIP | ADMET risk | MMGBSA | Final score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
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


def generate_failure_report(task_id: str, task_dir: Path, *, request: dict, error: str) -> dict[str, str]:
    report_dir = task_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path, html_path = report_dir / "failure_report.md", report_dir / "failure_report.html"
    manifest = {}
    manifest_path = request.get("input_manifest_path")
    if manifest_path and Path(manifest_path).is_file():
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    library_asset = manifest.get("library_asset", {})
    target_asset = manifest.get("target_asset", {})
    markdown = "\n".join([
        "# AutoVS-Agent 预检失败报告", "", f"- Task ID: `{task_id}`",
        f"- Query: {request.get('query', '')}",
        f"- Library source: `{library_asset.get('source', 'unknown')}`",
        f"- Target source: `{target_asset.get('source', 'unknown')}`", "",
        "## 失败原因", "", f"`{error}`", "",
        "## 计算状态", "", "未开始分子对接，未生成候选化合物或模拟科学结果。", "",
        "如果错误为 `pocket unresolved`，请提供口袋中心、可映射关键残基，或包含合理非共价配体的预处理 PDB。", "",
    ])
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>AutoVS Failure Report</title>"
        "<style>body{font:16px/1.55 sans-serif;max-width:1000px;margin:40px auto;padding:0 20px}pre{white-space:pre-wrap}</style>"
        f"<pre>{html.escape(markdown)}</pre>", encoding="utf-8",
    )
    return {"failure_report_md": str(md_path), "failure_report_html": str(html_path)}
