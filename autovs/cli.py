from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autovs.capabilities import health_report
from autovs.config import load_settings
from autovs.pipeline import PipelineService
from autovs.schemas import PocketSpec, TaskRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autovs", description="AutoVS-Agent unattended virtual screening")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="submit a virtual-screening task")
    run.add_argument("--query", required=True); run.add_argument("--protein", required=True); run.add_argument("--library", required=True)
    run.add_argument("--center", nargs=3, type=float); run.add_argument("--size", nargs=3, type=float, default=[24, 24, 24])
    run.add_argument("--key-residue", action="append", default=[]); run.add_argument("--ph", type=float, default=7.4)
    run.add_argument("--cpu-only", action="store_true"); run.add_argument("--baseline", action="store_true", help="skip LLM planning for CPU diagnostics")
    run.add_argument("--wait", action="store_true")
    for name in ("status", "resume", "report"):
        cmd = sub.add_parser(name); cmd.add_argument("task_id")
    sub.add_parser("doctor", help="check tool environments and capabilities")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = load_settings(); service = PipelineService(settings)
    if args.command == "doctor":
        print(json.dumps(health_report(settings), ensure_ascii=False, indent=2)); return 0
    if args.command == "run":
        request = TaskRequest(query=args.query, protein_path=args.protein, library_path=args.library,
                              pocket=PocketSpec(center=tuple(args.center) if args.center else None, size=tuple(args.size), key_residues=args.key_residue),
                              ph=args.ph, cpu_only=args.cpu_only)
        if args.wait:
            result = service.run_sync(request, use_llm_planning=not args.baseline)
            print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result.get("status") == "succeeded" else 1
        task_id = service.submit(request, use_llm_planning=not args.baseline)
        print(task_id); return 0
    if args.command == "resume":
        service.resume(args.task_id); print(args.task_id); return 0
    task = service.get_task(args.task_id)
    if not task:
        print("task not found", file=sys.stderr); return 2
    if args.command == "report":
        report = (task.get("result") or {}).get("reports", {}).get("report_html")
        print(report or "report is not available"); return 0 if report else 1
    print(json.dumps(task, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
