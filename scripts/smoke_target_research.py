"""Manual live-API smoke test for target identity resolution.

This script is intentionally excluded from pytest because it depends on the
current availability and content of external biomedical APIs.
"""

from __future__ import annotations

import argparse
import json

from src.agents.target_scout import TargetScoutAgent


CASES = [
    ("为人源 EGFR 寻找高选择性非共价抑制剂", "EGFR"),
    ("为人源 BCL-2 寻找抑制剂并避开同源蛋白", "BCL-2"),
    ("为人源 EP2 受体寻找激动剂", "EP2"),
    ("为人源 KRAS 寻找抑制剂", "KRAS"),
    ("针对结核分枝杆菌 inhA 寻找抑制剂", "inhA"),
    ("针对 SARS-CoV-2 main protease 寻找抑制剂", "main protease"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also collect PDB, ChEMBL, PubMed and trial evidence")
    args = parser.parse_args()
    agent = TargetScoutAgent()
    failed = False
    for query, hint in CASES:
        resolution = agent.resolve_target(query, target_hint=hint)
        print(json.dumps({
            "query": query, "status": resolution.get("status"),
            "identity": resolution.get("identity"),
            "candidates": resolution.get("candidates", [])[:3],
            "reason": resolution.get("reason", ""),
        }, ensure_ascii=False, indent=2))
        if resolution.get("status") == "resolved" and args.full:
            accession = resolution["identity"]["uniprot_accession"]
            report = agent.deep_research(query, target_hint=hint, selected_accession=accession)
            print(json.dumps({
                "target": report["target_name"], "status": report["status"],
                "structures": len(report["verified_pdb_structures"]),
                "activities": len(report["chembl_activities"]),
                "evidence_gaps": report["evidence_gaps"],
            }, ensure_ascii=False, indent=2))
        failed = failed or resolution.get("status") not in {"resolved", "needs_confirmation"}
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
