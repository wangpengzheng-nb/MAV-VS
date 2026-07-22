#!/usr/bin/env python3
"""ADMET-AI 预测 wrapper — 供 ToolManager 通过 subprocess 调用。

用法:
    python scripts/admet_wrapper.py <input_csv> <output_csv>

从 autovs-admet conda 环境中运行（通过 ToolManager 的 run_argv 指定路径）。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} <input.csv> <output.csv>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.is_file():
        print(f"输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 读取输入 CSV（source_id, smiles）
    molecules: list[dict[str, str]] = []
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("smiles"):
                molecules.append(row)

    if not molecules:
        print("输入 CSV 中没有有效分子", file=sys.stderr)
        sys.exit(1)

    # 调用 ADMET-AI 预测
    try:
        from admet_ai.admet_predict import predict  # type: ignore[import-untyped]

        smiles_list = [m["smiles"] for m in molecules]
        results = predict(smiles_list)

        # 写入输出
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            if results:
                writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
        print(json.dumps({"status": "ok", "molecules": len(results)}))
    except ImportError:
        print("ADMET-AI 包未安装。请确认已在 autovs-admet 环境中运行: conda activate autovs-admet", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ADMET-AI 预测失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
