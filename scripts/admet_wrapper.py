#!/usr/bin/env python3
"""ADMET-AI v2.0.1 预测 wrapper — 供 ToolManager 通过 conda run 调用。

用法:
    python scripts/admet_wrapper.py <input_csv> <output_csv>

input_csv 至少包含 source_id 和 smiles 列。
输出 CSV 包含原始列 + 104 个 ADMET 预测属性 + DrugBank 百分位。

从 autovs-admet conda 环境中运行。
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

    # 读取输入 CSV（需要 source_id, smiles 列）
    molecules: list[dict[str, str]] = []
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("smiles", "").strip():
                molecules.append({
                    "source_id": row.get("source_id", "").strip(),
                    "smiles": row["smiles"].strip(),
                })

    if not molecules:
        print("输入 CSV 中没有有效分子", file=sys.stderr)
        sys.exit(1)

    # 调用 ADMET-AI v2.0.1 预测
    try:
        from admet_ai import ADMETModel  # type: ignore[import-untyped]

        smiles_list = [m["smiles"] for m in molecules]
        model = ADMETModel()
        preds_df = model.predict(smiles_list)

        # 将 source_id 加回去作为第一列
        source_ids = [m["source_id"] for m in molecules]
        preds_df.insert(0, "source_id", source_ids)

        # 写入输出 CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(output_path, index=False)

        print(json.dumps({
            "status": "ok",
            "molecules": len(molecules),
            "properties": len(preds_df.columns) - 1,
        }))
    except ImportError as exc:
        print(
            f"ADMET-AI 包未安装。请确认已在 autovs-admet 环境中运行: "
            f"conda activate autovs-admet\n{exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"ADMET-AI 预测失败: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
