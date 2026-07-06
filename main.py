"""
AutoVS-Agent: 主入口
====================
基于纯干实验闭环 (Dry-Lab Closed-Loop) 的 8 步自动化虚拟筛选管道。

用法:
  python main.py --target Bcl-2 --pdb 60OK.pdb --library zinc20.smi

流程:
  Step 1  → Strategy Agent 战前侦察
  Step 2  → 化学空间聚类降维
  Step 3  → Watchdog 小样本演习
  Step 4  → 高通量虚拟筛选 (HTVS)
  Step 5  → MedChem Committee 绝对值淘汰
  Step 6  → MPO Elo 锦标赛排序
  Step 7  → MD Oracle 终极验证
  Step 8  → Meta-Review 闭环复盘 → Step 2 (迭代)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.state import (
    MACVSState,
    TargetInfo,
    create_initial_state,
    export_checkpoint,
    is_pipeline_complete,
)
from src.graph.workflow import create_workflow, run_pipeline


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="AutoVS-Agent: AI-Driven 8-Step Virtual Screening Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必需参数
    parser.add_argument(
        "--target", "-t", type=str, required=True,
        help="靶点蛋白名称 (e.g. Bcl-2, Bcl-xl, EGFR)",
    )
    parser.add_argument(
        "--pdb", type=str, required=True,
        help="受体 PDB 文件路径",
    )
    parser.add_argument(
        "--library", "-l", type=str, required=True,
        help="分子库文件路径 (.smi / .sdf)",
    )

    # 可选参数
    parser.add_argument("--uniprot", type=str, default="", help="UniProt ID")
    parser.add_argument("--pdb-id", type=str, default="", help="PDB 结构 ID (e.g. 60OK)")
    parser.add_argument("--target-class", type=str, default="PPI",
                        choices=["PPI", "Kinase", "GPCR", "Protease", "other"],
                        help="靶点类型")
    parser.add_argument("--organism", type=str, default="Homo sapiens")
    parser.add_argument("--description", type=str, default="")

    # 对接/筛选参数
    parser.add_argument("--box-center", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="对接盒子中心 (x y z)")
    parser.add_argument("--box-size", type=float, nargs=3, default=[20.0, 20.0, 20.0],
                        help="对接盒子尺寸 (sx sy sz)")
    parser.add_argument("--key-residues", type=str, nargs="*", default=[],
                        help="关键结合位点残基 (e.g. ASP103 TRP144)")

    # 管道控制参数
    parser.add_argument("--max-iter", type=int, default=5, help="最大闭环迭代轮次")
    parser.add_argument("--htvs-top-n", type=int, default=2000, help="HTVS 保留分子数")
    parser.add_argument("--medchem-target", type=int, default=300, help="MedChem 目标存活数")
    parser.add_argument("--mpo-target", type=int, default=20, help="锦标赛目标存活数")
    parser.add_argument("--md-ns", type=float, default=50.0, help="MD 模拟时长 (ns)")

    # Elo 参数
    parser.add_argument("--elo-k", type=float, default=32.0, help="Elo K 因子")
    parser.add_argument("--elo-init", type=float, default=1500.0, help="Elo 初始积分")
    parser.add_argument("--tournament-rounds", type=int, default=3, help="锦标赛轮次")

    # 输出
    parser.add_argument("--output", "-o", type=str, default="./output",
                        help="输出目录路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志输出")

    return parser.parse_args()


def build_target_info(args: argparse.Namespace) -> TargetInfo:
    """从命令行参数构建 TargetInfo。"""
    return TargetInfo(
        target_name=args.target,
        uniprot_id=args.uniprot,
        pdb_id=args.pdb_id,
        pdb_path=args.pdb,
        binding_site_center=args.box_center,
        binding_site_size=args.box_size,
        key_residues=args.key_residues,
        target_class=args.target_class,
        description=args.description or f"Virtual screening target: {args.target}",
        organism=args.organism,
    )


def main() -> int:
    """主函数: 解析参数 → 创建初始状态 → 运行管道 → 输出结果。"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 构建靶点信息
    target_info = build_target_info(args)

    # 统计库大小
    total_size = 0
    if os.path.exists(args.library):
        with open(args.library, "r") as f:
            total_size = sum(1 for line in f if line.strip())
        print(f"📚 分子库: {args.library} ({total_size:,} molecules)")

    print(f"🎯 靶点: {args.target} ({args.target_class})")
    print(f"📐 PDB: {args.pdb}")
    print(f"🔄 最大迭代: {args.max_iter} | MD 时长: {args.md_ns}ns")
    print(f"📊 HTVS→{args.htvs_top_n} | MedChem→{args.medchem_target} | MPO→{args.mpo_target}")
    print(f"{'='*60}")

    # 创建初始状态
    initial_state = create_initial_state(
        target_info=target_info,
        full_library_path=args.library,
        total_library_size=total_size,
        max_iterations=args.max_iter,
        htvs_top_n=args.htvs_top_n,
        medchem_target=args.medchem_target,
        mpo_target=args.mpo_target,
        md_simulation_ns=args.md_ns,
        elo_k_factor=args.elo_k,
        elo_initial_rating=args.elo_init,
        tournament_rounds=args.tournament_rounds,
    )

    print(f"🚀 Session: {initial_state['session_id']}")
    print(f"⏰ Started:  {initial_state['created_at']}")
    print(f"{'='*60}")

    # 运行管道
    try:
        final_state = run_pipeline(initial_state)

        # 输出结果
        print(f"\n{'='*60}")
        print(f"✅ 管道完成: {final_state['pipeline_stage']}")
        print(f"📋 最终命中: {len(final_state['final_hits'])} molecules")
        print(f"🔄 总迭代:  {final_state['al_state']['iteration']}")

        if final_state["final_hits"]:
            print(f"\n🏆 Top Hits:")
            for i, hit in enumerate(final_state["final_hits"][:10], 1):
                print(f"  {i:2d}. {hit.get('mol_id', 'N/A')}  "
                      f"ΔG={hit.get('md_dG', 'N/A')}  "
                      f"SMILES={hit.get('smiles', 'N/A')[:50]}...")

        # 导出 checkpoint
        ckpt = export_checkpoint(final_state)
        import json
        ckpt_path = os.path.join(args.output, f"checkpoint_{final_state['session_id'][:8]}.json")
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Checkpoint saved: {ckpt_path}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  管道被用户中断")
        return 130
    except Exception as e:
        print(f"\n❌ 管道异常: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
