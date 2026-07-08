#!/usr/bin/env python3
"""
AutoVS-Agent v2.0 锦标赛机制验证脚本
=====================================
模拟 KRAS G12D switch-II 隐蔽口袋的共价抑制剂虚拟筛选方案设计,
流式打印每个节点的状态变化, 重点展示:
  1. candidate_strategies 的内容
  2. 红军专家的互相攻击对话
  3. 裁判的打分和 Elo 排名
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 模拟输入: KRAS G12D
# =============================================================================

KRAS_G12D_TARGET = {
    "target_name": "KRAS G12D",
    "uniprot_id": "P01116",
    "pdb_id": "6GOD",
    "pdb_path": "",
    "binding_site_center": [0.0, 0.0, 0.0],
    "binding_site_size": [20.0, 20.0, 20.0],
    "key_residues": ["ASP12", "GLY60", "LYS117"],
    "target_class": "PPI",
    "description": (
        "KRAS 是最常见的致癌基因之一。G12D 突变 (Gly→Asp) 破坏了 GTPase 活性, "
        "使 KRAS 锁定在 GTP-bound 活性构象。Switch-II 区域含有一个可被诱导的隐蔽口袋 "
        "(cryptic pocket), 是共价抑制剂的关键结合位点。"
        "目标: 设计针对 G12D 突变半胱氨酸(需引入)或 switch-II 口袋的共价虚拟筛选方案。"
    ),
    "organism": "Homo sapiens",
}


# =============================================================================
# 美化打印函数
# =============================================================================

SEP = "=" * 70
SEP2 = "-" * 70

def print_header(text: str) -> None:
    print(f"\n{SEP}")
    print(f"  {text}")
    print(SEP)

def print_strategy(s: dict, idx: int) -> None:
    print(f"\n  📋 策略 {idx}: {s.get('strategy_name', '?')}")
    print(f"     🏷️  {s.get('strategy_tagline', '')}")
    print(f"     📐 方法: {s.get('approach_type', '?')}")
    print(f"     📊 存活率: {s.get('estimated_survival_rate', '?')}")
    print(f"     💡 {s.get('rationale', '')[:200]}...")
    af = s.get("absolute_filters", [])
    if af:
        print(f"     🔴 绝对过滤 ({len(af)}条):")
        for f in af:
            print(f"        - [{f.get('rule_id','?')}] {f.get('description','?')[:80]}")
    rr = s.get("relative_rankings", [])
    if rr:
        print(f"     🟡 相对排序 ({len(rr)}条):")
        for f in rr:
            print(f"        - [{f.get('rule_id','?')}] {f.get('description','?')[:80]}")
    cp = s.get("contingency_plan", {})
    if cp:
        print(f"     🟢 应急预案: {cp.get('trigger_condition', '?')}")
        steps = cp.get("relaxation_steps", [])
        for st in steps:
            print(f"        ↳ {st.get('rule_id','?')} → {st.get('new_value','?')} ({st.get('reason','')[:60]})")
    print(f"     ✅ 优势: {', '.join(s.get('strengths', [])[:3])}")
    print(f"     ⚠️  劣势: {', '.join(s.get('weaknesses', [])[:3])}")

def print_attack(atk: dict) -> None:
    sev_icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}.get(atk.get("severity", ""), "⚪")
    print(f"     {sev_icon} [{atk.get('persona_name', '?')}] 严重度={atk.get('severity','?')} | 认可度={atk.get('agreement_with_strategy', 0):.0%}")
    for p in atk.get("attack_points", []):
        print(f"        💢 {p}")
    for f in atk.get("suggested_fixes", []):
        print(f"        💊 建议: {f}")

def print_debate(record: dict) -> None:
    print(f"\n  ⚔️  Round: {record.get('round_id', '?')}")
    print(f"     {record.get('strategy_a', '?')} vs {record.get('strategy_b', '?')}")
    print(f"\n     📢 红军对 {record.get('strategy_a', '?')} 的攻击:")
    for atk in record.get("expert_attacks_on_a", []):
        print_attack(atk)
    print(f"\n     📢 红军对 {record.get('strategy_b', '?')} 的攻击:")
    for atk in record.get("expert_attacks_on_b", []):
        print_attack(atk)
    print(f"\n     ⚖️  裁判裁决:")
    print(f"        {record.get('judge_summary', '')[:300]}")
    print(f"     🏆 胜者: {record.get('winner', 'tie')}")
    print(f"     🎯 决定因素: {record.get('key_deciding_factor', '')[:150]}")


# =============================================================================
# 主测试: 流式运行 + 美化打印 (纯本地, 不调用 LLM)
# =============================================================================

def run_test() -> None:
    """运行锦标赛验证 (使用降级/启发式逻辑, 不依赖 LLM API)。"""
    print_header("AutoVS-Agent v2.0 锦标赛验证")
    print(f"  🎯 测试靶点: KRAS G12D")
    print(f"  📅 开始时间: {datetime.now().isoformat()}")
    print(f"  ⚠️  本测试使用降级/启发式逻辑 (不调用LLM), 仅验证架构和数据流")
    print(f"  提示: 设置 DEEPSEEK_API_KEY 环境变量可启用真实LLM调用")

    # ---- Step 1: 模拟 Target Scout ----
    print_header("Step 1: Target Scout — 靶点侦察")
    target_profile = {
        "target_name": "KRAS G12D",
        "structural_assessment": {
            "has_experimental_structure": True,
            "pdb_ids": ["6GOD", "4EPT", "6N2K"],
            "resolution_range": "1.5-2.2A",
            "has_cocrystal_with_ligand": True,
            "pocket_type": "cryptic",
            "pocket_volume_estimate": "medium (300-800 A3)",
            "pocket_polarity": "mixed",
            "flexibility_concern": "highly_flexible",
        },
        "known_ligand_info": {
            "has_known_active_ligands": True,
            "representative_ligands": ["ARS-1620", "MRTX849 (Adagrasib)", "AMG 510 (Sotorasib)"],
            "binding_affinity_range": "nM",
            "key_pharmacophore_features": ["共价弹头(丙烯酰胺)", "switch-II口袋占据", "疏水芳环"],
            "relevant_patents_or_papers": ["DOI:10.1038/s41586-019-1494-7"],
        },
        "priority_metrics": {
            "primary_metrics": ["共价弹头与G12D半胱氨酸的反应性", "switch-II口袋疏水互补面积", "对接能量(诱导契合后)"],
            "secondary_metrics": ["选择性vs KRAS WT", "细胞渗透性"],
            "red_flags": ["PAINS子结构", "非特异性共价弹头(如醛类)", "hERG抑制风险"],
            "suggested_thresholds": {"MW": "<800 (共价抑制剂可适当放宽)", "LogP": "2-6"},
        },
        "drug_design_challenges": [
            "switch-II口袋在apo状态下不可见, 需要诱导契合",
            "G12D突变引入的天冬氨酸改变了电荷分布",
            "GDP/GTP的皮摩尔级亲和力使得竞争极其困难",
            "KRAS表面极度光滑, 缺乏传统小分子结合口袋",
        ],
        "recommended_approaches": ["covalent", "SBDD", "FBDD"],
        "key_references": ["DOI:10.1038/s41586-019-1494-7", "DOI:10.1021/acs.jmedchem.0c00246"],
        "profile_timestamp": datetime.now().isoformat(),
    }
    print(f"  ✅ 靶点画像已生成")
    print(f"     口袋类型: {target_profile['structural_assessment']['pocket_type']}")
    print(f"     已知配体: {target_profile['known_ligand_info']['representative_ligands']}")
    print(f"     推荐路线: {target_profile['recommended_approaches']}")

    # ---- Step 2: 模拟 Strategy Generation ----
    print_header("Step 2: Strategy Generation — 多策略生成")
    from src.agents.strategy_generator import StrategyGeneratorAgent
    agent = StrategyGeneratorAgent()
    result = agent.generate_strategies(target_profile, KRAS_G12D_TARGET)
    strategies = result["strategies"]
    print(f"  ✅ {len(strategies)} 个差异化策略已生成")
    print(f"  💡 生成理由: {result.get('generation_rationale', '')[:200]}...")

    for i, s in enumerate(strategies, 1):
        print_strategy(s, i)

    # ---- Step 3/4: 模拟锦标赛 ----
    print_header("Step 3-4: 红军辩论锦标赛")

    # 初始化 Elo
    elo_ratings = {s["strategy_name"]: 1500.0 for s in strategies}
    pairings = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            pairings.append([strategies[i]["strategy_name"], strategies[j]["strategy_name"]])

    print(f"  🏟️  参赛策略: {len(strategies)} 个")
    print(f"  ⚔️  辩论场次: {len(pairings)} 场")
    print(f"  📊 初始 Elo: 全部 1500.0")

    tournament_history = []

    for round_num, (name_a, name_b) in enumerate(pairings, 1):
        sa = {s["strategy_name"]: s for s in strategies}[name_a]
        sb = {s["strategy_name"]: s for s in strategies}[name_b]

        # 红军评审
        from src.agents.expert_committee import RedTeamReviewer
        reviewer = RedTeamReviewer()
        debate_result = reviewer.debate_strategies(sa, sb, target_profile)

        # 裁判
        from src.agents.judge_agent import StrategyJudge
        judge = StrategyJudge()
        verdict = judge.judge_debate(
            sa, sb,
            debate_result["attacks_on_a"],
            debate_result["attacks_on_b"],
            target_profile,
        )

        # Elo
        winner = verdict.get("winner", "")
        if winner and winner != "tie":
            loser = name_b if winner == name_a else name_a
            from src.agents.judge_agent import StrategyJudge as SJ
            gain, loss = SJ.update_elo(elo_ratings, winner, loser, 32.0)
            elo_ratings[winner] += gain
            elo_ratings[loser] -= loss

        record = {
            "round_id": f"round_{round_num}",
            "strategy_a": name_a,
            "strategy_b": name_b,
            "expert_attacks_on_a": debate_result["attacks_on_a"],
            "expert_attacks_on_b": debate_result["attacks_on_b"],
            "judge_summary": verdict.get("judge_commentary", ""),
            "winner": winner,
            "key_deciding_factor": verdict.get("key_deciding_factor", ""),
            "timestamp": datetime.now().isoformat(),
        }
        tournament_history.append(record)
        print_debate(record)

    # ---- 排名 ----
    print_header("🏆 最终 Elo 排名")
    ranked = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, elo_score) in enumerate(ranked, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        s = {s["strategy_name"]: s for s in strategies}[name]
        print(f"  {medal} {rank}. {name}  Elo={elo_score:.0f}  [{s.get('approach_type','?')}]")
        print(f"      {s.get('strategy_tagline', '')}")

    # ---- MetaReview ----
    print_header("Step 5: MetaReview — 最佳策略进化")
    best_name = ranked[0][0]
    best_strategy = {s["strategy_name"]: s for s in strategies}[best_name]

    # 收集建议
    all_suggestions = []
    for debate in tournament_history:
        for atk_list in [debate["expert_attacks_on_a"], debate["expert_attacks_on_b"]]:
            for atk in atk_list:
                all_suggestions.extend(atk.get("suggested_fixes", []))
    unique_suggestions = list(dict.fromkeys(all_suggestions))[:10]

    print(f"  🏆 最佳策略: {best_name}")
    print(f"  📊 最终 Elo: {elo_ratings[best_name]:.0f}")
    print(f"  💊 进化建议 ({len(unique_suggestions)}条):")
    for sug in unique_suggestions:
        print(f"     - {sug}")

    print(f"\n{SEP}")
    print(f"  ✅ 锦标赛验证完成!")
    print(f"  📊 总结: {len(strategies)}策略 × {len(pairings)}场辩论 → 最佳策略: {best_name}")
    print(SEP)

    return {
        "strategies": strategies,
        "history": tournament_history,
        "elo": elo_ratings,
        "best": best_strategy,
    }


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    result = run_test()
