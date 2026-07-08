#!/usr/bin/env python3
"""
AutoVS-Agent v2.0 完整锦标赛验证
=================================
用法: python test_tournament.py

流程:
  YOUR_QUERY → 深度调研报告(2000字) → 5-10个详细策略 → 红军三人设辩论 → Elo排名
"""
from __future__ import annotations
import os, sys
from datetime import datetime
from typing import Any, Dict

# 加载 .env 中的 DEEPSEEK_API_KEY
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ╔══════════════════════════════════════════════════════════════════╗
# ║            👇 只需改这一行! 👇                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
YOUR_QUERY = "我要做一款治类风湿关节炎的药，靶点是 JAK1"
# ╔══════════════════════════════════════════════════════════════════╝

SEP = "=" * 70

def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")

def run_test():
    use_llm = bool(os.getenv("DEEPSEEK_API_KEY"))
    hdr(f"AutoVS-Agent v2.0 — {'LLM模式' if use_llm else '降级模式(设DEEPSEEK_API_KEY启用LLM)'}")
    print(f"  💬 查询: {YOUR_QUERY}")

    # ═══════════ Step 1: 深度调研 ═══════════
    hdr("Step 1: 深度调研 (TargetScout)")
    from src.agents.target_scout import TargetScoutAgent
    scout = TargetScoutAgent()
    report = scout.deep_research(YOUR_QUERY)
    for entry in report.get("_search_log", []):
        print(f"  {entry}")

    # ── 打印完整报告 ──
    print(f"\n  {'─'*60}")
    print(f"  📋 靶点深度调研报告")
    print(f"  {'─'*60}")
    print(f"  靶点: {report.get('target_name','?')} | 基因: {report.get('gene_name','?')} | UniProt: {report.get('uniprot_id','?')}")
    print(f"  类型: {report.get('target_class','?')} | 物种: {report.get('organism','?')}")
    bs = report.get('binding_site', {})
    print(f"  口袋: {bs.get('pocket_description','?')}")
    print(f"  体积: {bs.get('volume_angstrom3','?')} | 极性: {bs.get('polarity','?')} | 柔性: {bs.get('flexibility','?')}")
    print(f"  对接盒子: center={bs.get('center_coordinates',[])}, size={bs.get('suggested_box_size',[])}")
    residues = [f"{r.get('name','?')}({r.get('role','?')})" for r in bs.get('key_residues',[])]
    print(f"  关键残基: {', '.join(residues) if residues else '无'}")

    ligands = report.get('known_ligands', [])
    if ligands:
        print(f"\n  已知配体 ({len(ligands)}个):")
        for l in ligands[:8]:
            print(f"    • {l.get('name','?')}: {l.get('activity_type','?')}={l.get('activity_value','?')} [{l.get('mechanism','?')}]")

    refs = report.get('references', [])
    if refs:
        print(f"\n  参考文献:")
        for r in refs[:8]:
            print(f"    • {r}")

    # ── 保存调研报告 md ──
    out_dir = os.path.join(os.path.dirname(__file__), "分析文件")
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# 靶点深度调研报告: {report.get('target_name','?')}\n\n")
        f.write(f"**基因**: {report.get('gene_name','?')} | **UniProt**: {report.get('uniprot_id','?')}\n")
        f.write(f"**类型**: {report.get('target_class','?')} | **物种**: {report.get('organism','?')}\n\n")
        f.write(f"## 结合位点\n- 口袋: {bs.get('pocket_description','?')}\n")
        f.write(f"- 对接盒子: center={bs.get('center_coordinates',[])}, size={bs.get('suggested_box_size',[])}\n\n")
        if ligands:
            f.write("## 已知配体\n")
            for l in ligands: f.write(f"- {l.get('name','?')}: {l.get('activity_type','?')}={l.get('activity_value','?')}\n")
            f.write("\n")
        full_text = report.get('full_report_text', '')
        f.write(full_text)
        f.write("\n\n## 参考文献\n")
        for r in refs: f.write(f"- {r}\n")
    print(f"  📁 调研报告已保存: {report_file}")

    # ═══════════ Step 2: 策略生成 ═══════════
    hdr("Step 2: 详细策略生成 (StrategyGenerator)")
    from src.agents.strategy_generator import StrategyGeneratorAgent
    gen = StrategyGeneratorAgent()
    result = gen.generate_strategies(report)
    strategies = result["strategies"]
    print(f"  ✅ {len(strategies)} 个策略: {result.get('generation_rationale','')[:150]}")
    for i, s in enumerate(strategies, 1):
        steps = s.get("pipeline_steps", [])
        print(f"\n  📋 {i}. {s['strategy_name']} [{s['approach_type']}]")
        print(f"     🏷️  {s['strategy_tagline']}")
        print(f"     💡 {s['rationale'][:120]}...")
        print(f"     📊 {s.get('survival_estimate','?')} | ⏱️ {s.get('estimated_runtime','?')}")
        print(f"     📝 步骤 ({len(steps)}步):")
        for st in steps:
            print(f"        {st.get('step_number','?')}. {st.get('step_name','?')} [{st.get('tool','?')}]")
            print(f"           → {st.get('metric','?')}: {st.get('threshold','?')}")
        print(f"     🟢 应急: {s.get('contingency','')[:80]}...")
        print(f"     ✅ {', '.join(s.get('strengths',[])[:2])}")
        print(f"     ⚠️  {', '.join(s.get('weaknesses',[])[:2])}")

    # ── 每个策略保存为独立 md ──
    strat_dir = os.path.join(os.path.dirname(__file__), "分析文件", "strategies")
    os.makedirs(strat_dir, exist_ok=True)
    for i, s in enumerate(strategies, 1):
        safe_name = s['strategy_name'].replace('/','_').replace('\\','_').replace(':','_')
        sf = os.path.join(strat_dir, f"strategy_{i:02d}_{safe_name}.md")
        with open(sf, "w", encoding="utf-8") as f:
            f.write(f"# 策略 {i}: {s['strategy_name']}\n\n")
            f.write(f"**标签**: {s.get('strategy_tagline','')}\n\n")
            f.write(f"**方法**: {s.get('approach_type','?')} | **估算耗时**: {s.get('estimated_runtime','?')}\n\n")
            f.write(f"## 原理\n{s.get('rationale','')}\n\n")
            f.write(f"## 步骤\n")
            for st in s.get("pipeline_steps", []):
                f.write(f"### Step {st.get('step_number','?')}: {st.get('step_name','?')}\n")
                f.write(f"- **工具**: {st.get('tool','?')}\n")
                f.write(f"- **操作**: {st.get('action','?')}\n")
                f.write(f"- **指标**: {st.get('metric','?')}\n")
                f.write(f"- **阈值**: {st.get('threshold','?')}\n")
                f.write(f"- **理由**: {st.get('rationale','?')}\n\n")
            f.write(f"## 存活估算\n{s.get('survival_estimate','?')}\n\n")
            f.write(f"## 应急预案\n{s.get('contingency','?')}\n\n")
            f.write(f"## 优势\n" + "\n".join(f"- {x}" for x in s.get('strengths',[])) + "\n\n")
            f.write(f"## 劣势\n" + "\n".join(f"- {x}" for x in s.get('weaknesses',[])) + "\n\n")
            f.write(f"## 适用场景\n{s.get('suitable_when','?')}\n")
    print(f"  📁 {len(strategies)}个策略已保存: {strat_dir}/")

    # ═══════════ Step 3: 红军辩论 ═══════════
    hdr("Step 3: 红军三人设辩论")
    from src.agents.expert_committee import RedTeamReviewer
    from src.agents.judge_agent import StrategyJudge

    elo = {s["strategy_name"]: 1500.0 for s in strategies}
    pairings = [[strategies[i]["strategy_name"], strategies[j]["strategy_name"]]
                for i in range(len(strategies)) for j in range(i+1, len(strategies))]
    print(f"  🏟️  {len(strategies)}策略 × {len(pairings)}场 = 每场3位专家评审")

    reviewer = RedTeamReviewer()
    judge = StrategyJudge()

    for rn, (na, nb) in enumerate(pairings[:min(len(pairings), 10)], 1):
        sm = {s["strategy_name"]: s for s in strategies}
        sa, sb = sm[na], sm[nb]
        debate = reviewer.debate_strategies(sa, sb, report)
        verdict = judge.judge_debate(sa, sb, debate["attacks_on_a"], debate["attacks_on_b"], report)

        winner = verdict.get("winner", "")
        if winner and winner != "tie":
            loser = nb if winner == na else na
            g, _ = StrategyJudge.update_elo(elo, winner, loser, 32.0)
            elo[winner] += g; elo[loser] -= g

        print(f"\n  ⚔️  Round {rn}: {na[:30]} vs {nb[:30]}")
        for atk in debate["attacks_on_a"][:1]:
            pts = atk.get("attack_points", [])
            ref = atk.get("reference_to_report", "")
            print(f"     [{atk['persona_name']}] {atk['focus_area']} — 认可度={atk['agreement']:.0%}")
            if pts: print(f"        💢 {pts[0][:100]}")
            if ref: print(f"        📎 引用: {ref[:80]}")
        print(f"     ⚖️  裁判: {verdict.get('judge_commentary','?')[:150]}")
        print(f"     🏆 {winner or 'tie'} (A={verdict.get('strategy_a_score','?')}, B={verdict.get('strategy_b_score','?')})")

    # ═══════════ 排名 ═══════════
    hdr("🏆 最终排名")
    for rank, (name, esc) in enumerate(sorted(elo.items(), key=lambda x: x[1], reverse=True), 1):
        m = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else "  "
        s = {s["strategy_name"]: s for s in strategies}[name]
        print(f"  {m} {rank}. {name}  Elo={esc:.0f}  {s.get('strategy_tagline','')[:60]}")
    print(f"\n{SEP}\n  ✅ 完成! {len(strategies)}策略 × {len(pairings)}辩论 → 🏆 {max(elo.items(),key=lambda x:x[1])[0]}\n{SEP}")

if __name__ == "__main__":
    run_test()
