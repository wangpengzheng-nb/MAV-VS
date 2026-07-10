#!/usr/bin/env python3
"""AutoVS-Agent v2.0 — 任务隔离版。每次运行自动创建独立文件夹。"""
from __future__ import annotations
import hashlib, os, sys, re, glob as _glob
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YOUR_QUERY = "帮我筛选一个作用于BCL-2靶点的抗衰老小分子，要求不要与bcl-xl靶点相互作用，具有高选择性，同时具有抗衰老作用"
SKIP_TO_DEBATE = False
SKIP_RESEARCH = False

SEP = "=" * 70
def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")

def make_task_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(YOUR_QUERY.encode()).hexdigest()[:8]
    d = os.path.join(os.path.dirname(__file__), "分析文件", f"任务_{ts}_{h}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "query.txt"), "w") as f: f.write(YOUR_QUERY)
    return d

def find_latest_task():
    tasks = sorted(_glob.glob(os.path.join(os.path.dirname(__file__), "分析文件", "任务_*")))
    return tasks[-1] if tasks else None

def run_test():
    hdr(f"AutoVS-Agent v2.0 — LLM模式")
    print(f"  💬 查询: {YOUR_QUERY}")

    TASK_DIR = make_task_dir()
    print(f"  📁 任务: {os.path.basename(TASK_DIR)}")

    # Step 1: 调研
    hdr("Step 1: 深度调研")
    if SKIP_RESEARCH:
        prev = find_latest_task()
        rf = os.path.join(prev, "research_report.md") if prev else ""
        if rf and os.path.exists(rf):
            print(f"  ⏩ 跳过: {rf}")
            with open(rf, encoding="utf-8") as f: md = f.read()
            tn = re.search(r'靶点深度调研报告:\s*(.+?)\n', md)
            g = re.search(r'\*\*基因\*\*:\s*(.+?)\s*\|', md)
            u = re.search(r'\*\*UniProt\*\*:\s*(.+?)\n', md)
            tp = re.search(r'\*\*类型\*\*:\s*(.+?)\s*\|', md)
            o = re.search(r'\*\*物种\*\*:\s*(.+?)\n', md)
            bd = re.search(r'## 结合位点.*?\n(.+)', md, re.DOTALL)
            report = {"target_name": tn.group(1).strip() if tn else "?",
                      "gene_symbol": g.group(1).strip() if g else "",
                      "uniprot_id": u.group(1).strip() if u else "",
                      "target_macromolecule_type": tp.group(1).strip() if tp else "Protein",
                      "target_organism": o.group(1).strip() if o else "Homo sapiens",
                      "full_report_text": bd.group(1).strip()[:8000] if bd else md[:8000],
                      "biology_overview":"","structural_analysis":"","druggability_assessment":"",
                      "known_ligands_text":"","verified_pdb_structures":[],"chembl_activities":[],
                      "binding_site":{},"known_ligands":[],"references":[],"_search_log":["Skipped"]}
        else:
            from src.agents.target_scout import TargetScoutAgent
            report = TargetScoutAgent().deep_research(YOUR_QUERY)
            for e in report.get("_search_log",[]): print(f"  {e}")
    else:
        from src.agents.target_scout import TargetScoutAgent
        report = TargetScoutAgent().deep_research(YOUR_QUERY)
        for e in report.get("_search_log",[]): print(f"  {e}")
    print(f"  📋 {report.get('target_name','?')}")

    # Save report
    rf = os.path.join(TASK_DIR, "research_report.md")
    bs = report.get('binding_site',{})
    ft = report.get('full_report_text','')
    with open(rf, "w", encoding="utf-8") as f:
        f.write(f"# 靶点深度调研报告: {report.get('target_name','?')}\n\n")
        f.write(f"## 结合位点\n- 口袋: {bs.get('pocket_description','?')}\n\n")
        f.write(ft + "\n")
    print(f"  📁 {rf}")

    # Step 2: 策略
    hdr("Step 2: 策略生成")
    report["_user_query"] = YOUR_QUERY  # 🆕 传递用户原始任务要求
    from src.agents.strategy_generator import StrategyGeneratorAgent
    result = StrategyGeneratorAgent().generate_strategies(report)
    strategies = result["strategies"]
    print(f"  ✅ {len(strategies)} 策略")
    if not strategies: return

    strat_dir = os.path.join(TASK_DIR, "strategies")
    os.makedirs(strat_dir, exist_ok=True)
    for i, s in enumerate(strategies, 1):
        sf = os.path.join(strat_dir, f"strategy_{i:02d}_{s['strategy_name'].replace('/','_').replace(':','_')}.md")
        with open(sf, "w", encoding="utf-8") as f:
            f.write(f"# 策略 {i}: {s['strategy_name']}\n\n**标签**: {s.get('strategy_tagline','')} | **方法**: {s.get('approach_type','?')} | **耗时**: {s.get('estimated_runtime','?')}\n\n")
            f.write(f"## 原理\n{s.get('rationale','')}\n\n## 步骤\n")
            for st in s.get("pipeline_steps",[]):
                f.write(f"### Step {st.get('step_number','?')}: {st.get('step_name','?')}\n")
                f.write(f"| 工具 | 指标 | 阈值 |\n|---|---|---|\n| {st.get('tool','?')} | {st.get('metric','?')} | {st.get('threshold','?')} |\n\n")
                f.write(f"**操作**: {st.get('action','?')}\n\n**理由**: {st.get('rationale','?')}\n\n")
            f.write(f"## 存活: {s.get('survival_estimate','?')}\n\n## 应急: {s.get('contingency','?')}\n")
            f.write(f"## 优势\n"+"\n".join(f"- {x}" for x in s.get('strengths',[]))+"\n\n## 劣势\n"+"\n".join(f"- {x}" for x in s.get('weaknesses',[]))+"\n")
    print(f"  📁 {strat_dir}/")

    if len(strategies) < 2: return

    # Step 3: 辩论
    hdr("Step 3: 红军辩论")
    from src.agents.expert_committee import RedTeamReviewer
    from src.agents.judge_agent import StrategyJudge
    elo = {s["strategy_name"]: 1500.0 for s in strategies}
    pairings = [[strategies[i]["strategy_name"], strategies[j]["strategy_name"]]
                for i in range(len(strategies)) for j in range(i+1, len(strategies))]
    print(f"  🏟️  {len(strategies)}×{len(pairings)}场")
    debate_dir = os.path.join(TASK_DIR, "debates")
    os.makedirs(debate_dir, exist_ok=True)
    for rn, (na, nb) in enumerate(pairings, 1):
        sm = {s["strategy_name"]:s for s in strategies}
        d = RedTeamReviewer().debate_strategies(sm[na], sm[nb], report)
        v = StrategyJudge().judge_debate(sm[na], sm[nb], d["attacks_on_a"], d["attacks_on_b"], report)
        w = v.get("winner","")
        if w and w!="tie":
            loser = nb if w==na else na
            g,_=StrategyJudge.update_elo(elo,w,loser)
            elo[w]+=g;elo[loser]-=g
        print(f"  {rn}. {na[:25]} vs {nb[:25]} → {w or 'tie'}")
        df = os.path.join(debate_dir, f"debate_{rn:02d}.md")
        with open(df, "w", encoding="utf-8") as f:
            f.write(f"# {na} vs {nb}\n\n**胜者**: {w or 'tie'} | **A**:{v.get('strategy_a_score','?')} **B**:{v.get('strategy_b_score','?')}\n")
            f.write(f"**决定因素**: {v.get('key_deciding_factor','')}\n\n## 裁判点评\n{v.get('judge_commentary','')}\n\n")
            for label, attacks in [("策略A: "+na, d.get("attacks_on_a",[])), ("策略B: "+nb, d.get("attacks_on_b",[]))]:
                f.write(f"## {label}\n\n")
                for atk in attacks:
                    f.write(f"### {atk.get('persona_name','?')} ({atk.get('focus_area','?')})\n")
                    f.write(f"严重度: {atk.get('severity','?')} | 认可度: {atk.get('agreement',0):.0%}\n\n")
                    f.write("\n".join(f"- {p}" for p in atk.get('attack_points',[]))+"\n\n")
                    if atk.get('suggested_fixes'): f.write("**建议**:\n"+"\n".join(f"- {p}" for p in atk['suggested_fixes'])+"\n\n")
    print(f"  📁 {debate_dir}/")

    hdr("🏆 排名")
    for rank, (name, esc) in enumerate(sorted(elo.items(), key=lambda x: x[1], reverse=True), 1):
        m = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else "  "
        s = {s["strategy_name"]:s for s in strategies}[name]
        print(f"  {m} {rank}. {name}  Elo={esc:.0f}")
    print(f"\n{SEP}\n  ✅ {len(strategies)}策略×{len(pairings)}辩论\n{SEP}")

if __name__ == "__main__":
    run_test()
