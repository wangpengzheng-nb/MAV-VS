#!/usr/bin/env python3
"""AutoVS-Agent v3.0 — 策略排名锦标赛 (瑞士制版)"""
from __future__ import annotations
import hashlib, os, sys, re, glob as _glob, json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YOUR_QUERY = "基于靶点bcl-2去筛选一个抗衰老药物，要求其具有高选择性，不能作用于bcl-xl，且具有良好的ADMET性质。"
PRIOR_KNOWLEDGE = ""
# 先验知识示例 (取消注释即可启用):
# PRIOR_KNOWLEDGE = """
# 工具选择规则:
# - PPI类型口袋 → 使用 Diffdock 进行对接
# - 其他类型口袋 → 使用 gnina 进行对接
# - 二分类结合预测 → 使用 Boltz-2
# """
SKIP_RESEARCH = False  # 跳过Step0, 从已有文件加载
SKIP_STRATEGY = False  # 跳过Step0-1, 从已有文件加载
LOAD_STRATEGIES_DIR = "/users_home/wangpengzheng/药物筛选智能体/分析文件/任务_20260710_164401_59b4bdbb/strategies"
RESEARCH_REPORT_DIR = "/users_home/wangpengzheng/药物筛选智能体/分析文件/任务_20260710_164401_59b4bdbb"
SWISS_ROUNDS = 4
SKIP_EVALUATION = False   # 跳过Step2-3, 从已有文件加载
LOAD_FROM_DIR = "/users_home/wangpengzheng/药物筛选智能体/分析文件/任务_20260710_164401_59b4bdbb"  # 已有任务目录
EVOLVE_TOP_N = 3         # Step 4: 进化前N名, 0=跳过进化

SEP = "=" * 70
def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")

def make_task_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(YOUR_QUERY.encode()).hexdigest()[:8]
    d = os.path.join(os.path.dirname(__file__), "分析文件", f"任务_{ts}_{h}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "query.txt"), "w") as f: f.write(YOUR_QUERY)
    return d

# =========================================================================
# 策略加载
# =========================================================================

def load_strategies_from_dir(strat_dir: str) -> list:
    strategies = []
    for fname in sorted(os.listdir(strat_dir)):
        if not fname.endswith(".md"): continue
        with open(os.path.join(strat_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        s = _parse_strategy_md(text)
        if s: strategies.append(s)
    return strategies

def _parse_strategy_md(text: str) -> dict:
    s = {"strategy_name":"","strategy_tagline":"","approach_type":"","rationale":"",
         "pipeline_steps":[],"survival_estimate":"","contingency":"",
         "strengths":[],"weaknesses":[],"estimated_runtime":"","suitable_when":""}
    m = re.search(r'^#\s*策略\s*\d+\s*:\s*(.+)$', text, re.MULTILINE)
    if m: s["strategy_name"] = m.group(1).strip()
    m2 = re.search(r'\*\*标签\*\*:\s*(.+?)\s*\|\s*\*\*方法\*\*:\s*(.+?)\s*\|\s*\*\*耗时\*\*:\s*(.+)', text, re.MULTILINE)
    if m2:
        s["strategy_tagline"] = m2.group(1).strip()
        s["approach_type"] = m2.group(2).strip()
        s["estimated_runtime"] = m2.group(3).strip()
    m = re.search(r'##\s*原理\n(.+?)(?:\n##\s|\Z)', text, re.DOTALL)
    if m: s["rationale"] = m.group(1).strip()
    for sn, sname, tool, metric, threshold, action, rational in re.findall(
        r'###\s*Step\s*(\d+):\s*(.+?)\n\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\n\n\*\*操作\*\*:\s*(.+?)\n\n\*\*理由\*\*:\s*(.+)',
        text, re.DOTALL):
        s["pipeline_steps"].append({"step_number":int(sn),"step_name":sname.strip(),
            "tool":tool.strip(),"action":action.strip(),"metric":metric.strip(),
            "threshold":threshold.strip(),"rationale":rational.strip()})
    m = re.search(r'(?:##\s*)?\*\*存活\*\*:\s*(.+?)$', text, re.MULTILINE)
    if m: s["survival_estimate"] = m.group(1).strip()
    m = re.search(r'(?:##\s*)?\*\*应急\*\*:\s*(.+?)$', text, re.MULTILINE)
    if m: s["contingency"] = m.group(1).strip()
    in_adv = in_disadv = False
    for line in text.split("\n"):
        if line.startswith("## 优势"): in_adv, in_disadv = True, False; continue
        if line.startswith("## 劣势"): in_adv, in_disadv = False, True; continue
        if line.startswith("## ") and "优势" not in line and "劣势" not in line: in_adv = in_disadv = False; continue
        if in_adv and line.startswith("- "): s["strengths"].append(line[2:].strip())
        if in_disadv and line.startswith("- "): s["weaknesses"].append(line[2:].strip())
    return s if s["strategy_name"] else None

def load_research_report(report_dir: str) -> dict:
    rf = os.path.join(report_dir, "research_report.md")
    if not os.path.exists(rf): return {}
    with open(rf, "r", encoding="utf-8") as f: full_text = f.read()
    report = {"full_report_text": full_text}
    for key, pattern in [
        ("target_name", r'靶点深度调研报告:\s*(.+?)\n'),
        ("gene_symbol", r'\*\*基因\*\*:\s*(.+?)\s*\|'),
        ("uniprot_id", r'\*\*UniProt\*\*:\s*(.+?)\n'),
        ("target_organism", r'\*\*物种\*\*:\s*(.+?)\n'),
    ]:
        m = re.search(pattern, full_text)
        if m: report[key] = m.group(1).strip()
    return report

# =========================================================================
# 瑞士制配对
# =========================================================================

def swiss_pairings(strategies: list, elo: dict, history: set) -> list:
    """按当前Elo降序, 相邻配对, 跳过已打过的配对。"""
    ranked = sorted(strategies, key=lambda s: elo.get(s["strategy_name"],1500), reverse=True)
    names = [s["strategy_name"] for s in ranked]
    pairings = []
    for i in range(0, len(names), 2):
        if i+1 >= len(names): break
        a, b = names[i], names[i+1]
        pair_key = tuple(sorted([a, b]))
        if pair_key in history:
            swapped = False
            for j in range(i+2, len(names)):
                alt_key = tuple(sorted([a, names[j]]))
                if alt_key not in history:
                    pairings.append((a, names[j])); history.add(alt_key); swapped = True; break
            if not swapped:
                pairings.append((a, b)); history.add(pair_key)
        else:
            pairings.append((a, b)); history.add(pair_key)
    return pairings

# =========================================================================
# 主流程
# =========================================================================

def run_test():
    hdr("AutoVS-Agent v3.0 — 策略排名锦标赛 (瑞士制)")
    TASK_DIR = make_task_dir()
    print(f"  📁 任务: {os.path.basename(TASK_DIR)}")
    print(f"  💬 查询: {YOUR_QUERY}")

    # ── Step 0: 调研报告 ──
    report = {}
    if SKIP_RESEARCH and RESEARCH_REPORT_DIR:
        report = load_research_report(RESEARCH_REPORT_DIR)
        print(f"  ⏩ 跳过调研, 加载: {report.get('target_name','?')} ({report.get('gene_symbol','?')})")
    else:
        hdr("Step 0: 深度调研")
        from src.agents.target_scout import TargetScoutAgent
        report = TargetScoutAgent().deep_research(YOUR_QUERY)

    # 保存调研报告
    rf = os.path.join(TASK_DIR, "research_report.md")
    bs = report.get('binding_site', {})
    summary = report.get('executive_summary', '')
    with open(rf, "w", encoding="utf-8") as f:
        f.write(f"# 靶点深度调研报告: {report.get('target_name','?')}\n\n")
        f.write(f"## 靶点信息\n")
        f.write(f"**基因**: {report.get('gene_symbol','?')} | "
                f"**UniProt**: {report.get('uniprot_id','?')} | "
                f"**类型**: {report.get('target_macromolecule_type','Protein')} | "
                f"**物种**: {report.get('target_organism','?')}\n\n")
        f.write(f"## 结合位点\n- 口袋: {bs.get('pocket_description','?')}\n\n")
        if summary:
            f.write(f"## 执行摘要\n{summary}\n\n---\n\n## 完整报告\n")
        f.write(report.get('full_report_text', '') + "\n")
    print(f"  📁 {rf}")

    # ── Step 1: 策略加载 ──
    if SKIP_STRATEGY and LOAD_STRATEGIES_DIR:
        hdr("Step 1: 加载策略")
        strategies = load_strategies_from_dir(LOAD_STRATEGIES_DIR)
        print(f"  ⏩ 跳过生成, 加载 {len(strategies)} 个策略")
    else:
        hdr("Step 1: 策略生成")
        report["_user_query"] = YOUR_QUERY
        from src.agents.strategy_generator import StrategyGeneratorAgent
        result = StrategyGeneratorAgent().generate_strategies(report, prior_knowledge=PRIOR_KNOWLEDGE)
        strategies = result["strategies"]
        print(f"  ✅ {len(strategies)} 策略")

        # 保存策略
        strat_dir = os.path.join(TASK_DIR, "strategies")
        os.makedirs(strat_dir, exist_ok=True)
        for i, s in enumerate(strategies, 1):
            sf = os.path.join(strat_dir,
                f"strategy_{i:02d}_{s['strategy_name'].replace('/','_').replace(':','_')[:60]}.md")
            with open(sf, "w", encoding="utf-8") as f:
                f.write(f"# 策略 {i}: {s['strategy_name']}\n\n")
                f.write(f"**标签**: {s.get('strategy_tagline','')} | "
                        f"**方法**: {s.get('approach_type','?')} | "
                        f"**耗时**: {s.get('estimated_runtime','?')}\n\n")
                f.write(f"## 原理\n{s.get('rationale','')}\n\n## 步骤\n")
                for st in s.get("pipeline_steps", []):
                    f.write(f"### Step {st.get('step_number','?')}: {st.get('step_name','?')}\n")
                    f.write(f"| 工具 | 指标 | 阈值 |\n|---|---|---|\n"
                            f"| {st.get('tool','?')} | {st.get('metric','?')} | {st.get('threshold','?')} |\n\n")
                    f.write(f"**操作**: {st.get('action','?')}\n\n**理由**: {st.get('rationale','?')}\n\n")
                f.write(f"**存活**: {s.get('survival_estimate','?')}\n\n")
                f.write(f"**应急**: {s.get('contingency','?')}\n\n")
                f.write(f"## 优势\n" + "\n".join(f"- {x}" for x in s.get('strengths',[])) + "\n\n")
                f.write(f"## 劣势\n" + "\n".join(f"- {x}" for x in s.get('weaknesses',[])) + "\n")
        print(f"  📁 {strat_dir}/")

    if not strategies:
        print("  ❌ 无策略可评估"); return
    for i, s in enumerate(strategies, 1):
        print(f"  {i}. {s['strategy_name'][:65]}")
    if len(strategies) < 2:
        print("  ⚠️ 需要至少2个策略"); return

    # ── Step 2: 策略审评 (全排列 pairwise 投票) ──
    hdr("Step 2: 策略审评 (全排列 pairwise 投票)")
    from src.agents.expert_committee import TournamentReviewer, REVIEWER_CONFIGS
    from src.agents.judge_agent import VoteAggregator
    import itertools

    reviewer = TournamentReviewer()
    aggregator = VoteAggregator()

    # 全排列配对
    n = len(strategies)
    pairs = list(itertools.combinations(range(n), 2))
    total_matches = len(pairs) * len(REVIEWER_CONFIGS)
    print(f"  🏟️  {n}策略 × C({n},2)={len(pairs)}对 × {len(REVIEWER_CONFIGS)}评审官 = {total_matches}次投票")
    print(f"  📊 并发控制: 最大10并发 + 自动重试3次")

    # 同步执行(简化版, 不用asyncio): 逐对评审, 每对的3个评审官并行
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tenacity import retry, stop_after_attempt, wait_exponential

    def review_one_pair_with_retry(rid, mid, sa, sb):
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
        def _call():
            return reviewer.compare_strategies(sa, sb, report, YOUR_QUERY,
                                               PRIOR_KNOWLEDGE, reviewer_id=rid, match_id=mid)
        try:
            return _call()
        except Exception as e:
            print(f"    ❌ [{rid}] 重试3次后仍失败: {e}", flush=True)
            return {"reviewer_id": rid, "match_id": mid, "overall_verdict": "tie",
                    "verdict_confidence": "low", "dimension_votes": [],
                    "critical_concerns": {}, "suggestions": {}, "decision_logic": f"失败: {e}"}

    for pi, (i, j) in enumerate(pairs):
        sa, sb = strategies[i], strategies[j]
        na, nb = sa["strategy_name"], sb["strategy_name"]
        mid = f"{na[:20]}_vs_{nb[:20]}"
        print(f"\n  [{pi+1}/{len(pairs)}] {na[:30]} vs {nb[:30]}", end="", flush=True)

        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(review_one_pair_with_retry, cfg["id"], mid, sa, sb): cfg
                    for cfg in REVIEWER_CONFIGS}
            for fut in as_completed(futs):
                result = fut.result()
                aggregator.add_result(result)
                vid = result.get("reviewer_id", "?")
                vv = result.get("overall_verdict", "tie")
                conf = result.get("verdict_confidence", "?")
                print(f"\n    [{vid}] → {vv} (conf={conf})", end="", flush=True)

    # ── Step 3: 统票 + 排名 + 诊断 ──
    hdr("Step 3: 统票 + 排名 + 诊断报告")
    ranking = aggregator.rank(strategies)
    diagnostics = aggregator.generate_diagnostic(top_n=3)

    print(f"\n  🏆 最终排名:")
    for r in ranking:
        emoji = "🥇" if r['rank']==1 else "🥈" if r['rank']==2 else "🥉" if r['rank']==3 else "  "
        print(f"  {emoji} {r['rank']}. {r['strategy_name'][:55]}")
        print(f"     得票: {r['total_votes']:.1f} | 胜{r['wins']}负{r['losses']}平{r['draws']} | "
              f"Fatal:{r['fatal_count']} High:{r['high_suggestions']}")

    print(f"\n  📊 Top 3 诊断报告(输入进化智能体):")
    for d in diagnostics:
        concerns = len(d['aggregated_concerns'])
        suggs = len(d['aggregated_suggestions'])
        print(f"\n  [{d['strategy_name'][:50]}]")
        print(f"    致命/警告: {concerns}条  |  建议: {suggs}条")
        if d['dimension_strengths']:
            print(f"    优势维度: {', '.join(d['dimension_strengths'][:3])}")
        if d['dimension_weaknesses']:
            print(f"    劣势维度: {', '.join(d['dimension_weaknesses'][:3])}")

    ranked_names = [(r['strategy_name'], r['total_votes']) for r in ranking]

    # ── Step 4: 策略进化 (基于诊断报告) ──
    if EVOLVE_TOP_N > 0 and len(strategies) >= 3 and diagnostics:
        hdr("Step 4: 策略进化 (基于诊断报告)")
        from src.agents.strategy_evolver import StrategyEvolver

        evolver = StrategyEvolver()
        # 用诊断报告替代旧的 review_results
        diagnostic_map = {d["strategy_name"]: d for d in diagnostics}
        evolved_strategies = evolver.evolve_top_n(
            strategies, diagnostic_map, [],
            report, YOUR_QUERY, n=min(EVOLVE_TOP_N, len(diagnostics)),
            prior_knowledge=PRIOR_KNOWLEDGE)

        evo_dir = os.path.join(TASK_DIR, "evolved_strategies")
        os.makedirs(evo_dir, exist_ok=True)
        for i, s in enumerate(evolved_strategies, 1):
            if "(v2" in s.get("strategy_name", ""):
                sf = os.path.join(evo_dir, f"evolved_{i:02d}_{s['strategy_name'][:50]}.md")
                with open(sf, "w", encoding="utf-8") as f:
                    f.write(f"# 进化策略 {i}: {s['strategy_name']}\n\n")
                    for c in s.get('evolution_changelog', []): f.write(f"- {c}\n")
                    f.write(f"\n## 原理\n{s.get('rationale','')}\n")
        print(f"  📁 {evo_dir}/")
    else:
        evolved_strategies = strategies

    # 保存评审结果
    review_dir = os.path.join(TASK_DIR, "reviews")
    os.makedirs(review_dir, exist_ok=True)
    with open(os.path.join(review_dir, "all_votes.json"), "w", encoding="utf-8") as f:
        json.dump({"results": aggregator.results, "ranking": ranking,
                   "diagnostics": diagnostics}, f, ensure_ascii=False, indent=2)

    # 保存结果 (v4: 投票+诊断)
    result_file = os.path.join(TASK_DIR, "tournament_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"query": YOUR_QUERY, "ranking": ranking,
                   "diagnostics": diagnostics,
                   "total_votes": len(aggregator.results),
                   }, f, ensure_ascii=False, indent=2)
    print(f"\n  Result: {result_file}  |  Reviews: {review_dir}/")
    print(f"\n{SEP}\n  Done: {len(strategies)}x{len(pairs)}pairs x3={total_matches}votes\n{SEP}")

if __name__ == "__main__":
    run_test()
