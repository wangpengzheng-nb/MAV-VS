#!/usr/bin/env python3
"""AutoVS-Agent v3.0 — 策略排名锦标赛 (瑞士制版)"""
from __future__ import annotations
import hashlib, os, sys, re, glob as _glob, json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YOUR_QUERY = "基于靶点bcl-2去筛选一个抗衰老药物，要求其具有高选择性，不能作用于bcl-xl，且具有良好的ADMET性质。"
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
    with open(rf, "w", encoding="utf-8") as f:
        f.write(f"# 靶点深度调研报告: {report.get('target_name','?')}\n\n")
        f.write(f"## 靶点信息\n")
        f.write(f"**基因**: {report.get('gene_symbol','?')} | "
                f"**UniProt**: {report.get('uniprot_id','?')} | "
                f"**类型**: {report.get('target_macromolecule_type','Protein')} | "
                f"**物种**: {report.get('target_organism','?')}\n\n")
        f.write(f"## 结合位点\n- 口袋: {bs.get('pocket_description','?')}\n\n")
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
        result = StrategyGeneratorAgent().generate_strategies(report)
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

    # ── Step 2-3: 评审+锦标赛 ──
    from src.agents.expert_committee import TournamentReviewer
    from src.agents.judge_agent import StrategyJudge

    reviewer = TournamentReviewer()
    review_results = {}
    elo = {}
    ranked = []
    tournament_records = []

    if SKIP_EVALUATION and LOAD_FROM_DIR:
        hdr("Step 2-3: 加载已有评审和锦标赛结果")
        review_dir = os.path.join(LOAD_FROM_DIR, "reviews")
        if os.path.isdir(review_dir):
            for fname in os.listdir(review_dir):
                with open(os.path.join(review_dir, fname), "r", encoding="utf-8") as f:
                    r = json.load(f)
                review_results[r["strategy_name"]] = r
            print(f"  ⏩ 加载 {len(review_results)} 个评审")
        tr_file = os.path.join(LOAD_FROM_DIR, "tournament_results.json")
        total_matches = 0
        if os.path.exists(tr_file):
            with open(tr_file, "r", encoding="utf-8") as f:
                tr = json.load(f)
            elo = tr.get("elo_final", {})
            ranked = tr.get("ranking", [])
            tournament_records = tr.get("records", [])
            total_matches = tr.get("total_matches", len(tournament_records))
            for name, score in tr.get("independent_scores", {}).items():
                if name not in review_results:
                    review_results[name] = {"weighted_score": score, "reports": [], "all_critical_flaws": []}
            print(f"  ⏩ 加载锦标赛: {len(ranked)}排名, {len(tournament_records)}场判例")
        print(f"\n  📊 已有排名:")
        for rank, (name, esc) in enumerate(ranked, 1):
            emoji = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else "  "
            init_score = review_results.get(name, {}).get("weighted_score", "?")
            print(f"  {emoji} {rank}. {name[:55]}  评分={init_score:.1f} | Elo={esc:.0f}")
    else:
        hdr("Step 2: 三人设独立评审")
        for i, s in enumerate(strategies, 1):
            print(f"\n  [{i}/{len(strategies)}]", end="", flush=True)
            rr = reviewer.review_strategy(s, report, YOUR_QUERY)
            review_results[s["strategy_name"]] = rr
        review_results = reviewer.calibrate_all(review_results, strategies, report, YOUR_QUERY)
        print(f"\n\n  📊 独立评分排名(校准后):")
        sorted_by_score = sorted(review_results.items(), key=lambda x: x[1]["weighted_score"], reverse=True)
        for rank, (name, rr) in enumerate(sorted_by_score, 1):
            emoji = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else "  "
            flaws = len(rr.get("all_critical_flaws",[]))
            print(f"  {emoji} {rank}. {name[:55]}  {rr['weighted_score']:.1f}分"
                  f"{' ⚠️'+str(flaws)+'致命缺陷' if flaws else ''}")
        scores = [rr["weighted_score"] for rr in review_results.values()]
        score_range = max(scores) - min(scores)
        mean_score = sum(scores) / len(scores)
        std_dev = math.sqrt(sum((s-mean_score)**2 for s in scores)/len(scores))
        print(f"\n  📊 分数统计: 范围={score_range:.1f} | 均值={mean_score:.1f} | 标准差={std_dev:.1f}")

        hdr("Step 3: 瑞士制锦标赛")
        for name, rr in review_results.items():
            elo[name] = StrategyJudge.compute_initial_elo(rr)
        init_elo = dict(elo)
        total_matches = 0
        judge = StrategyJudge()
        history = set()
        for rn in range(1, SWISS_ROUNDS + 1):
            pairings = swiss_pairings(strategies, elo, history)
            if not pairings: break
            print(f"\n  ── 第{rn}轮 ({len(pairings)}场) ──")
            for na, nb in pairings:
                sa = next(s for s in strategies if s["strategy_name"] == na)
                sb = next(s for s in strategies if s["strategy_name"] == nb)
                ra, rb = review_results.get(na, {}), review_results.get(nb, {})
                verdict = judge.judge_match(sa, sb, ra, rb, report, YOUR_QUERY)
                winner = verdict.get("winner", ""); total_matches += 1
                if winner and winner != "tie":
                    loser = nb if winner == na else na
                    _, _, elo = StrategyJudge.update_elo(elo, winner, loser, False)
                elif winner == "tie":
                    _, _, elo = StrategyJudge.update_elo(elo, na, nb, True)
                print(f"    {na[:28]} vs {nb[:28]} → "
                      f"{'tie' if winner=='tie' else winner[:40]} | "
                      f"conf={verdict.get('confidence',.5):.0%}")
                tournament_records.append({"round": rn, "strategy_a": na,
                    "strategy_b": nb, "verdict": verdict})
            max_shift = max(abs(elo[n]-init_elo.get(n,elo[n])) for n in elo) if init_elo else 0
            if rn >= 3 and max_shift < 5: break

        hdr("🏆 最终排名")
        ranked = sorted(elo.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, esc) in enumerate(ranked, 1):
        emoji = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else "  "
        init_score = review_results.get(name, {}).get("weighted_score", "?")
        try:
            change = esc - init_elo[name]
        except (NameError, KeyError):
            change = 0
        print(f"  {emoji} {rank}. {name[:55]}")
        print(f"     独立评分={init_score:.1f} | Elo={esc:.0f} "
              f"({'↑'+str(int(change)) if change>0 else '↓'+str(int(-change)) if change<0 else '—'})")

    # ── Step 4: 策略进化 + 迷你锦标赛验证 ──
    if EVOLVE_TOP_N > 0 and len(strategies) >= 3:
        hdr("Step 4: 策略进化 + 迷你锦标赛验证")
        from src.agents.strategy_evolver import StrategyEvolver

        evolver = StrategyEvolver()
        evolved_strategies = evolver.evolve_top_n(
            strategies, review_results, tournament_records,
            report, YOUR_QUERY, n=EVOLVE_TOP_N)

        evo_dir = os.path.join(TASK_DIR, "evolved_strategies")
        os.makedirs(evo_dir, exist_ok=True)
        for i, s in enumerate(evolved_strategies, 1):
            if "(v2" in s.get("strategy_name", ""):
                sf = os.path.join(evo_dir,
                    f"evolved_{i:02d}_{s['strategy_name'].replace('/','_').replace(':','_')[:60]}.md")
                with open(sf, "w", encoding="utf-8") as f:
                    f.write(f"# 进化策略 {i}: {s['strategy_name']}\n\n"
                            f"**标签**: {s.get('strategy_tagline','')} | "
                            f"**方法**: {s.get('approach_type','?')} | "
                            f"**耗时**: {s.get('estimated_runtime','?')}\n\n")
                    changelog = s.get('evolution_changelog', [])
                    if changelog:
                        f.write(f"## 进化日志\n" + "\n".join(f"- {c}" for c in changelog) + "\n\n")
                    f.write(f"## 原理\n{s.get('rationale','')}\n\n## 步骤\n")
                    for st in s.get("pipeline_steps", []):
                        f.write(f"### Step {st.get('step_number','?')}: {st.get('step_name','?')}\n"
                                f"| 工具 | 指标 | 阈值 |\n|---|---|---|\n"
                                f"| {st.get('tool','?')} | {st.get('metric','?')} | {st.get('threshold','?')} |\n\n"
                                f"**操作**: {st.get('action','?')}\n\n**理由**: {st.get('rationale','?')}\n\n")
                    f.write(f"**存活**: {s.get('survival_estimate','?')}\n\n"
                            f"**应急**: {s.get('contingency','?')}\n\n"
                            f"## 优势\n" + "\n".join(f"- {x}" for x in s.get('strengths',[])) + "\n\n"
                            f"## 劣势\n" + "\n".join(f"- {x}" for x in s.get('weaknesses',[])) + "\n")
        print(f"  📁 {evo_dir}/")

        evo_names = [s["strategy_name"] for s in evolved_strategies
                     if "(v2" in s.get("strategy_name", "")]
        if evo_names and len(evo_names) >= 1:
            orig_top = [ranked[i][0] for i in range(min(EVOLVE_TOP_N, len(ranked)))]
            orig_strategies = [s for s in strategies if s["strategy_name"] in orig_top]
            evo_strategies_only = [s for s in evolved_strategies if s["strategy_name"] in evo_names]
            mini_strategies = orig_strategies + evo_strategies_only
            print(f"\n  🏟️  迷你锦标赛: {len(mini_strategies)}策略 "
                  f"({len(orig_strategies)}原始 + {len(evo_strategies_only)}进化)", flush=True)

            judge = StrategyJudge()
            mini_reviewer = TournamentReviewer()
            mini_reviews = {}
            for i, s in enumerate(mini_strategies, 1):
                print(f"  [{i}/{len(mini_strategies)}]", end="", flush=True)
                rr = mini_reviewer.review_strategy(s, report, YOUR_QUERY)
                mini_reviews[s["strategy_name"]] = rr
            mini_reviews = mini_reviewer.calibrate_all(mini_reviews, mini_strategies, report, YOUR_QUERY)

            mini_elo = {}
            for name, rr in mini_reviews.items():
                mini_elo[name] = StrategyJudge.compute_initial_elo(rr)
            mini_history = set()
            for rn in range(1, 4):
                pairs = swiss_pairings(mini_strategies, mini_elo, mini_history)
                if not pairs: break
                for na, nb in pairs:
                    sa = next(s for s in mini_strategies if s["strategy_name"] == na)
                    sb = next(s for s in mini_strategies if s["strategy_name"] == nb)
                    ra, rb = mini_reviews.get(na, {}), mini_reviews.get(nb, {})
                    v = judge.judge_match(sa, sb, ra, rb, report, YOUR_QUERY)
                    w = v.get("winner", "")
                    if w and w != "tie":
                        loser = nb if w == na else na
                        _, _, mini_elo = StrategyJudge.update_elo(mini_elo, w, loser, False)
                    elif w == "tie":
                        _, _, mini_elo = StrategyJudge.update_elo(mini_elo, na, nb, True)

            print(f"\n  📊 进化效果对比:")
            for name in orig_top:
                orig_score = review_results.get(name, {}).get("weighted_score", 0)
                evo_name = next((en for en in evo_names if name in en), None)
                evo_score = mini_reviews.get(evo_name, {}).get("weighted_score", 0) if evo_name else 0
                evo_elo = mini_elo.get(evo_name, 0) if evo_name else 0
                orig_elo_val = elo.get(name, 0)
                if evo_name:
                    delta = evo_score - orig_score
                    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '—')
                    print(f"  {name[:40]} → {evo_name[:40]}")
                    print(f"    评分: {orig_score:.1f} → {evo_score:.1f} "
                          f"({arrow}{abs(int(delta))}) | Elo: {orig_elo_val:.0f} → {evo_elo:.0f}")

    # 保存结果
    result_file = os.path.join(TASK_DIR, "tournament_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "query": YOUR_QUERY,
            "independent_scores": {k: v["weighted_score"] for k, v in review_results.items()},
            "elo_final": elo, "ranking": [(n, e) for n, e in ranked],
            "total_matches": total_matches, "swiss_rounds": SWISS_ROUNDS,
            "records": tournament_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 结果: {result_file}")

    # 评审详情
    detail_dir = os.path.join(TASK_DIR, "reviews")
    os.makedirs(detail_dir, exist_ok=True)
    for name, rr in review_results.items():
        fname = name[:50].replace("/","_").replace(":","_")
        with open(os.path.join(detail_dir, f"review_{fname}.json"), "w", encoding="utf-8") as f:
            json.dump(rr, f, ensure_ascii=False, indent=2)
    print(f"  📁 评审详情: {detail_dir}/")

    print(f"\n{SEP}\n  ✅ {len(strategies)}策略 × {total_matches}场辩论 完成\n{SEP}")

if __name__ == "__main__":
    run_test()
