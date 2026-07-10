#!/usr/bin/env python3
"""AutoVS-Agent v3.0 — 策略排名锦标赛 (瑞士制版)"""
from __future__ import annotations
import hashlib, os, sys, re, glob as _glob, json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

YOUR_QUERY = "Finding ligands targeting the triple Tudor domain of SETDB1"
SKIP_RESEARCH = False
SKIP_STRATEGY = False
LOAD_STRATEGIES_DIR = "/users_home/wangpengzheng/药物筛选智能体/分析文件/任务_20260710_110327_bdc114e6/strategies"
RESEARCH_REPORT_DIR = "/users_home/wangpengzheng/药物筛选智能体/分析文件/任务_20260710_110327_bdc114e6"
SWISS_ROUNDS = 4

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

    # ── Step 2: 三人设独立评审 ──
    hdr("Step 2: 三人设独立评审")
    from src.agents.expert_committee import TournamentReviewer

    reviewer = TournamentReviewer()
    review_results = {}

    for i, s in enumerate(strategies, 1):
        print(f"\n  [{i}/{len(strategies)}]", end="", flush=True)
        rr = reviewer.review_strategy(s, report, YOUR_QUERY)
        review_results[s["strategy_name"]] = rr

    # 🔧 评审后校准: 让评审官看到全局分布后重新拉开分数
    review_results = reviewer.calibrate_all(review_results, strategies, report, YOUR_QUERY)

    print(f"\n\n  📊 独立评分排名(校准后):")
    sorted_by_score = sorted(review_results.items(),
                             key=lambda x: x[1]["weighted_score"], reverse=True)
    for rank, (name, rr) in enumerate(sorted_by_score, 1):
        emoji = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else "  "
        flaws = len(rr.get("all_critical_flaws",[]))
        print(f"  {emoji} {rank}. {name[:55]}  {rr['weighted_score']:.1f}分"
              f"{' ⚠️'+str(flaws)+'致命缺陷' if flaws else ''}")

    # 分数分布统计
    scores = [rr["weighted_score"] for rr in review_results.values()]
    score_range = max(scores) - min(scores)
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    import math
    std_dev = math.sqrt(variance)
    print(f"\n  📊 分数统计: 范围={score_range:.1f} | 均值={mean_score:.1f} | 标准差={std_dev:.1f}")
    if score_range < 20 or std_dev < 8:
        print(f"  ⚠️ 评分区分度不足! 建议检查评审官prompt或手动拉开差距")
    else:
        print(f"  ✅ 评分区分度合理")

    # ── Step 3: 瑞士制锦标赛 ──
    hdr("Step 3: 瑞士制锦标赛")
    from src.agents.judge_agent import StrategyJudge

    elo = {}
    for name, rr in review_results.items():
        elo[name] = StrategyJudge.compute_initial_elo(rr)
    init_elo = dict(elo)

    total_matches = 0
    full_pairings = min(len(strategies) * (len(strategies)-1) // 2, 45)
    print(f"\n  🏟️  瑞士制: {len(strategies)}策略 × ≤{SWISS_ROUNDS}轮 "
          f"(最多{len(strategies)//2 * SWISS_ROUNDS}场, 全量配对需{full_pairings}场)")

    judge = StrategyJudge()
    history = set()
    tournament_records = []

    for rn in range(1, SWISS_ROUNDS + 1):
        pairings = swiss_pairings(strategies, elo, history)
        if not pairings:
            print(f"\n  ✅ 第{rn}轮无新配对, 锦标赛结束"); break

        print(f"\n  ── 第{rn}轮 ({len(pairings)}场) ──")
        round_shifts = []

        for na, nb in pairings:
            sa = next(s for s in strategies if s["strategy_name"] == na)
            sb = next(s for s in strategies if s["strategy_name"] == nb)
            ra = review_results.get(na, {})
            rb = review_results.get(nb, {})

            verdict = judge.judge_match(sa, sb, ra, rb, report, YOUR_QUERY)
            winner = verdict.get("winner", "")
            is_tie = (winner == "tie")
            total_matches += 1

            if not is_tie:
                loser = nb if winner == na else na
                _, _, elo = StrategyJudge.update_elo(elo, winner, loser, False)
                round_shifts.append(abs(elo[winner] - init_elo.get(winner, elo[winner]-10) if winner in init_elo else 10))
            else:
                _, _, elo = StrategyJudge.update_elo(elo, na, nb, True)

            conf = verdict.get("confidence", 0.5)
            scores = f"A:{verdict.get('strategy_a_score','?'):.0f} B:{verdict.get('strategy_b_score','?'):.0f}"
            print(f"    {na[:28]} vs {nb[:28]}")
            print(f"    → {'tie' if is_tie else winner[:40]} | {scores} | conf={conf:.0%}")

            tournament_records.append({"round": rn, "strategy_a": na,
                "strategy_b": nb, "verdict": verdict})

        # 收敛检查
        max_shift = max(abs(elo[n] - init_elo.get(n, elo[n])) for n in elo) if init_elo else 0
        if rn >= 3 and max_shift < 5:
            print(f"\n    ✅ Elo变化 <5, 已收敛, 提前结束"); break

    # ── 最终排名 ──
    hdr("🏆 最终排名")
    ranked = sorted(elo.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, esc) in enumerate(ranked, 1):
        emoji = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else "  "
        init_score = review_results.get(name, {}).get("weighted_score", "?")
        change = esc - init_elo.get(name, esc)
        print(f"  {emoji} {rank}. {name[:55]}")
        print(f"     独立评分={init_score:.1f} | Elo={esc:.0f} "
              f"({'↑'+str(int(change)) if change>0 else '↓'+str(int(-change)) if change<0 else '—'})")

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
