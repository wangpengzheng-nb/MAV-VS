#!/usr/bin/env python3
"""分步测试管线 — 每步独立, 输出持久化, 支持断点续跑。"""
from __future__ import annotations
import os, sys, json, hashlib, argparse, itertools
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════
# 配置 — 修改这里
# ═══════════════════════════════════════════
TASK_QUERY = "Finding ligands targeting the triple Tudor domain of SETDB1"
PRIOR_KNOWLEDGE = ""
TASK_DIR = ""  # 修改此处或通过 --task-dir 参数指定

SEP = "=" * 60


def get_task_dir() -> str:
    if TASK_DIR and os.path.isdir(TASK_DIR):
        return TASK_DIR
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(TASK_QUERY.encode()).hexdigest()[:8]
    d = os.path.join(os.path.dirname(__file__), "分析文件", f"任务_{ts}_{h}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "query.txt"), "w") as f:
        f.write(TASK_QUERY)
    return d


def file_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

# ═══════════════════════════════════════════
# Step 0: 靶点调研
# ═══════════════════════════════════════════
def step0_research(td: str) -> dict:
    rf = os.path.join(td, "research_report.md")
    if file_ok(rf):
        print(f"  ⏩ 已有调研报告, 跳过")
        with open(rf, encoding="utf-8") as f:
            return {"full_report_text": f.read(), "target_name": "loaded"}
    from src.agents.target_scout import TargetScoutAgent
    print("  🔬 开始靶点调研...")
    report = TargetScoutAgent().deep_research(TASK_QUERY)
    bs = report.get('binding_site', {})
    summary = report.get('executive_summary', '')
    with open(rf, "w", encoding="utf-8") as f:
        f.write(f"# 靶点深度调研报告: {report.get('target_name','?')}\n\n")
        f.write(f"## 靶点信息\n**基因**: {report.get('gene_symbol','?')} | "
                f"**UniProt**: {report.get('uniprot_id','?')} | "
                f"**物种**: {report.get('target_organism','?')}\n\n")
        f.write(f"## 结合位点\n- 口袋: {bs.get('pocket_description','?')}\n\n")
        if summary:
            f.write(f"## 执行摘要\n{summary}\n\n---\n\n")
            f.write(f"## 完整报告\n")
        f.write(report.get('full_report_text', '') + "\n")
    print(f"  ✅ {rf}")
    return report


# ═══════════════════════════════════════════
# Step 1: 策略生成
# ═══════════════════════════════════════════
def step1_generate(td: str, report: dict) -> list:
    sd = os.path.join(td, "strategies")
    existing = sorted([f for f in os.listdir(sd) if f.endswith('.md')]) if os.path.isdir(sd) else []
    if existing:
        print(f"  ⏩ 已有 {len(existing)} 个策略文件, 跳过")
        from test_tournament import load_strategies_from_dir
        return load_strategies_from_dir(sd)
    from src.agents.strategy_generator import StrategyGeneratorAgent
    print("  📝 开始策略生成...")
    report["_user_query"] = TASK_QUERY
    result = StrategyGeneratorAgent().generate_strategies(report, prior_knowledge=PRIOR_KNOWLEDGE)
    strategies = result["strategies"]
    os.makedirs(sd, exist_ok=True)
    for i, s in enumerate(strategies, 1):
        sf = os.path.join(sd, f"strategy_{i:02d}_{s['strategy_name'][:60]}.md")
        with open(sf, "w", encoding="utf-8") as f:
            f.write(f"# 策略 {i}: {s.get('strategy_name','?')}\n\n"
                    f"**方法**: {s.get('approach_category','?')} | "
                    f"**标签**: {s.get('strategy_tagline','')}\n\n"
                    f"## 原理\n{s.get('rationale','')}\n")
        # 保存完整JSON
        with open(os.path.join(sd, f"strategy_{i:02d}.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {len(strategies)} 策略 → {sd}/")
    return strategies


# ═══════════════════════════════════════════
# Step 2: 策略审评 (全排列 pairwise 投票)
# ═══════════════════════════════════════════
def step2_evaluate(td: str, strategies: list, report: dict) -> dict:
    vf = os.path.join(td, "reviews", "all_votes.json")
    if file_ok(vf):
        print(f"  ⏩ 已有审评结果, 跳过")
        with open(vf, encoding="utf-8") as f:
            return json.load(f)
    from src.agents.expert_committee import TournamentReviewer, REVIEWER_CONFIGS
    from src.agents.judge_agent import VoteAggregator
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tenacity import retry, stop_after_attempt, wait_exponential

    reviewer = TournamentReviewer()
    aggregator = VoteAggregator()
    n = len(strategies)
    pairs = list(itertools.combinations(range(n), 2))
    total_matches = len(pairs) * len(REVIEWER_CONFIGS)
    print(f"  🏟️  {n}策略 × {len(pairs)}对 × {len(REVIEWER_CONFIGS)}评审官 = {total_matches}次投票")

    def vote_one(rid, mid, sa, sb):
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
        def _call():
            return reviewer.compare_strategies(sa, sb, report, TASK_QUERY,
                                               PRIOR_KNOWLEDGE, reviewer_id=rid, match_id=mid)
        try:
            return _call()
        except Exception as e:
            print(f"    ❌ [{rid}] 失败: {e}", flush=True)
            return {"reviewer_id": rid, "match_id": mid, "overall_verdict": "tie",
                    "verdict_confidence": "low", "dimension_votes": [],
                    "critical_concerns": {}, "suggestions": {}, "decision_logic": f"失败: {e}"}

    for pi, (i, j) in enumerate(pairs):
        sa, sb = strategies[i], strategies[j]
        mid = f"{sa['strategy_name'][:20]}_vs_{sb['strategy_name'][:20]}"
        print(f"  [{pi+1}/{len(pairs)}]", end="", flush=True)
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = {ex.submit(vote_one, cfg["id"], mid, sa, sb): cfg for cfg in REVIEWER_CONFIGS}
            for fut in as_completed(futs):
                aggregator.add_result(fut.result())

    ranking = aggregator.rank(strategies)
    diagnostics = aggregator.generate_diagnostic(top_n=3)
    print(f"\n  🏆 冠军: {ranking[0]['strategy_name'][:40]} (得票: {ranking[0]['total_votes']:.1f})")

    os.makedirs(os.path.join(td, "reviews"), exist_ok=True)
    result = {"results": aggregator.results, "ranking": ranking, "diagnostics": diagnostics}
    with open(vf, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 审评结果 → {vf}")
    return result


# ═══════════════════════════════════════════
# Step 3: 策略进化
# ═══════════════════════════════════════════
def step3_evolve(td: str, strategies: list, report: dict, eval_result: dict) -> list:
    ed = os.path.join(td, "evolved_strategies")
    existing = sorted([f for f in os.listdir(ed) if f.endswith('.md')]) if os.path.isdir(ed) else []
    if existing:
        print(f"  ⏩ 已有 {len(existing)} 个进化策略, 跳过")
        return strategies
    from src.agents.strategy_evolver import StrategyEvolver
    from src.agents.judge_agent import VoteAggregator as VA
    ranking = eval_result.get("ranking", [])
    if not ranking:
        print("  ⚠️ 无排名数据, 跳过进化"); return strategies

    # 用 VoteAggregator 重新加载结果, 准备进化输入
    agg = VA()
    agg.add_results(eval_result.get("results", []))
    TOP_N = 4
    top_names = [r["strategy_name"] for r in ranking[:TOP_N]]

    # 为每个Top策略准备进化输入 (blueprint + UUID + diagnosis)
    evolution_inputs = []
    for name in top_names:
        s = next((st for st in strategies if st["strategy_name"] == name), None)
        if s:
            evo_in = agg.prepare_evolution_input(s, name)
            evolution_inputs.append(evo_in)
            concerns_n = len(evo_in["diagnosis"]["concerns"])
            suggs_n = len(evo_in["diagnosis"]["suggestions"])
            print(f"  📋 [{name[:40]}] concerns={concerns_n} suggestions={suggs_n}")

    evolver = StrategyEvolver()
    print(f"  🧬 进化 Top {len(evolution_inputs)} 策略...")
    # 构建进化用的 review_results 兼容格式
    evo_review_map = {ei["blueprint"]["strategy_name"]: ei["diagnosis"] for ei in evolution_inputs}
    evolved = evolver.evolve_top_n(strategies, evo_review_map, [], report,
                                    TASK_QUERY, n=TOP_N,
                                    prior_knowledge=PRIOR_KNOWLEDGE)
    os.makedirs(ed, exist_ok=True)
    for i, s in enumerate(evolved, 1):
        if "(v2" in s.get("strategy_name", ""):
            sf = os.path.join(ed, f"evolved_{i:02d}_{s['strategy_name'][:50]}.md")
            with open(sf, "w", encoding="utf-8") as f:
                f.write(f"# 进化策略 {i}: {s['strategy_name']}\n")
                for c in s.get('evolution_changelog', []): f.write(f"- {c}\n")
                f.write(f"\n## 原理\n{s.get('rationale','')}\n")
    print(f"  ✅ 进化策略 → {ed}/")
    return evolved


# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="AutoVS-Agent 分步测试管线")
    p.add_argument("--step", default="all", choices=["all","research","generate","evaluate","evolve"])
    p.add_argument("--to-end", action="store_true", help="从指定step运行到结束")
    p.add_argument("--task-dir", default="", help="指定任务目录(复用已有上游文件)")
    args = p.parse_args()

    if args.task_dir:
        global TASK_DIR
        TASK_DIR = args.task_dir
    td = get_task_dir()
    print(f"\n{SEP}\n  Pipeline Test\n  Task: {os.path.basename(td)}\n  Query: {TASK_QUERY}\n{SEP}\n")

    steps = ["research", "generate", "evaluate", "evolve"]
    if args.step != "all":
        start_idx = steps.index(args.step)
        if not args.to_end:
            steps = [args.step]
        else:
            steps = steps[start_idx:]

    # Step 0: 调研
    report = {}
    if "research" in steps:
        print(f"\n{'─'*40}\n  [Step 0] 靶点调研\n{'─'*40}")
        report = step0_research(td)
    else:
        rf = os.path.join(td, "research_report.md")
        if file_ok(rf):
            with open(rf, encoding="utf-8") as f:
                report = {"full_report_text": f.read(), "target_name": "loaded"}
            print(f"  📋 加载已有调研报告: {rf}")

    # Step 1: 策略生成
    strategies = []
    if "generate" in steps:
        print(f"\n{'─'*40}\n  [Step 1] 策略生成\n{'─'*40}")
        strategies = step1_generate(td, report)
    else:
        sd = os.path.join(td, "strategies")
        if os.path.isdir(sd):
            json_files = sorted([f for f in os.listdir(sd) if f.endswith('.json')])
            if json_files:
                for jf in json_files:
                    with open(os.path.join(sd, jf), encoding="utf-8") as f:
                        strategies.append(json.load(f))
            else:
                from test_tournament import load_strategies_from_dir
                strategies = load_strategies_from_dir(sd)
            print(f"  📋 加载已有 {len(strategies)} 策略")
    if not strategies and any(s in steps for s in ["generate","evaluate","evolve"]):
        print("  ❌ 无策略, 停止"); return

    # Step 2: 审评
    eval_result = {}
    if "evaluate" in steps:
        print(f"\n{'─'*40}\n  [Step 2] 策略审评\n{'─'*40}")
        eval_result = step2_evaluate(td, strategies, report)
    else:
        vf = os.path.join(td, "reviews", "all_votes.json")
        if file_ok(vf):
            with open(vf, encoding="utf-8") as f:
                eval_result = json.load(f)
            ranking = eval_result.get("ranking", [])
            if ranking:
                print(f"  📋 加载已有审评: {len(ranking)}排名, 冠军={ranking[0]['strategy_name'][:30]}")

    # Step 3: 进化
    if "evolve" in steps:
        print(f"\n{'─'*40}\n  [Step 3] 策略进化\n{'─'*40}")
        step3_evolve(td, strategies, report, eval_result)

    print(f"\n{SEP}\n  Done. Task dir: {td}\n{SEP}")


if __name__ == "__main__":
    main()
