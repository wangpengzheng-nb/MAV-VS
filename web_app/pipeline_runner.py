"""
Pipeline Runner — 包装现有5步流水线, 加进度回调
=================================================
不修改任何现有代码, 纯包装层。
"""
from __future__ import annotations
import os, sys, json, math, re, threading
from typing import Callable, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

ProgressCallback = Callable[[str, str, int], None]
# callback(step_name, status, percent)


class PipelineRunner:
    """5步虚拟筛选流水线, 每步通过 callback 推送进度。

    步骤:
      0. 靶点调研 (0-20%)
      1. 策略生成 (20-40%)
      2. 策略审评 (40-70%)
      3. 策略进化 (70-95%)
      4. 输出结果 (95-100%)
    """

    STEPS = [
        ("靶点调研", 0, 20),
        ("策略生成", 20, 40),
        ("策略审评", 40, 70),
        ("策略进化", 70, 95),
        ("输出结果", 95, 100),
    ]

    def __init__(self, progress_callback: Optional[ProgressCallback] = None):
        self.callback = progress_callback or (lambda s, st, p: None)
        self._task_dir = None

    def _emit(self, step: str, status: str, percent: int, msg: str = ""):
        self.callback(step, status, percent, msg)

    def _sub_progress(self, step_idx: int, fraction: float):
        """在某个步骤内更新子进度"""
        _, start, end = self.STEPS[step_idx]
        pct = int(start + (end - start) * fraction)
        self.callback(self.STEPS[step_idx][0], "running", pct)

    # =========================================================================
    # 主入口
    # =========================================================================

    def run(self, query: str, task_dir: str) -> Dict[str, Any]:
        """运行完整流水线。"""
        from src.agents.target_scout import TargetScoutAgent
        from src.agents.strategy_generator import StrategyGeneratorAgent
        from src.agents.expert_committee import TournamentReviewer
        from src.agents.judge_agent import StrategyJudge
        from src.agents.strategy_evolver import StrategyEvolver

        self._task_dir = task_dir
        os.makedirs(task_dir, exist_ok=True)

        # ── Step 0: 靶点调研 ──
        self._emit("靶点调研", "running", 0, "正在搜索基因/PDB/ChEMBL/PubMed...")
        scout = TargetScoutAgent()
        report = scout.deep_research(query)
        self._emit("靶点调研", "running", 10, f"靶点: {report.get('target_name','?')}")

        # 保存调研报告
        rf = os.path.join(task_dir, "research_report.md")
        bs = report.get('binding_site', {})
        with open(rf, "w", encoding="utf-8") as f:
            f.write(f"# 靶点深度调研报告: {report.get('target_name','?')}\n\n"
                    f"## 靶点信息\n"
                    f"**基因**: {report.get('gene_symbol','?')} | "
                    f"**UniProt**: {report.get('uniprot_id','?')} | "
                    f"**物种**: {report.get('target_organism','?')}\n\n"
                    f"## 结合位点\n- 口袋: {bs.get('pocket_description','?')}\n\n"
                    f"{report.get('full_report_text', '')}\n")
        self._emit("靶点调研", "done", 20,
                   f"靶点: {report.get('target_name','?')}, "
                   f"PDB: {len(report.get('verified_pdb_structures',[]))}个, "
                   f"ChEMBL: {len(report.get('chembl_activities',[]))}条")

        # ── Step 1: 策略生成 ──
        self._emit("策略生成", "running", 20, "正在生成10个虚拟筛选策略...")
        report["_user_query"] = query
        gen = StrategyGeneratorAgent()
        result = gen.generate_strategies(report)
        strategies = result["strategies"]
        self._emit("策略生成", "running", 30, f"已生成 {len(strategies)} 个策略")

        # 保存策略
        strat_dir = os.path.join(task_dir, "strategies")
        os.makedirs(strat_dir, exist_ok=True)
        for i, s in enumerate(strategies, 1):
            sf = os.path.join(strat_dir,
                f"strategy_{i:02d}_{s['strategy_name'].replace('/','_').replace(':','_')[:60]}.md")
            with open(sf, "w", encoding="utf-8") as f:
                f.write(f"# 策略 {i}: {s['strategy_name']}\n\n"
                        f"**标签**: {s.get('strategy_tagline','')} | "
                        f"**方法**: {s.get('approach_type','?')} | "
                        f"**耗时**: {s.get('estimated_runtime','?')}\n\n")
                f.write(f"## 原理\n{s.get('rationale','')}\n\n## 步骤\n")
                for st in s.get("pipeline_steps", []):
                    f.write(f"### Step {st.get('step_number','?')}: {st.get('step_name','?')}\n"
                            f"| 工具 | 指标 | 阈值 |\n|---|---|---|\n"
                            f"| {st.get('tool','?')} | {st.get('metric','?')} | {st.get('threshold','?')} |\n\n"
                            f"**操作**: {st.get('action','?')}\n\n**理由**: {st.get('rationale','?')}\n\n")
                f.write(f"**存活**: {s.get('survival_estimate','?')}\n\n"
                        f"**应急**: {s.get('contingency','?')}\n\n"
                        f"## 优势\n"+"\n".join(f"- {x}" for x in s.get('strengths',[]))+"\n\n"
                        f"## 劣势\n"+"\n".join(f"- {x}" for x in s.get('weaknesses',[]))+"\n")
        self._emit("策略生成", "done", 40)

        if len(strategies) < 2:
            self._emit("输出结果", "done", 100, "策略不足, 跳过后续步骤")
            return self._build_output(report, strategies, {}, [], [], [])

        # ── Step 2: 策略审评 ──
        self._emit("策略审评", "running", 40, "正在进行三人设独立评审...")

        reviewer = TournamentReviewer()
        review_results = {}
        for i, s in enumerate(strategies):
            self._sub_progress(2, 0.05 + 0.4 * (i / len(strategies)))
            rr = reviewer.review_strategy(s, report, query)
            review_results[s["strategy_name"]] = rr

        self._emit("策略审评", "running", 55, "评审后校准...")
        review_results = reviewer.calibrate_all(review_results, strategies, report, query)

        # 锦标赛
        self._emit("策略审评", "running", 58, "正在进行瑞士制锦标赛...")
        judge = StrategyJudge()
        elo = {}
        for name, rr in review_results.items():
            elo[name] = StrategyJudge.compute_initial_elo(rr)
        init_elo = dict(elo)

        history = set()
        tournament_records = []
        SWISS_ROUNDS = 4

        for rn in range(1, SWISS_ROUNDS + 1):
            self._sub_progress(2, 0.5 + 0.4 * (rn / SWISS_ROUNDS))
            pairs = self._swiss_pairings(strategies, elo, history)
            if not pairs: break
            for na, nb in pairs:
                sa = next(s for s in strategies if s["strategy_name"] == na)
                sb = next(s for s in strategies if s["strategy_name"] == nb)
                ra, rb = review_results.get(na, {}), review_results.get(nb, {})
                v = judge.judge_match(sa, sb, ra, rb, report, query)
                w = v.get("winner", "")
                if w and w != "tie":
                    loser = nb if w == na else na
                    _, _, elo = StrategyJudge.update_elo(elo, w, loser, False)
                elif w == "tie":
                    _, _, elo = StrategyJudge.update_elo(elo, na, nb, True)
                tournament_records.append({"round": rn, "strategy_a": na,
                    "strategy_b": nb, "verdict": v})
            max_shift = max(abs(elo[n]-init_elo.get(n,elo[n])) for n in elo) if init_elo else 0
            if rn >= 3 and max_shift < 5: break

        ranked = sorted(elo.items(), key=lambda x: x[1], reverse=True)

        # 保存评审
        review_dir = os.path.join(task_dir, "reviews")
        os.makedirs(review_dir, exist_ok=True)
        for name, rr in review_results.items():
            fname = name[:50].replace("/","_").replace(":","_")
            with open(os.path.join(review_dir, f"review_{fname}.json"), "w", encoding="utf-8") as f:
                json.dump(rr, f, ensure_ascii=False, indent=2)

        # 保存锦标赛结果
        with open(os.path.join(task_dir, "tournament_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "query": query,
                "independent_scores": {k: v["weighted_score"] for k, v in review_results.items()},
                "elo_final": elo, "ranking": [(n, e) for n, e in ranked],
                "records": tournament_records,
            }, f, ensure_ascii=False, indent=2)

        self._emit("策略审评", "done", 70,
                   f"🏆 {ranked[0][0][:30]} Elo={ranked[0][1]:.0f}")

        # ── Step 3: 策略进化 ──
        self._emit("策略进化", "running", 70, "正在进化 Top 3 策略...")
        EVOLVE_TOP_N = 3
        evolved_strategies = list(strategies)

        if len(strategies) >= 3:
            evolver = StrategyEvolver()
            evolved_strategies = evolver.evolve_top_n(
                strategies, review_results, tournament_records,
                report, query, n=EVOLVE_TOP_N)

            self._emit("策略进化", "running", 80, "正在进行迷你锦标赛验证...")

            evo_names = [s["strategy_name"] for s in evolved_strategies
                         if "(v2" in s.get("strategy_name", "")]
            if evo_names:
                orig_top = [ranked[i][0] for i in range(min(EVOLVE_TOP_N, len(ranked)))]
                orig_strategies = [s for s in strategies if s["strategy_name"] in orig_top]
                evo_only = [s for s in evolved_strategies if s["strategy_name"] in evo_names]
                mini = orig_strategies + evo_only
                mini_reviewer = TournamentReviewer()
                mini_reviews = {}
                for i, s in enumerate(mini):
                    self._sub_progress(3, 0.6 + 0.3 * (i / len(mini)))
                    rr = mini_reviewer.review_strategy(s, report, query)
                    mini_reviews[s["strategy_name"]] = rr
                    # 显示进化效果对比
                    for ii, oname in enumerate(orig_top):
                        ename = next((e for e in evo_names if oname in e), None)
                        if ename and ename in mini_reviews and oname in review_results:
                            delta = (mini_reviews[ename].get("weighted_score", 0) -
                                     review_results[oname].get("weighted_score", 0))
                            arrow = "↑" if delta > 0 else "↓"
                            self._emit("策略进化", "running", 85 + ii * 3,
                                       f"进化效果: {oname[:20]}→{ename[:20]} {arrow}{abs(int(delta))}")

        # 保存进化策略
        evo_dir = os.path.join(task_dir, "evolved_strategies")
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
                    for c in s.get('evolution_changelog', []):
                        f.write(f"- {c}\n")
                    f.write(f"\n## 原理\n{s.get('rationale','')}\n\n## 步骤\n")
                    for st in s.get("pipeline_steps", []):
                        f.write(f"### Step {st.get('step_number','?')}: {st.get('step_name','?')}\n"
                                f"| 工具 | 指标 | 阈值 |\n|---|---|---|\n"
                                f"| {st.get('tool','?')} | {st.get('metric','?')} | {st.get('threshold','?')} |\n\n"
                                f"**操作**: {st.get('action','?')}\n\n**理由**: {st.get('rationale','?')}\n\n")
                    f.write(f"**存活**: {s.get('survival_estimate','?')}\n\n"
                            f"**应急**: {s.get('contingency','?')}\n\n"
                            f"## 优势\n"+"\n".join(f"- {x}" for x in s.get('strengths',[]))+"\n\n"
                            f"## 劣势\n"+"\n".join(f"- {x}" for x in s.get('weaknesses',[]))+"\n")

        self._emit("策略进化", "done", 95, "进化完成")

        # ── Step 4: 输出 ──
        self._emit("输出结果", "running", 95, "正在整理输出...")
        output = self._build_output(report, strategies, review_results,
                                     ranked, tournament_records, evolved_strategies)
        self._emit("输出结果", "done", 100, "完成!")

        return output

    # =========================================================================
    # 辅助
    # =========================================================================

    @staticmethod
    def _swiss_pairings(strategies: list, elo: dict, history: set) -> list:
        ranked = sorted(strategies, key=lambda s: elo.get(s["strategy_name"], 1500), reverse=True)
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

    def _build_output(self, report, strategies, review_results, ranked,
                      tournament_records, evolved_strategies) -> dict:
        """组装最终输出, 供前端渲染。"""
        # 调研报告
        report_md = report.get("full_report_text", "")

        # 排名
        ranking_list = []
        for rank, (name, esc) in enumerate(ranked[:5], 1):
            rs = review_results.get(name, {})
            ranking_list.append({
                "rank": rank,
                "name": name[:80],
                "score": rs.get("weighted_score", 0),
                "elo": esc,
            })

        # 进化策略 (只有v2版本)
        evolved_list = []
        for s in evolved_strategies:
            if "(v2" in s.get("strategy_name", ""):
                orig_name = s["strategy_name"].replace(" (v2 进化版)", "")
                orig_score = review_results.get(orig_name, {}).get("weighted_score", 0)
                evolved_list.append({
                    "name": s["strategy_name"][:80],
                    "tagline": s.get("strategy_tagline", "")[:200],
                    "approach": s.get("approach_type", ""),
                    "rationale": s.get("rationale", "")[:2000],
                    "changelog": s.get("evolution_changelog", []),
                    "steps": s.get("pipeline_steps", []),
                    "survival": s.get("survival_estimate", ""),
                    "strengths": s.get("strengths", []),
                    "weaknesses": s.get("weaknesses", []),
                    "orig_score": orig_score,
                })

        return {
            "target_name": report.get("target_name", "?"),
            "gene": report.get("gene_symbol", ""),
            "organism": report.get("target_organism", ""),
            "report_md": report_md,
            "strategy_count": len(strategies),
            "ranking": ranking_list,
            "evolved_strategies": evolved_list,
            "task_dir": os.path.basename(self._task_dir or ""),
        }
