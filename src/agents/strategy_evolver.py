"""
AutoVS-Agent v3.0: Strategy Evolver — 基于锦标赛反馈的策略进化
=================================================================
读取锦标赛评审报告, 提取Top3策略的弱点, 定向修复生成进化版。
"""

from __future__ import annotations

import json, os, re
from typing import Any, Dict, List, Optional
from openai import OpenAI


EVOLVER_SYSTEM_PROMPT = """\
你是虚拟筛选策略进化专家。你的任务是基于评审反馈, 定向修复策略的弱点,
生成进化版本。

## 进化原则
1. 保留原策略的核心方法和已验证的优势 — 不要重新发明
2. 每个弱点必须有对应的修复步骤 — 不能只说不改
3. 修复要具体可操作 — 给出工具名/参数/阈值
4. 可以添加新步骤, 也可以修改现有步骤的参数
5. 如果某个弱点是根本性的(如靶点错配), 标注为不可修复并说明原因

## 输出格式
⚠️ 关键: pipeline_steps 必须最先输出且最详细! 每个step的step_name/tool/action/metric/threshold/rationale都必须填写具体内容, 禁止"?"!
{
  "strategy_name": "原名称 (v2 进化版)",
  "strategy_tagline": "...",
  "approach_type": "...",
  "pipeline_steps": [
    {"step_number":1, "step_name":"具体步骤名", "tool":"工具名+版本",
     "action":"详细操作(100-200字)", "metric":"指标", "threshold":"具体数值",
     "rationale":"理由(50-100字)"}
  ],
  "survival_estimate": "...",
  "contingency": "...",
  "rationale": "...(简短版: 修复了哪些弱点, 50-150字)",
  "strengths": [...],
  "weaknesses": [...],
  "estimated_runtime": "...",
  "suitable_when": "...",
  "evolution_changelog": ["修复1: ...", "修复2: ..."]
}
"""


class StrategyEvolver:
    """策略进化器 — 基于评审反馈定向修复策略弱点。"""

    def __init__(self, model="deepseek-chat", api_key=None, api_base=None,
                 temperature=0.4, max_tokens=16384):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=f"{self.api_base}/v1")
        return self._client

    # =========================================================================
    # 主入口
    # =========================================================================

    def evolve_top_n(self, strategies: list, review_results: dict,
                     tournament_records: list, research_report: dict,
                     user_query: str, n: int = 3) -> list:
        """对Top-N策略进行进化。

        Args:
            strategies: 所有策略列表
            review_results: {name: review_dict} 三人设评审结果
            tournament_records: 锦标赛判例记录
            research_report: 调研报告
            user_query: 用户原始任务
            n: 进化前N名

        Returns:
            进化后的策略列表 (保持原始顺序, 非TopN的保持不变)
        """
        # 按加权分排序, 取Top N
        sorted_items = sorted(review_results.items(),
                             key=lambda x: x[1]["weighted_score"], reverse=True)
        top_names = [name for name, _ in sorted_items[:n]]

        print(f"\n  🧬 进化 Top {n} 策略: {[n[:30] for n in top_names]}", flush=True)

        evolved = {}
        for name in top_names:
            s = next((st for st in strategies if st["strategy_name"] == name), None)
            if not s:
                continue
            rr = review_results.get(name, {})

            # 汇总该策略在所有判例中被提到的弱点
            verdict_weaknesses = self._extract_verdict_weaknesses(name, tournament_records)

            evo = self.evolve_strategy(s, rr, verdict_weaknesses,
                                       research_report, user_query)
            if evo:
                evolved[name] = evo
                print(f"    ✅ {name[:40]} → {evo.get('strategy_name','?')[:40]}", flush=True)
            else:
                print(f"    ❌ {name[:40]} 进化失败", flush=True)

        # 构建最终策略列表: 替换进化的, 保留其他的
        result = []
        for s in strategies:
            if s["strategy_name"] in evolved:
                result.append(evolved[s["strategy_name"]])
            else:
                result.append(s)
        return result

    def evolve_strategy(self, strategy: dict, review_result: dict,
                        verdict_weaknesses: list, research_report: dict,
                        user_query: str) -> Optional[dict]:
        """对单个策略进行定向进化。

        Args:
            strategy: 原始策略
            review_result: 三人设评审结果
            verdict_weaknesses: 裁判判例中提到的弱点
            research_report: 调研报告
            user_query: 用户原始任务

        Returns:
            进化版策略dict, 或None(进化失败)
        """
        # 汇总所有弱点
        all_weaknesses = self._extract_all_weaknesses(review_result, verdict_weaknesses)

        if not all_weaknesses:
            print(f"    ⚠️ 无弱点可修复, 跳过进化", flush=True)
            return None

        prompt = self._build_evolution_prompt(strategy, all_weaknesses,
                                               research_report, user_query)

        try:
            is_reasoner = "reasoner" in self.model.lower()
            kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                          messages=[{"role":"system","content":EVOLVER_SYSTEM_PROMPT},
                                    {"role":"user","content":prompt}])
            if not is_reasoner:
                kwargs["temperature"] = self.temperature
                kwargs["response_format"] = {"type":"json_object"}

            resp = self.client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if "{" in raw:
                raw = raw[raw.find("{"):]
            parsed = self._robust_json_parse(raw)

            if not parsed or "strategy_name" not in parsed:
                return None

            # 🆕 质量校验: pipeline_steps不能全是"?"
            steps = parsed.get("pipeline_steps", [])
            if steps:
                q_count = sum(1 for st in steps
                             if st.get("step_name","?") == "?"
                             or st.get("action","?") == "?"
                             or st.get("tool","?") == "?")
                if q_count >= len(steps) * 0.5:
                    print(f"    ⚠️ 进化版{len(steps)}步中{q_count}步为'?', token不足, 丢弃", flush=True)
                    return None

            # 确保有 evolution_changelog
            if "evolution_changelog" not in parsed:
                parsed["evolution_changelog"] = []

            # 确保pipeline_steps格式正确
            if "pipeline_steps" in parsed:
                for i, step in enumerate(parsed["pipeline_steps"]):
                    if "step_number" not in step:
                        step["step_number"] = i + 1

            return parsed

        except Exception as e:
            print(f"    ❌ 进化LLM调用失败: {e}", flush=True)
            return None

    # =========================================================================
    # 弱点提取
    # =========================================================================

    def _extract_all_weaknesses(self, review_result: dict,
                                 verdict_weaknesses: list) -> list:
        """汇总所有评审来源的弱点。"""
        weaknesses = []

        for rep in review_result.get("reports", []):
            reviewer = rep.get("reviewer_name", "?")
            score = rep.get("overall_score", "?")

            # 维度级别的弱点
            for d in rep.get("dimension_scores", []):
                if d.get("score", 50) < 60:  # 低于60分的维度
                    weaknesses.append({
                        "source": f"{reviewer} - {d.get('name','?')}({d.get('score','?')}分)",
                        "issue": d.get("comment", ""),
                        "severity": "major" if d.get("score", 50) < 40 else "minor",
                    })

            # 明确列出的弱点
            for w in rep.get("key_weaknesses", []):
                weaknesses.append({
                    "source": reviewer,
                    "issue": w,
                    "severity": "minor",
                })

            # 致命缺陷
            for f in rep.get("critical_flaws", []):
                weaknesses.append({
                    "source": f"{reviewer} [致命缺陷]",
                    "issue": f,
                    "severity": "critical",
                })

            # 改进建议 (反向提取为弱点)
            # (在reviewer的维度comment中已经包含了, 这里不再重复)

        # 裁判发现的弱点
        for vw in verdict_weaknesses:
            weaknesses.append({
                "source": "锦标赛裁判",
                "issue": vw,
                "severity": "minor",
            })

        # 去重 (基于issue的相似度)
        seen = set()
        unique = []
        for w in weaknesses:
            key = w["issue"][:80]
            if key not in seen:
                seen.add(key)
                unique.append(w)

        return unique

    @staticmethod
    def _extract_verdict_weaknesses(strategy_name: str,
                                     tournament_records: list) -> list:
        """从锦标赛判例中提取涉及该策略的弱点。"""
        weaknesses = []
        for rec in tournament_records:
            v = rec.get("verdict", {})
            # 看该策略是A还是B
            if rec.get("strategy_a") == strategy_name:
                suggestions = v.get("suggestions_a", [])
                for s in suggestions:
                    weaknesses.append(s)
                # 从维度对比中提取
                for d in v.get("dimension_comparison", []):
                    if d.get("winner") != "A" and d.get("winner") != "tie":
                        weaknesses.append(f"[{d.get('dimension','?')}] {d.get('reasoning','')[:200]}")
            elif rec.get("strategy_b") == strategy_name:
                suggestions = v.get("suggestions_b", [])
                for s in suggestions:
                    weaknesses.append(s)
                for d in v.get("dimension_comparison", []):
                    if d.get("winner") != "B" and d.get("winner") != "tie":
                        weaknesses.append(f"[{d.get('dimension','?')}] {d.get('reasoning','')[:200]}")
        return weaknesses

    # =========================================================================
    # Prompt
    # =========================================================================

    def _build_evolution_prompt(self, strategy: dict, weaknesses: list,
                                 research_report: dict, user_query: str) -> str:
        parts = []

        if user_query:
            parts.append(f"## 用户原始任务\n{user_query}\n")

        # 策略原文
        parts.append(f"## 原始策略\n")
        parts.append(f"名称: {strategy.get('strategy_name','?')}")
        parts.append(f"方法: {strategy.get('approach_type','?')}")
        parts.append(f"标签: {strategy.get('strategy_tagline','?')}")
        parts.append(f"原理: {strategy.get('rationale','?')[:500]}")
        parts.append(f"\n### 原始步骤:")
        for st in strategy.get("pipeline_steps", []):
            parts.append(f"  Step{st.get('step_number','?')}: {st.get('step_name','?')} "
                        f"[{st.get('tool','?')}]")
            parts.append(f"    操作: {st.get('action','?')[:200]}")
            parts.append(f"    指标: {st.get('metric','?')} | 阈值: {st.get('threshold','?')}")
        parts.append(f"\n存活估算: {strategy.get('survival_estimate','?')}")
        parts.append(f"应急预案: {strategy.get('contingency','?')}")
        parts.append(f"优势: {strategy.get('strengths',[])}")
        parts.append(f"劣势: {strategy.get('weaknesses',[])})")

        # 需要修复的弱点
        parts.append(f"\n## ⚠️ 需要修复的弱点 ({len(weaknesses)}个)\n")
        critical = [w for w in weaknesses if w["severity"] == "critical"]
        major = [w for w in weaknesses if w["severity"] == "major"]
        minor = [w for w in weaknesses if w["severity"] == "minor"]

        if critical:
            parts.append("### 🚨 致命缺陷 (必须修复!)")
            for i, w in enumerate(critical, 1):
                parts.append(f"{i}. [{w['source']}] {w['issue']}")

        if major:
            parts.append("\n### ⚠️ 重要问题 (强烈建议修复)")
            for i, w in enumerate(major, 1):
                parts.append(f"{i}. [{w['source']}] {w['issue']}")

        if minor:
            parts.append("\n### 💡 改进建议")
            for i, w in enumerate(minor[:8], 1):  # 最多8条
                parts.append(f"{i}. [{w['source']}] {w['issue']}")

        # 靶点背景 (精简, 只保留关键数据)
        parts.append(f"\n## 靶点背景")
        parts.append(f"靶点: {research_report.get('target_name','?')} | "
                     f"基因: {research_report.get('gene_symbol','?')} | "
                     f"物种: {research_report.get('target_organism','?')}")
        if research_report.get('full_report_text'):
            parts.append(f"\n调研报告摘要: {research_report['full_report_text'][:800]}")

        parts.append("\n## 任务\n"
                     "请生成进化版策略。保留优势, 针对每个弱点给出具体修复。\n"
                     "输出完整JSON, strategy_name加'(v2 进化版)'后缀。")

        return "\n\n".join(parts)

    # =========================================================================
    # 辅助
    # =========================================================================

    @staticmethod
    def _robust_json_parse(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```\w*\n', '', cleaned)
            cleaned = re.sub(r'\n```$', '', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.JSONDecoder().raw_decode(cleaned[s:e+1])[0]
            except json.JSONDecodeError:
                pass
        return {}
