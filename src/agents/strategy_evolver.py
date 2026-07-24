"""
AutoVS-Agent v4: Strategy Evolver v2 — 隔离进化 + JSON输出 + 逐条引用
=========================================================================
每个策略独立进化, 只看到自己的诊断报告, 每项修改必须引用诊断中的具体问题ID。
"""
from __future__ import annotations
import json, os, re, time
from typing import Any, Dict, List, Optional
from openai import OpenAI


EVOLVER_SYSTEM_PROMPT = """\
你是虚拟筛选策略进化专家。你只看到**一个**策略的诊断报告, 对其进行定向修复。

## 进化原则
1. 只修复诊断报告中明确指出的问题 — 不要自己"觉得"哪里需要改
2. 每个修改的 diagnosis_ref 必须指向诊断报告中的具体问题ID (如 concern-1, suggestion-2)
3. 保留策略中未被诊断报告批评的部分 — 不要"顺手优化"
4. 如果某个concern是根本性的(如靶点错配), 标注为不可修复
5. 第一版只允许以下action_type: library_preparation, protein_preparation,
   binding_site_detection, physicochemical_filtering, diversity_selection,
   molecular_docking, interaction_analysis, admet_filtering,
   molecular_dynamics, final_ranking, report_generation,
   similarity_screening, pharmacophore_screening, shape_matching,
   fragment_screening, consensus_scoring, target_structure_prediction
6. 暂未接入工具可以保留为科学路线，但必须保持 execution_status 和 missing_capabilities；禁止把 future capability 伪装成 currently_executable
7. 禁止插入共价、FEP、生成式设计或人工目视检查步骤，除非诊断明确要求并且标注为 future capability

## 输出JSON格式
{
  "strategy_name": "xxx (v2 进化版)",
  "problem_focus": "保留或修复后的核心问题",
  "diversity_axis": "保留原多样性轴",
  "execution_status": "currently_executable / partially_executable / future_capability_required",
  "missing_capabilities": [],
  "changes": [
    {
      "change_id": "chg-1",
      "diagnosis_ref": "concern-1",
      "dsl_action": "UPDATE_PARAM a-de0f3dc1 require_positive_charge true",
      "rationale": "根据concern-1: 未包含阳离子基团, 可能遗漏关键π-阳离子相互作用",
      "affected_step_id": "a-de0f3dc1",
      "priority": "High"
    }
  ],
  "updated_pipeline": [...],
  "unchanged_notes": "保留了xxx的优势"
}

## DSL指令
UPDATE_PARAM <step_id> <param> <value>
ADD_PARAM <step_id> <param> <value>
INSERT_STEP AFTER <step_id> <action_type>
INSERT_STEP BEFORE <step_id> <action_type>
INSERT_STEP AT_END <action_type>
REMOVE_STEP <step_id>
REPLACE_ACTION <step_id> <new_action_type>

每个change必须引用诊断报告中的ID。如果诊断报告没有提到的问题请不要修改。
"""


class StrategyEvolver:
    def __init__(self, model="deepseek-chat", api_key=None, api_base=None,
                 temperature=0.3, max_tokens=16384):
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

    # ═══════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════

    def evolve_top_n(self, strategies: list, review_results: dict,
                     tournament_records: list, research_report: dict,
                     user_query: str, n: int = 3, prior_knowledge: str = "") -> list:
        """对TopN策略逐一进化 (隔离模式)。"""
        def _sort_key(item):
            v = item[1]
            if isinstance(v, dict):
                if "weighted_score" in v: return v["weighted_score"]
                return 100 - len(v.get("concerns", [])) * 5 - len(v.get("suggestions", [])) * 2
            return 0
        sorted_items = sorted(review_results.items(), key=_sort_key, reverse=True)
        top_names = [name for name, _ in sorted_items[:n]]
        print(f"\n  🧬 进化 Top {n} 策略: {[nm[:30] for nm in top_names]}", flush=True)

        evolved = {}
        for name in top_names:
            s = next((st for st in strategies if st["strategy_name"] == name), None)
            if not s: continue
            rr = review_results.get(name, {})
            evo = self.evolve_strategy(s, rr, research_report, user_query, prior_knowledge)
            if evo:
                evolved[name] = evo
                chg_n = len(evo.get("changes", []))
                print(f"    ✅ {name[:40]} → {chg_n}项修改", flush=True)
            else:
                print(f"    ❌ {name[:40]} 进化失败", flush=True)

        result = []
        for s in strategies:
            result.append(evolved.get(s["strategy_name"], s))
        return result

    def evolve_strategy(self, strategy: dict, diagnosis: dict,
                        research_report: dict, user_query: str,
                        prior_knowledge: str = "") -> Optional[dict]:
        """单策略隔离进化 — 只看到自己的诊断报告。"""
        prompt = self._build_evolution_prompt(strategy, diagnosis, research_report,
                                               user_query, prior_knowledge)
        # Retry loop for transient API failures (rate limiting, timeouts, etc.)
        last_error = ""
        resp = None
        for attempt in range(3):
            try:
                is_reasoner = "reasoner" in self.model.lower()
                kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                              messages=[{"role":"system","content":EVOLVER_SYSTEM_PROMPT},
                                        {"role":"user","content":prompt}])
                if not is_reasoner:
                    kwargs["temperature"] = self.temperature
                    kwargs["response_format"] = {"type":"json_object"}

                resp = self.client.chat.completions.create(**kwargs)
                break  # success, exit retry loop
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    wait = (attempt + 1) * 3
                    print(f"    ⚠️ API调用失败 (尝试{attempt+1}/3): {e}，{wait}秒后重试...",
                          flush=True)
                    time.sleep(wait)
                else:
                    print(f"    ❌ 进化LLM调用失败 (3次重试后): {last_error}", flush=True)
                    return None

        if resp is None:
            print(f"    ❌ 进化LLM调用失败 (无响应)", flush=True)
            return None

        try:
            raw = (resp.choices[0].message.content or "").strip()
            if "{" in raw:
                raw = raw[raw.find("{"):]
            parsed = self._robust_parse(raw)

            if not parsed or "changes" not in parsed:
                return None

            # 质量校验
            changes = parsed.get("changes", [])
            if not changes:
                return None
            # 检查是否有空的 diagnosis_ref
            unlinked = [c for c in changes if not c.get("diagnosis_ref")]
            if unlinked:
                print(f"    ⚠️ {len(unlinked)}项修改缺少diagnosis_ref, 丢弃", flush=True)
                changes = [c for c in changes if c.get("diagnosis_ref")]
                parsed["changes"] = changes

            # 确保 updated_pipeline 存在
            if "updated_pipeline" not in parsed or not parsed["updated_pipeline"]:
                parsed["updated_pipeline"] = strategy.get("pipeline", [])
            # 注入 step_id (如果缺少)
            import uuid as _uuid
            for st in parsed["updated_pipeline"]:
                if not st.get("step_id"):
                    st["step_id"] = f"a-{_uuid.uuid4().hex[:8]}"

            parsed.setdefault("strategy_name",
                              strategy.get("strategy_name", "?") + " (v2 进化版)")
            for key in (
                "problem_focus", "target_evidence_refs", "user_requirement_coverage",
                "diversity_axis", "risk_level", "why_this_strategy_fits_target",
                "execution_status", "required_capabilities", "missing_capabilities",
                "target_profile", "approach_category", "strategy_tagline",
                "applicability_conditions",
            ):
                if key not in parsed and key in strategy:
                    parsed[key] = strategy[key]
            return parsed

        except Exception as e:
            print(f"    ❌ 进化结果解析失败: {e}", flush=True)
            return None

    # ═══════════════════════════════════════════
    # Prompt 构建
    # ═══════════════════════════════════════════

    def _build_evolution_prompt(self, strategy: dict, diagnosis: dict,
                                 research_report: dict, user_query: str,
                                 prior_knowledge: str = "") -> str:
        parts = []

        # 任务
        if user_query:
            parts.append(f"## 用户任务\n{user_query}\n")

        # 先验知识
        if prior_knowledge and prior_knowledge.strip():
            parts.append(f"## 先验知识\n{prior_knowledge.strip()}\n")

        # 靶点背景 (只用结构化执行摘要, 不用全文)
        summary = research_report.get("executive_summary", "")
        if summary:
            # JSON格式 → 直接展示结构化数据
            parts.append(f"## 靶点: {research_report.get('target_name','?')} "
                         f"({research_report.get('gene_symbol','?')})\n\n{summary}")
        else:
            parts.append(f"## 靶点: {research_report.get('target_name','?')} "
                         f"({research_report.get('gene_symbol','?')})\n\n"
                         f"{(research_report.get('full_report_text', '') or '')[:2000]}")

        # 策略底稿
        parts.append(self._fmt_blueprint(strategy))

        # 结构化诊断报告 (带编号)
        parts.append(self._fmt_diagnosis(diagnosis))

        parts.append("## 任务\n请基于诊断报告逐项修复, 输出完整JSON。每个change的diagnosis_ref必须引用上面诊断中的ID。")
        return "\n\n".join(parts)

    @staticmethod
    def _fmt_blueprint(s: dict) -> str:
        lines = [f"## 策略底稿\n名称: {s.get('strategy_name','?')}\n"
                 f"方法: {s.get('approach_category','?')}\n"
                 f"标签: {s.get('strategy_tagline','')}\n"
                 f"问题焦点: {s.get('problem_focus','')}\n"
                 f"多样性轴: {s.get('diversity_axis','')}\n"
                 f"执行状态: {s.get('execution_status','')}\n"
                 f"缺失能力: {s.get('missing_capabilities', [])}\n"
                 f"证据引用: {s.get('target_evidence_refs', [])}\n"
                 f"存活: {s.get('survival_estimate','?')}\n"]
        for st in s.get("pipeline", s.get("pipeline_steps", [])):
            sid = st.get("step_id", f"step_{st.get('step_number','?')}")
            at = st.get("action_type", "?")
            an = st.get("action_name", st.get("step_name", "?"))
            desc = st.get("description", st.get("action", ""))
            params = st.get("parameters", {})
            lines.append(f"\n  [{sid}] {an} ({at})")
            lines.append(f"    desc: {desc[:150]}")
            if params:
                lines.append(f"    params: {json.dumps(params, ensure_ascii=False)[:120]}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_diagnosis(d: dict) -> str:
        """格式化诊断报告: 每个concern/suggestion分配ID, 保留DSL action。"""
        lines = ["## 诊断报告 (每项有唯一ID, 修改时必须引用)"]

        concerns = d.get("concerns", [])
        suggestions = d.get("suggestions", [])

        if concerns:
            lines.append(f"\n### 问题 ({len(concerns)}条)")
            for i, c in enumerate(concerns, 1):
                sid = c.get("step_id", c.get("step_number", "?"))
                severity = c.get("severity", "?")
                issue = c.get("issue", str(c))
                consequence = c.get("consequence", "")
                lines.append(f"\n  [ID: concern-{i}] 严重度={severity} | step={sid}")
                lines.append(f"    问题: {issue}")
                if consequence:
                    lines.append(f"    后果: {consequence}")

        if suggestions:
            lines.append(f"\n### 改进建议 ({len(suggestions)}条)")
            for i, s in enumerate(suggestions, 1):
                priority = s.get("priority", "?")
                action = s.get("action", "")  # DSL 指令!
                rationale = s.get("rationale", "")
                feasibility = s.get("feasibility", "?")
                step_id = s.get("step_id", s.get("step_number", "?"))
                lines.append(f"\n  [ID: suggestion-{i}] 优先级={priority} | "
                             f"可行性={feasibility} | step={step_id}")
                if action:
                    lines.append(f"    DSL: {action}")
                if rationale:
                    lines.append(f"    原因: {rationale}")

        # 维度强弱
        strengths = d.get("strengths", [])
        weaknesses = d.get("weaknesses", [])
        if strengths:
            lines.append(f"\n### 优势维度\n  " + ", ".join(strengths))
        if weaknesses:
            lines.append(f"\n### 劣势维度\n  " + ", ".join(weaknesses))

        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # 辅助
    # ═══════════════════════════════════════════

    @staticmethod
    def _extract_all_weaknesses(review_result: dict, verdict_weaknesses: list) -> list:
        """向后兼容: 提取弱点列表 (保留 v4 diagnosis format 的 DSL)"""
        weaknesses = []
        for c in review_result.get("concerns", []):
            if isinstance(c, dict):
                weaknesses.append({
                    "source": f"diagnosis-{c.get('action_type','?')}",
                    "issue": c.get("issue", str(c)),
                    "severity": c.get("severity", "Warning").lower(),
                })
        for s in review_result.get("suggestions", []):
            if isinstance(s, dict):
                weaknesses.append({
                    "source": f"diagnosis-{s.get('priority','?')}",
                    "issue": s.get("action", s.get("rationale", str(s))),  # 保留DSL!
                    "severity": "minor",
                })
        for w in review_result.get("weaknesses", []):
            weaknesses.append({"source": "维度统计", "issue": w, "severity": "minor"})
        return weaknesses

    @staticmethod
    def _extract_verdict_weaknesses(strategy_name: str, records: list) -> list:
        """从判例中提取裁判对特定策略的弱点 (保留向后兼容)。"""
        w = []
        for rec in records:
            v = rec.get("verdict", {})
            for side, key in [("A", "suggestions_a"), ("B", "suggestions_b")]:
                if rec.get(f"strategy_{side.lower()}") == strategy_name:
                    for s in v.get(key, []):
                        w.append(s)
        return w

    @staticmethod
    def _robust_parse(raw: str) -> Dict[str, Any]:
        try: return json.loads(raw)
        except json.JSONDecodeError: pass
        c = raw
        if c.startswith("```"): c = re.sub(r'^```\w*\n', '', c); c = re.sub(r'\n```$', '', c)
        try: return json.loads(c)
        except json.JSONDecodeError: pass
        s, e = c.find("{"), c.rfind("}")
        if s >= 0 and e > s:
            try: return json.JSONDecoder().raw_decode(c[s:e+1])[0]
            except json.JSONDecodeError: pass
        return {}
