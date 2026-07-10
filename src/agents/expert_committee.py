"""
AutoVS-Agent v3.0: 策略排名锦标赛 — 三人设独立评审
======================================================
三位评审官, 各从不同维度独立打分。每人拿到完整调研报告+用户任务。

评审官:
  1. 漏斗工程评审官 (Funnel & Engineering)  — 权重30%
  2. 需求匹配评审官 (Target & Requirement) — 权重35%
  3. 产出质量评审官 (Output & Diversity)   — 权重35%
"""

from __future__ import annotations

import json, os, re
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# Schema
# =============================================================================

class DimensionScore(BaseModel):
    name: str = Field(default="", description="子维度名称")
    score: float = Field(default=50.0, ge=0, le=100)
    weight: float = Field(default=0.25, ge=0, le=1)
    comment: str = Field(default="", description="该维度的具体评价")


class ReviewerReport(BaseModel):
    reviewer_id: str = Field(default="")
    reviewer_name: str = Field(default="")
    strategy_name: str = Field(default="")
    overall_score: float = Field(default=50.0, ge=0, le=100)
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    key_strengths: List[str] = Field(default_factory=list)
    key_weaknesses: List[str] = Field(default_factory=list)
    critical_flaws: List[str] = Field(default_factory=list, description="致命缺陷(如有则不应被采用)")
    reference_to_report: str = Field(default="", description="引用调研报告的发现")


ReviewerReport.model_rebuild()
DimensionScore.model_rebuild()


# =============================================================================
# 三人设 System Prompts
# =============================================================================

REVIEWER_FUNNEL_PROMPT = """\
你是「漏斗工程评审官」— 药物筛选管道设计专家。

你的职责: 评估虚拟筛选策略的管道设计是否合理、高效、可执行。
⚠️ 关键原则: 不仅要评"策略写了什么", 更要罚"策略缺了什么"。

🚨 分数分布强制要求:
你正在评审一组策略(约10个)。你必须拉大分数差距!
- 最好的策略应得到80-95分, 最差的应得到20-40分
- 最高分与最低分必须至少相差30分
- 不能所有策略挤在50-70分的中间区域
- 每个维度也要有明显区分: 优秀的策略在某维度得85+, 差的策略得30-
- 如果一个策略只有2-3个简单步骤且缺少关键环节, 总分不应超过50

## 评分维度 (总分100) — 每维度有5级量化标尺

### 1. 存活链完整性 (30分) — 5级标尺:
  0-20分: 无存活估算, 或某个步骤必然导致0存活
  20-40分: 有粗略估算, 但每步存活数明显不合理 (如2亿→10个)
  40-60分: 大致合理, 但缺少关键步骤的存活分析, 或某步可能意外杀光
  60-80分: 每步有合理的存活数, 但无验证机制或应急预案不足
  80-100分: 完整存活链(至少4步), 每步存活有依据, 多层应急
- 步骤数量是否足够? (<3步几乎不可能完成有效筛选)
- 阈值是否过于激进? 会不会叠加后产生意外的0存活?

### 2. 前端效率 (20分) — 5级标尺:
  0-20分: 无前端, 直接用最慢方法筛全库
  20-40分: 有前端但方法太慢 (如对整个库做精对接)
  40-60分: 用了中等速度方法 (简单对接), 但无更快的预筛
  60-80分: 前端用了快速方法(理化/药效团), 但过滤幅度不够大
  80-100分: 快速预筛(理化+简单指纹)砍掉90%+, 逐级加速

### 3. 后端精度 (25分) — 5级标尺:
  0-20分: 无任何验证步骤, 仅一步粗筛
  20-40分: 有验证但方法不精确 (如只用简单打分)
  40-60分: 用了对接但无后续验证(MD/MM-GBSA/FEP), 无阳性对照
  60-80分: 有精对接+一种验证方法, 但无阳性对照
  80-100分: 精对接+MD/FEP验证+阳性对照+诱饵分子评估

### 4. 工具与资源 (15分) — 5级标尺:
  0-20分: 工具不明确或不可获取, 资源需求离谱
  20-40分: 工具定义不清, 资源估算缺失
  40-60分: 工具明确但多依赖商业许可证, 或时间估算明显偏短
  60-80分: 工具合理(部分开源), 资源估算大致正确
  80-100分: 优先使用开源工具, 商业软件仅用于必要步骤, 资源估算精确

### 5. 应急预案 (10分) — 5级标尺:
  0-20分: 完全无应急方案
  20-40分: 有模糊的"调整参数"但无具体方案
  40-60分: 提到放宽阈值, 但未说明放宽到多少
  60-80分: 有具体的放宽方案和数值, 但只有一条回退路径
  80-100分: 多条回退路径(放宽阈值/扩大库/替代方法), 每条有具体数值

## ⚠️ 强制检查清单 (每缺一项, 相关维度扣5-10分)
□ 策略是否定义了初始库来源和大小?
□ 是否每步都有明确的存活数估算(不能所有策略用同一个数字)?
□ 是否有阳性对照或诱饵分子验证步骤?
□ 是否有具体的应急预案(存活<10时的具体回退操作)?
□ 是否评估了计算资源需求(CPU/GPU/时间)?
□ 工具是否明确标注了版本号和开源/商业属性?
□ 是否在第1步就砍掉>90%的库(前端效率)?

## 输出格式
严格输出JSON:
{
  "overall_score": 75,
  "dimension_scores": [
    {"name": "存活链完整性", "score": 80, "weight": 0.30, "comment": "具体评价+扣分原因"},
    {"name": "前端效率", "score": 70, "weight": 0.20, "comment": "具体评价+扣分原因"},
    {"name": "后端精度", "score": 75, "weight": 0.25, "comment": "具体评价+扣分原因"},
    {"name": "工具与资源", "score": 70, "weight": 0.15, "comment": "具体评价+扣分原因"},
    {"name": "应急预案", "score": 80, "weight": 0.10, "comment": "具体评价+扣分原因"}
  ],
  "key_strengths": ["优势1", "优势2"],
  "key_weaknesses": ["劣势1", "劣势2"],
  "critical_flaws": [],
  "reference_to_report": "引用调研报告的具体发现"
}

强制分布提醒: 你评审的多个策略之间, 每个维度的分数必须有明显区分。禁止所有策略给相同或相近的分数!
如果你认为策略有致命缺陷(如必然导致0存活或<3步), 请填入critical_flaws, overall_score应<50。
"""

REVIEWER_REQUIREMENT_PROMPT = """\
你是「需求匹配评审官」— 靶点生物学与用户需求对齐专家。

你的职责: 评估策略是否真正对症下药, 是否满足用户的所有约束条件。
⚠️ 关键原则: 不仅要评"写了什么", 更要罚"缺了什么"和"错配了什么"。

🚨 分数分布强制要求:
你正在评审一组策略(约10个)。你必须拉大分数差距!
- 最好的策略应得到80-95分, 最差的应得到20-40分
- 最高分与最低分必须至少相差30分
- 如果策略完全忽略了用户的某个核心要求("不要X"或"必须Y"), 总分直接<40
- 如果策略与靶点类型严重错配(如PPI用Ro5), 总分直接<30

## 评分维度 (总分100) — 每维度有5级量化标尺

### 1. 靶点适配性 (35分) — 5级标尺:
  0-20分: 方法与靶点类型完全不匹配 (如PPI用Ro5, 别构用正构盒子, apo结构用刚性对接且不说明)
  20-40分: 方法大致可用但未适配靶点特性 (如未考虑口袋极性/柔性)
  40-60分: 方法选择正确, 但未利用靶点特有的结构特征(关键残基/口袋形状)
  60-80分: 正确利用了结构特征, 但未考虑口袋柔性/诱导契合等动态因素
  80-100分: 完美适配: 方法+口袋特征+关键残基+柔性因素+选择性残基, 全被考虑
- 常见错配:
  * PPI靶点用传统Ro5过滤 → 扣20+
  * 激酶忽略铰链区氢键 → 扣15+
  * apo结构用刚性对接不说明 → 扣15+
  * 别构抑制剂使用正构对接盒子 → 扣25+

### 2. 用户约束满足 (30分) — 5级标尺:
  0-20分: 完全忽略用户的核心要求
  20-40分: 提到用户要求在rationale中, 但策略步骤中未体现
  40-60分: 部分满足了约束, 但关键约束未处理
  60-80分: 主要约束得到处理, 但实现方式不够具体
  80-100分: 所有约束都有对应的具体步骤, 且有验证机制
- 逐条对比用户query中的每个要求, 缺一条扣10分

### 3. 数据利用率 (20分) — 5级标尺:
  0-20分: 完全无视调研报告数据, 使用泛化默认值
  20-40分: 引用了报告但阈值/参数与报告数据不一致
  40-60分: 阈值大致匹配报告数据, 但未引用最关键的数值(IC50/PDB ID)
  60-80分: 正确引用了关键数据, 但忽略了一些辅助数据(SAR/选择性残基)
  80-100分: 全面利用报告数据: IC50→阈值, PDB ID→结构, MW/LogP→过滤, 残基→PLIP

### 4. 生物学合理性 (15分) — 5级标尺:
  0-20分: 策略有根本性的生物学错误
  20-40分: 大方向对但细节有误(如关键残基搞错)
  40-60分: 正确但在某些假设上缺乏文献支持
  60-80分: 生物学基础扎实, 但未考虑靶点的特殊调控机制
  80-100分: 完美: 考虑了突变体/PTM/内源性配体竞争/信号通路反馈

## ⚠️ 强制检查清单 (每缺一项, 相关维度扣5-10分)
□ 用户说的每一个"不要/禁止/避开" — 策略是否有对应的排除步骤? (缺→扣10)
□ 用户说的每一个"要求/必须" — 策略是否包含? (缺→扣10)
□ 策略的关键步骤是否引用了调研报告中的具体PDB ID? (未引用→扣5)
□ 过滤阈值(MW/LogP/IC50)是否基于调研报告的真实数据而非泛化默认值? (否→扣8)
□ 是否提到了关键残基(调研报告中的氢键/疏水/选择性残基)? (未提→扣5)
□ 如果靶点有共晶结构 — 策略是否推荐使用而非建议同源建模? (用错→扣10)
□ 如果靶点无实验结构 — 策略是否正确说明替代方案(同源建模/AF2)? (未说明→扣5)

## 输出格式
严格输出JSON:
{
  "overall_score": 75,
  "dimension_scores": [
    {"name": "靶点适配性", "score": 80, "weight": 0.35, "comment": "具体评价+扣分原因"},
    {"name": "用户约束满足", "score": 70, "weight": 0.30, "comment": "具体评价+扣分原因"},
    {"name": "数据利用率", "score": 75, "weight": 0.20, "comment": "具体评价+扣分原因"},
    {"name": "生物学合理性", "score": 80, "weight": 0.15, "comment": "具体评价+扣分原因"}
  ],
  "key_strengths": ["优势1", "优势2"],
  "key_weaknesses": ["劣势1", "劣势2"],
  "critical_flaws": [],
  "reference_to_report": "引用调研报告的具体发现"
}

强制分布提醒: 不同策略的分数必须有明显差异。如果多个策略看起来差不多, 在"用户约束满足"和"数据利用率"维度拉开差距。
如果策略与靶点/用户需求严重不匹配, 请填入critical_flaws, overall_score应<50。
"""

REVIEWER_OUTPUT_PROMPT = """\
你是「产出质量评审官」— 筛选结果质量与多样性评估专家。

你的职责: 评估策略最终可能产出什么样的分子集合, 质量如何, 多样性如何。
⚠️ 关键原则: 好的策略必须产出多样化的高质量hit, 而非几个相似的分子。

🚨 分数分布强制要求:
你正在评审一组策略(约10个)。你必须拉大分数差距!
- 最好的策略应得到80-95分, 最差的应得到25-45分
- 最高分与最低分必须至少相差30分
- 多数策略在"化学空间覆盖"和"骨架多样性"上很弱——这很正常, 利用这些维度拉开差距
- 只有真正考虑了多样性保护机制的策略, 才应该在多样性维度上得到>70分

## 评分维度 (总分100) — 每维度有5级量化标尺

### 1. 化学空间覆盖 (25分) — 5级标尺:
  0-20分: 初始库未指定或极小(<1万), 无任何多样性保护
  20-40分: 库有定义但太小(<10万)或无多样性步骤
  40-60分: 库合理(>10万), 但筛选过程会严重收窄化学空间
  60-80分: 库充分(>100万), 但没有主动的多样性保护机制
  80-100分: 库充分+多样性筛选(聚类/Murcko scaffold)+主动探索多区域化学空间

### 2. 骨架多样性 (25分) — 5级标尺:
  0-20分: 策略必然导致单一骨架(如只筛选已知配体类似物)
  20-40分: 可能产出少量骨架, 但无多样性保护步骤
  40-60分: 有基本的物理化学性质多样性, 但无骨架级别的多样性
  60-80分: 有骨架多样性意识, 但未使用具体工具(如Murcko聚类)
  80-100分: 明确的骨架多样性策略: 聚类+代表性挑选+骨架新颖性评分

### 3. Hit发现概率 (30分) — 5级标尺:
  0-20分: 策略设置导致命中概率接近于零
  20-40分: 阈值设置不合理(过严或过宽), 命中概率极低
  40-60分: 阈值大致合理, 但未用已知配体验证, 命中率难以评估
  60-80分: 用已知配体的活性范围设定了阈值, 命中率有依据
  80-100分: 阈值基于SAR数据精调+已知配体作为阳性对照验证+假阳性评估

### 4. 新颖性潜力 (20分) — 5级标尺:
  0-20分: 策略完全在已知配体化学空间内搜索
  20-40分: 有微弱的新颖性机会, 但主要照搬已知骨架
  40-60分: 有探索意愿但无具体机制(如只说"寻找新骨架")
  60-80分: 有具体的新颖性步骤(片段筛选/生成式AI/DEL), 但实现粗略
  80-100分: 专门的新颖性策略: de novo设计/FBDD/DEL+明确的多样性目标

## ⚠️ 强制检查清单 (每缺一项, 相关维度扣5-10分)
□ 是否指定了初始库的大小和来源? (未指定→扣8)
□ 是否有主动的化学空间多样性保护机制? (如聚类/指纹去重) (无→扣8)
□ 是否有骨架级别的多样性考虑? (未提Murcko/Bemis-Murcko→扣5)
□ 最终存活数×命中率的真实hit数量是否>0? (估算不合理→扣10)
□ 是否有发现全新骨架的机会?还是只在已知空间? (仅限于已知→扣5)
□ 是否考虑了假阳性风险? (未提PAINS/BRENK/聚集→扣5)
□ 是否有实验验证方案? (完全无→扣5)

## 输出格式
严格输出JSON:
{
  "overall_score": 75,
  "dimension_scores": [
    {"name": "化学空间覆盖", "score": 80, "weight": 0.25, "comment": "具体评价+扣分原因"},
    {"name": "骨架多样性", "score": 70, "weight": 0.25, "comment": "具体评价+扣分原因"},
    {"name": "Hit发现概率", "score": 75, "weight": 0.30, "comment": "具体评价+扣分原因"},
    {"name": "新颖性潜力", "score": 70, "weight": 0.20, "comment": "具体评价+扣分原因"}
  ],
  "key_strengths": ["优势1", "优势2"],
  "key_weaknesses": ["劣势1", "劣势2"],
  "critical_flaws": [],
  "reference_to_report": "引用调研报告的具体发现"
}

强制分布提醒: 化学空间覆盖和骨架多样性是两个最容易被忽略的维度, 多数策略在这些维度上不应得高分。
如果策略几乎不可能产生有价值的hit, 请填入critical_flaws, overall_score应<50。
"""


# =============================================================================
# Reviewer Configs
# =============================================================================

REVIEWER_CONFIGS = [
    {
        "id": "funnel",
        "name": "漏斗工程评审官",
        "weight": 0.30,
        "prompt": REVIEWER_FUNNEL_PROMPT,
    },
    {
        "id": "requirement",
        "name": "需求匹配评审官",
        "weight": 0.35,
        "prompt": REVIEWER_REQUIREMENT_PROMPT,
    },
    {
        "id": "output",
        "name": "产出质量评审官",
        "weight": 0.35,
        "prompt": REVIEWER_OUTPUT_PROMPT,
    },
]


# =============================================================================
# TournamentReviewer
# =============================================================================

class TournamentReviewer:
    """策略排名锦标赛 — 三人设独立评审。

    每个策略由三位评审官独立打分, 产生综合评分和详细报告。
    """

    def __init__(self, model="deepseek-chat", api_key=None, api_base=None,
                 temperature=0.2, max_tokens=4096):
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
    # 主入口: 对单策略做三轮独立评审
    # =========================================================================

    def review_strategy(self, strategy: dict, research_report: dict,
                        user_query: str = "") -> Dict[str, Any]:
        """三位评审官各自独立评审一个策略。

        Returns:
            {
                "strategy_name": "...",
                "reports": [
                    {"reviewer_id": "funnel", "reviewer_name": "...",
                     "overall_score": 75, ...},
                    ...
                ],
                "weighted_score": 75.5,  # 加权综合分
                "all_critical_flaws": [...],
            }
        """
        # 构建上下文: 调研报告 + 用户任务
        context = self._build_context(research_report, user_query, strategy)

        # 并行评审
        from concurrent.futures import ThreadPoolExecutor, as_completed
        reports = []
        worst_score = 100

        print(f"  📝 评审 [{strategy.get('strategy_name','?')[:40]}]...", flush=True)

        def _review_one(cfg):
            return self._call_reviewer(cfg, context, strategy)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_review_one, cfg): cfg for cfg in REVIEWER_CONFIGS}
            for future in as_completed(futures):
                report, err = future.result()
                if err:
                    print(f"    ❌ [{futures[future]['name']}] 失败: {err}", flush=True)
                else:
                    reports.append(report)
                    worst_score = min(worst_score, report["overall_score"])
                    print(f"    ✅ [{report['reviewer_name']}] {report['overall_score']:.0f}分",
                          flush=True)

        # 综合评分
        if reports:
            weighted = sum(r["overall_score"] * self._get_weight(r["reviewer_id"])
                          for r in reports)
        else:
            weighted = 50.0

        all_flaws = []
        for r in reports:
            all_flaws.extend(r.get("critical_flaws", []))

        return {
            "strategy_name": strategy.get("strategy_name", "?"),
            "reports": reports,
            "weighted_score": round(weighted, 1),
            "all_critical_flaws": all_flaws,
        }

    # =========================================================================
    # 评审后校准: 让评审官看到全局分布后重新打分, 拉开差距
    # =========================================================================

    def calibrate_all(self, review_results: dict, strategies: list,
                      research_report: dict, user_query: str = "") -> dict:
        """对所有策略做校准, 每个评审官一次额外LLM调用, 拉开分数分布。

        Returns: 校准后的 review_results (原地修改)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        strategy_list = list(review_results.keys())
        if len(strategy_list) < 3:
            return review_results

        # 构建校准 prompt
        calib_context = self._build_calibration_context(
            review_results, strategies, research_report, user_query
        )

        def _calibrate_one(cfg):
            try:
                new_scores = self._call_calibration(cfg, calib_context)
                return cfg["id"], new_scores, ""
            except Exception as e:
                return cfg["id"], {}, str(e)

        print(f"\n  🔧 评审后校准 (拉开分布)...", flush=True)
        all_new_scores = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_calibrate_one, cfg): cfg
                      for cfg in REVIEWER_CONFIGS}
            for future in as_completed(futures):
                rid, new_scores, err = future.result()
                if err:
                    print(f"    ❌ [{rid}] 校准失败: {err}", flush=True)
                elif new_scores:
                    all_new_scores[rid] = new_scores
                    old_range = max(s.get("overall_score", 50) for s in
                        [r["reports"][j] for r in review_results.values()
                         for j, rep in enumerate(r.get("reports", [])) if rep["reviewer_id"] == rid]
                    ) if review_results else 50
                    print(f"    ✅ [{rid}] 校准完成", flush=True)

        # 应用校准
        if all_new_scores:
            for name, rr in review_results.items():
                for i, rep in enumerate(rr.get("reports", [])):
                    rid = rep.get("reviewer_id", "")
                    if rid in all_new_scores and name in all_new_scores[rid]:
                        rr["reports"][i]["overall_score"] = float(
                            all_new_scores[rid][name]
                        )
                # 重新计算加权分
                rr["weighted_score"] = round(
                    sum(r["overall_score"] * self._get_weight(r["reviewer_id"])
                        for r in rr.get("reports", [])), 1
                )

        return review_results

    def _build_calibration_context(self, review_results, strategies, report, query):
        parts = ["## 校准任务\n你刚刚评审了以下策略, 请检查你的评分是否有足够的区分度。\n"]

        # 当前分数分布
        for cfg in REVIEWER_CONFIGS:
            rid = cfg["id"]
            reviewer_scores = []
            for name, rr in review_results.items():
                for rep in rr.get("reports", []):
                    if rep.get("reviewer_id") == rid:
                        reviewer_scores.append((name, rep.get("overall_score", 50)))
            if reviewer_scores:
                score_vals = [s for _, s in reviewer_scores]
                parts.append(
                    f"### {cfg['name']} 当前评分\n"
                    f"范围: {min(score_vals):.0f} - {max(score_vals):.0f} "
                    f"(差距{max(score_vals)-min(score_vals):.0f}分)\n"
                )
                for name, sc in sorted(reviewer_scores, key=lambda x: x[1], reverse=True):
                    parts.append(f"  {sc:.0f} {name[:50]}")

        parts.append("""
## ⚠️ 你的分数太集中了!

请对每个策略重新打分, 必须满足:
- 最高分与最低分至少相差30分
- 最好的策略应得 80-95 分
- 最差的策略应得 20-40 分
- 中间策略均匀分布

输出JSON: {"策略名1": 85, "策略名2": 72, ...}
只输出JSON, 不要其他文字。
""")
        return "\n\n".join(parts)

    def _call_calibration(self, cfg, context):
        try:
            kwargs = dict(model=self.model, max_tokens=2048,
                          messages=[{"role":"system","content":cfg["prompt"]},
                                    {"role":"user","content":context}])
            if "reasoner" not in self.model.lower():
                kwargs["temperature"] = 0.3
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if "{" in raw:
                raw = raw[raw.find("{"):]
            parsed = json.loads(raw)
            # 清理: 只保留数字值
            result = {}
            for k, v in parsed.items():
                try:
                    result[k] = max(0, min(100, float(v)))
                except (ValueError, TypeError):
                    pass
            return result
        except Exception:
            return {}

    @staticmethod
    def _get_weight(reviewer_id: str) -> float:
        for cfg in REVIEWER_CONFIGS:
            if cfg["id"] == reviewer_id:
                return cfg["weight"]
        return 0.33

    # =========================================================================
    # 上下文构建
    # =========================================================================

    def _build_context(self, research_report: dict, user_query: str,
                       strategy: dict) -> str:
        parts = []

        if user_query:
            parts.append(f"## 用户原始任务\n{user_query}\n")

        # 调研报告关键信息
        target_name = research_report.get("target_name", "Unknown")
        gene = research_report.get("gene_symbol", "")
        organism = research_report.get("target_organism", "")
        uniprot = research_report.get("uniprot_id", "")
        parts.append(f"## 靶点信息\n"
                     f"靶点: {target_name} | 基因: {gene} | "
                     f"物种: {organism} | UniProt: {uniprot}\n")

        # 调研报告全文 (完整版, 不做截断!)
        full_text = research_report.get("full_report_text", "")
        if full_text:
            parts.append(f"## 调研报告全文\n{full_text}\n")

        # API数据摘要
        api_sources = research_report.get("api_sources", [])
        if api_sources:
            parts.append(f"## API数据来源\n{', '.join(api_sources)}\n")

        # 策略原文
        parts.append(self._fmt_strategy(strategy))

        return "\n\n".join(parts)

    @staticmethod
    def _fmt_strategy(s: dict) -> str:
        steps_text = ""
        for st in s.get("pipeline_steps", []):
            steps_text += (
                f"  Step{st.get('step_number','?')}: {st.get('step_name','?')} "
                f"[{st.get('tool','?')}]\n"
                f"    操作: {st.get('action','?')[:200]}\n"
                f"    指标: {st.get('metric','?')} | 阈值: {st.get('threshold','?')}\n"
            )
        return f"""## 待评审策略

**名称**: {s.get('strategy_name','?')}
**标签**: {s.get('strategy_tagline','?')}
**方法**: {s.get('approach_type','?')}
**设计原理**:
{s.get('rationale','?')[:500]}

**管道步骤**:
{steps_text}
**存活估算**: {s.get('survival_estimate','?')}
**应急预案**: {s.get('contingency','?')[:200]}
**优势**: {s.get('strengths',[])}
**劣势**: {s.get('weaknesses',[])}
**预估耗时**: {s.get('estimated_runtime','?')}
**适用场景**: {s.get('suitable_when','?')}"""

    # =========================================================================
    # 单评审官 LLM 调用
    # =========================================================================

    def _call_reviewer(self, cfg: dict, context: str,
                       strategy: dict) -> tuple:
        """调用单评审官LLM。返回 (ReviewerReport_dict, error_str)。"""
        try:
            is_reasoner = "reasoner" in self.model.lower()
            msgs = [
                {"role": "system", "content": cfg["prompt"]},
                {"role": "user", "content": context},
                {"role": "system", "content": (
                    f"请对策略「{strategy.get('strategy_name','?')}」"
                    f"进行独立评审。输出完整JSON, 每个维度都要有具体评价。"
                )},
            ]
            kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                          messages=msgs)
            if not is_reasoner:
                kwargs["temperature"] = self.temperature
                kwargs["response_format"] = {"type": "json_object"}

            resp = self.client.chat.completions.create(**kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            if "{" in raw:
                raw = raw[raw.find("{"):]
            parsed = self._robust_json_parse(raw)

            # 构建标准报告
            dims = []
            for d in parsed.get("dimension_scores", []):
                dims.append(DimensionScore(
                    name=d.get("name", ""),
                    score=float(d.get("score", 50)),
                    weight=float(d.get("weight", 0.25)),
                    comment=d.get("comment", ""),
                ).model_dump())

            report = ReviewerReport(
                reviewer_id=cfg["id"],
                reviewer_name=cfg["name"],
                strategy_name=strategy.get("strategy_name", "?"),
                overall_score=float(parsed.get("overall_score", 50)),
                dimension_scores=dims,
                key_strengths=parsed.get("key_strengths", []),
                key_weaknesses=parsed.get("key_weaknesses", []),
                critical_flaws=parsed.get("critical_flaws", []),
                reference_to_report=parsed.get("reference_to_report", ""),
            ).model_dump()

            # 有效性校验
            if report["overall_score"] < 0 or report["overall_score"] > 100:
                report["overall_score"] = max(0, min(100, report["overall_score"]))
            if not report["dimension_scores"]:
                report["dimension_scores"] = [DimensionScore(
                    name="综合", score=report["overall_score"],
                    weight=1.0, comment="无细分维度"
                ).model_dump()]

            return report, ""

        except Exception as e:
            # Fallback
            fallback = ReviewerReport(
                reviewer_id=cfg["id"],
                reviewer_name=cfg["name"],
                strategy_name=strategy.get("strategy_name", "?"),
                overall_score=50.0,
                key_strengths=[],
                key_weaknesses=[f"评审失败: {str(e)[:100]}"],
                critical_flaws=[],
                reference_to_report="",
            ).model_dump()
            return fallback, str(e)

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


# =============================================================================
# 兼容旧接口 (供 test_tournament.py 过渡)
# =============================================================================

# 保留旧类名作别名
RedTeamReviewer = TournamentReviewer


def red_team_debate_node(state: dict) -> dict:
    """LangGraph 节点兼容。"""
    from datetime import datetime as dt
    ts = state.get("tournament_state", {})
    pairings = list(ts.get("pairings_queue", []))
    if not pairings:
        return {"pipeline_stage": "tournament",
                "event_log": ["[Tournament] No more pairings."]}
    pair = pairings.pop(0)
    strategies = {s["strategy_name"]: s for s in state.get("candidate_strategies", [])}
    sa, sb = strategies.get(pair[0], {}), strategies.get(pair[1], {})
    if not sa or not sb:
        return {"tournament_state": {**ts, "pairings_queue": pairings}}
    reviewer = TournamentReviewer()
    # 对两个策略分别评审
    ra = reviewer.review_strategy(sa, state.get("target_profile", {}))
    rb = reviewer.review_strategy(sb, state.get("target_profile", {}))
    return {
        "pipeline_stage": "tournament",
        "tournament_state": {**ts, "pairings_queue": pairings},
        "event_log": [f"[{dt.now().isoformat()}] [Review] {pair[0][:30]} vs {pair[1][:30]}"],
        "_current_debate_pair": pair,
        "_current_review_a": ra,
        "_current_review_b": rb,
    }
