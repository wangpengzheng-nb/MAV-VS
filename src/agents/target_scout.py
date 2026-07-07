"""
AutoVS-Agent: Strategy Agent (情报智能体) — 首发核心
======================================================
职责 (Step 1: 战前侦察):
  - 接收靶点信息 (TargetInfo)，通过 LLM 生物学知识推理
  - 输出《动态药化过滤协议》(Rulebook / DynamicFilterProtocol)
  - 根据靶点类型 (激酶/GPCR/PPI/PROTAC) 智能切换 Ro5 ↔ bRo5
  - 提取必需药效团特征，列出毒性/PAINS 黑名单

下游消费方:
  - Step 5 MedChem Committee 严格按此 Rulebook 执行绝对值淘汰

架构:
  RulebookSchema (Pydantic) → StrategyAgent.generate_protocol() → scouting_node()
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# 1. 结构化输出模型 (Pydantic Schema)
# =============================================================================

class RulebookSchema(BaseModel):
    """Strategy Agent 产出的《动态药化过滤协议》结构化 Schema。

    该 Schema 完美映射 MACVSState 中的 DynamicFilterProtocol TypedDict。
    每个 Field 的 description 即为指导 LLM 输出正确数值的微观 Prompt。

    核心原则:
      - 动态阈值: 绝不机械套用 Lipinski，而根据靶点类型 (激酶/GPCR/PPI/PROTAC) 智能调节
      - 科学依据: 每个数值都应有文献或结构生物学依据
      - 可执行性: 下游 RDKit 脚本可直接消费所有 SMARTS/数值字段
    """

    # =========================================================================
    # 物理化学规则 (动态阈值)
    # =========================================================================

    mw_range: List[float] = Field(
        default_factory=lambda: [150.0, 500.0],
        description=(
            "分子量允许范围 [min, max]，单位 Dalton。\n"
            "- 经典激酶/GPCR 靶点: [150, 500] (Ro5)\n"
            "- PPI 大口袋靶点 (结合面 >800 Å²): [150, 1000] (bRo5)\n"
            "- PROTAC/分子胶: [300, 1200]\n"
            "- 中枢神经靶点 (需穿透 BBB): [150, 450]\n"
            "下限不低于 150 (太小的分子缺乏足够相互作用)，"
            "上限根据靶点口袋体积和溶剂暴露面积动态设定。"
        ),
    )

    logp_range: List[float] = Field(
        default_factory=lambda: [-2.0, 5.0],
        description=(
            "脂水分配系数 cLogP 允许范围 [min, max]。\n"
            "- 经典靶点: [-2, 5] (Ro5)\n"
            "- PPI/bRo5: [-2, 7]\n"
            "- 中枢神经靶点: [1, 4] (过极性不能穿透 BBB，过脂溶性会非特异性结合)\n"
            "- 肾清除靶点: [-3, 3] (偏极性利于溶解和排泄)\n"
            "LogP < -2 分子极性过高难穿透细胞膜，LogP > 7 脂溶性过强导致溶解性差和非特异性毒性。"
        ),
    )

    hbd_max: int = Field(
        default=5,
        description=(
            "氢键供体 (HBD) 数量上限。\n"
            "- Ro5 标准: ≤5\n"
            "- bRo5/PPI: ≤6 (大环肽等特殊骨架允许更多)\n"
            "- 激酶铰链区靶点: ≤4 (过多 HBD 会与水竞争铰链区结合)\n"
            "HBD = NH + OH 计数。每增加一个 HBD，血脑屏障穿透率下降约 10 倍。"
        ),
    )

    hba_max: int = Field(
        default=10,
        description=(
            "氢键受体 (HBA) 数量上限。\n"
            "- Ro5 标准: ≤10\n"
            "- bRo5/PPI: ≤12\n"
            "- 激酶 ATP 口袋: ≤8 (需与铰链区残基形成精确氢键)\n"
            "HBA = N + O 计数。HBA 过多可能导致脱靶效应，但 PPI 靶点的大型结合面"
            "通常需要更多极性相互作用。"
        ),
    )

    rotatable_bonds_max: int = Field(
        default=10,
        description=(
            "可旋转键数量上限。\n"
            "- 经典靶点: ≤10\n"
            "- bRo5: ≤15\n"
            "- PROTAC: ≤20 (Linker 区域天然需要柔性)\n"
            "可旋转键每增加一个，口服生物利用度平均下降约 0.5%。"
            "但过度刚性 (≤2) 的分子可能无法诱导契合。"
        ),
    )

    tpsa_range: List[float] = Field(
        default_factory=lambda: [20.0, 140.0],
        description=(
            "拓扑极性表面积 (TPSA) 允许范围 [min, max]，单位 Å²。\n"
            "- 口服药物: [20, 140]\n"
            "- 需穿透 BBB: [20, 90]\n"
            "- PPI/bRo5: [40, 200] (更大的结合面允许更高极性)\n"
            "- 抗生素/抗真菌: [20, 180] (往往需要更高极性)\n"
            "TPSA < 60 通常穿透 BBB 良好；TPSA > 140 口服吸收差但可能适用于注射给药。"
        ),
    )

    num_aromatic_rings_range: List[int] = Field(
        default_factory=lambda: [0, 5],
        description=(
            "芳香环数量允许范围 [min, max]。\n"
            "- 经典靶点: [1, 4]\n"
            "- PPI 靶点疏水口袋: [2, 6]\n"
            "- 激酶: [2, 5] (大多数激酶抑制剂含 2-4 个芳香环)\n"
            "芳香环过多 (>5) 易导致平面性过强、溶解性差和 π-π 堆积诱变风险。"
            "完全没有芳香环 (0 个) 对大多数靶点是允许的，但可能缺乏疏水锚点。"
        ),
    )

    # =========================================================================
    # 药效团需求
    # =========================================================================

    pharmacophore_required: List[str] = Field(
        default_factory=list,
        description=(
            "必需药效团特征列表。缺少任一特征的分子将被一票否决。\n"
            "使用标准药效团术语:\n"
            "  - 'hbond_donor'      氢键供体\n"
            "  - 'hbond_acceptor'   氢键受体\n"
            "  - 'hydrophobic_ring' 疏水芳环\n"
            "  - 'positive_ion'     可质子化正电中心 (如胺基 pKa>7)\n"
            "  - 'negative_ion'     可去质子化负电中心 (如羧酸 pKa<5)\n"
            "  - 'aromatic_stack'   π-π 堆积芳环\n"
            "  - 'halogen_bond'     卤键供体 (如 C-I/C-Br)\n"
            "典型场景:\n"
            "  - 激酶铰链区: ['hbond_donor', 'hbond_acceptor', 'hydrophobic_ring']\n"
            "  - Bcl-2 BH3 口袋: ['hydrophobic_ring', 'negative_ion']\n"
            "  - 蛋白酶催化位点: ['hbond_donor', 'hbond_acceptor', 'hydrophobic_ring']\n"
            "该列表应为具体靶点量身定制，切勿使用通用占位符。"
        ),
    )

    pharmacophore_optional: List[str] = Field(
        default_factory=list,
        description=(
            "加分但非必需的药效团特征列表。存在这些特征的分子在 MedChem Committee"
            "中会获得额外加分，但不具备也不会被淘汰。\n"
            "例如: 卤键供体 (如含 Br/I 的分子可与主链羰基形成额外相互作用)、"
            "额外的疏水接触区域、可与溶剂形成稳定水桥的极性基团。\n"
            "术语体系与 pharmacophore_required 一致。"
        ),
    )

    pharmacophore_excluded: List[str] = Field(
        default_factory=list,
        description=(
            "禁止出现的药效团特征列表。具有这些特征的分子将被一票否决。\n"
            "例如: 某些靶点的选择性口袋不允许带正电基团 (会与关键残基产生静电排斥)，"
            "或某些变构口袋要求分子完全不带氢键供体 (避免与主链竞争)。"
            "通常情况下此列表为空，仅在结构生物学有明确证据时填写。"
        ),
    )

    # =========================================================================
    # 子结构黑名单 (毒性/PAINS/反应性)
    # =========================================================================

    excluded_substructures: List[str] = Field(
        default_factory=list,
        description=(
            "PAINS (Pan-Assay Interference Compounds) 和 BRENK 预警子结构黑名单，"
            "以 SMARTS 字符串表示。含有任一子结构的分子将被一票否决。\n"
            "\n"
            "经典 PAINS 示例 (必须根据靶点定制):\n"
            "  - 'C1=CC=C(C=C1)C(=O)C=C'         查尔酮类 (共价修饰蛋白)\n"
            "  - 'O=C1C(=O)C=CC=C1'               醌类 (氧化还原循环)\n"
            "  - 'C1=CC=C(C=C1)N=NC2=CC=CC=C2'   偶氮苯 (光异构化假阳性)\n"
            "  - 'C1=CC=C2C(=C1)C(=O)NC2=O'      靛红酸酐 (非特异性酰化)\n"
            "  - 'O=C(N)C(=O)N'                   巴比妥类 (聚集假阳性)\n"
            "  - 'S=C(N)N'                         硫脲 (金属螯合)\n"
            "\n"
            "BRENK 毒性预警:\n"
            "  - 'N=O'  亚硝基/羟肟酸 (致突变)\n"
            "  - 'S(=O)(=O)O'  磺酸 (代谢不稳定)\n"
            "  - 'C#N'  氰基 (需评估代谢氰化物释放风险)\n"
            "\n"
            "如果靶点需要特定反应性基团 (如共价抑制剂弹头)，"
            "则仅排除与该弹头类型冲突的其他反应性基团，保留目标弹头。"
        ),
    )

    toxic_groups: List[str] = Field(
        default_factory=list,
        description=(
            "明确毒性基团 SMARTS 列表。区别于 PAINS (假阳性干扰)，"
            "此列表针对真正的体内毒性风险。\n"
            "\n"
            "常见毒性基团:\n"
            "  - 'N=O'             亚硝胺/羟肟酸 (致突变/致癌)\n"
            "  - 'c1ccccc1N'       苯胺 (代谢生成亚硝胺)\n"
            "  - 'O=P(O)(O)O'      磷酸酯 (肾毒性风险)\n"
            "  - 'C1=CC=CC=C1[N+]' 苯基重氮盐 (致癌)\n"
            "  - 'C(F)(F)F'        三氟甲基 (需评估代谢稳定性)\n"
            "  - 含重金属配位基团 (如硫醇过量)\n"
            "\n"
            "注意: 某些含氟/含氮基团在现代药物化学中广泛使用且安全，"
            "请根据具体结构上下文判断，避免过度排除。"
        ),
    )

    reactive_groups: List[str] = Field(
        default_factory=list,
        description=(
            "化学活泼性/不稳定性基团 SMARTS 列表。\n"
            "这些基团可能在生理条件下自发分解、与血浆蛋白非特异性共价结合、"
            "或在存储期间降解。\n"
            "\n"
            "常见反应性基团:\n"
            "  - 'C(=O)Cl'      酰氯 (剧烈水解)\n"
            "  - 'C(=O)OC(=O)'  酸酐 (水解不稳定)\n"
            "  - 'S(=O)(=O)Cl'  磺酰氯 (非特异性共价修饰)\n"
            "  - 'N=C=S'        异硫氰酸酯 (非特异性共价)\n"
            "  - 'C(=O)ON'      羟胺酯 (水解/重排)\n"
            "  - 'O1CCO1'       缩醛/缩酮 (胃酸不稳定)\n"
            "  - 'C=C(C=O)'     迈克尔受体 (但在共价抑制剂设计中可能是弹头)\n"
            "\n"
            "若靶点为共价抑制剂目标，必须区分\"目标弹头\"和\"非特异性反应性基团\"。"
        ),
    )

    # =========================================================================
    # 对接评分阈值
    # =========================================================================

    docking_score_min: float = Field(
        default=-7.0,
        description=(
            "最低对接分数门槛 (GNINA CNNscore 或 smina affinity, kcal/mol)。\n"
            "\n"
            "参考标准:\n"
            "  - -7.0 kcal/mol: 基础门槛 (大多数靶点适用)\n"
            "  - -8.0 kcal/mol: 高难度靶点 (如 PPI 平面大口袋)\n"
            "  - -6.0 kcal/mol: 宽松门槛 (如金属酶、共价抑制剂前体)\n"
            "  - -9.0 kcal/mol: 极高门槛 (仅用于深度精筛)\n"
            "\n"
            "分数越负越好。此阈值用于 HTVS (Step 4) 初筛，"
            "不建议在初筛阶段设得过严 (>=-9.0) 以免漏掉真阳性。"
        ),
    )

    # =========================================================================
    # 元信息
    # =========================================================================

    rule_category: Literal["Ro5", "bRo5", "custom"] = Field(
        default="Ro5",
        description=(
            "本过滤协议的类别:\n"
            "  - 'Ro5':  经典 Lipinski 成药五规则，适用于大多数酶靶点和 GPCR\n"
            "  - 'bRo5':  Beyond Rule-of-5，适用于 PPI、PROTAC、大环肽类靶点\n"
            "  - 'custom': 定制规则，适用于特殊靶点类型 (如共价抑制剂、金属酶)\n"
            "\n"
            "选择依据: 根据靶点口袋体积 (>800 Å³ → bRo5)、"
            "结合面性质 (疏水大平面 → bRo5)、"
            "已知配体的理化性质分布来决定。"
        ),
    )

    rationale: str = Field(
        default="",
        description=(
            "制定本规则的科学依据叙述。\n"
            "必须包含以下内容:\n"
            "  1. 靶点类型及对应的规则类别选择理由\n"
            "  2. 每个数值阈值的设定依据 (如: 口袋体积 X Å³ 限制了分子量上限)\n"
            "  3. 药效团需求的来源 (如: 需与铰链区残基 Xxx 形成氢键)\n"
            "  4. 特殊排除规则的说明 (如: 该靶点已知的假阳性结构类型)\n"
            "  5. 已知的阳性对照/上市药物的理化性质参考\n"
            "\n"
            "用中文撰写，科学严谨，引用文献 PMID 或结构生物学 PDB 数据作为支撑。"
            "字数控制在 500-1500 字。"
        ),
    )

    literature_refs: List[str] = Field(
        default_factory=list,
        description=(
            "支持本规则的参考文献列表。\n"
            "格式: 'PMID:12345678' 或 'DOI:10.xxxx/xxxxx' 或 'PDB:6OKK'\n"
            "至少应引用:\n"
            "  - 靶点结构确定文献 (PDB 条目)\n"
            "  - 已知配体的 SAR 研究\n"
            "  - 相关药化规则参考 (如 bRo5 适用性的文献)\n"
            "建议 3-10 条参考文献。"
        ),
    )

    version: int = Field(
        default=1,
        description="协议版本号。每次规则修订 (如闭环反馈后的Rulebook更新) 递增 1。",
    )

    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="协议生成时间，ISO 8601 格式 (UTC)。",
    )


# Pydantic v2 + `from __future__ import annotations` 需手动触发模型重建，
# 以解析 List[float] / Literal["Ro5","bRo5","custom"] 等延迟求值的类型注解。
RulebookSchema.model_rebuild()


# =============================================================================
# 2. 核心系统提示词 (System Prompt)
# =============================================================================

STRATEGY_AGENT_SYSTEM_PROMPT = """\
# 人设与身份 (Persona)

你是一位**首席计算生物学家 (Principal Computational Biologist)**，供职于一家世界顶级的 AI 药物发现公司。
你拥有 20 年以上的:
- 结构生物学 (X-ray 晶体学、冷冻电镜 Cryo-EM 数据分析) 经验
- 计算化学 (分子对接、分子动力学 MM/PBSA、自由能微扰 FEP) 经验
- 药物化学 (基于结构的药物设计 SBDD、基于片段的药物发现 FBDD) 经验
- ADMET/毒理学 (DMPK、体内毒性机制) 经验

你的职责是为虚拟筛选管道制定**科学严谨、化学可行、量身定制**的《动态药化过滤规则协议》(Rulebook)。
下游的自动化多专家委员会将 100% 严格按你的规则执行绝对值淘汰 —— 你所写的每一个数字、每一个 SMARTS，
都直接决定了哪些分子能活到下一轮。你有责任:

1. **绝不漏杀 (No False Negatives)**: 规则过严会错误淘汰潜在真阳性，这对项目是灾难性的。
2. **精准淘汰 (Precision Filtering)**: 放任明显不可成药的分子进入下游会浪费计算资源。
3. **科学可解释 (Interpretable)**: 每一个阈值都必须有结构生物学或药物化学依据。

---

# 动态规则制定框架 (Dynamic Rule Adjustment)

## 1. 靶点分类 → 规则体系选择

根据靶点类型选择基础规则体系:

### A. 经典酶靶点 (激酶 Kinase、蛋白酶 Protease、磷酸酶 Phosphatase)
- **规则体系**: Ro5 (Lipinski's Rule-of-Five)
- **MW**: [150, 500] Da
- **cLogP**: [-2, 5]
- **HBD** ≤ 5, **HBA** ≤ 10
- **TPSA**: [20, 140]
- **关键特征**: 精准的氢键网络匹配、较少的可旋转键 (刚性结构通常更优)
- **典型阳性参考**: Imatinib (MW=493, LogP=3.5), Dasatinib (MW=488, LogP=2.5)

### B. GPCR 靶点 (A/B/C 类)
- **规则体系**: Ro5，但需根据配体类型微调
  - A 类 (视紫红质样): 正构配体通常较小 (MW 250-450)，偏向疏水
  - B 类 (分泌素样): 肽类配体 → bRo5 放宽
- **MW**: 正构 [200, 500]; 变构 [150, 600]
- **cLogP**: [1, 6] (GPCR 配体通常偏疏水以穿透细胞膜到达膜蛋白)
- **关键特征**: 至少一个可质子化胺基 (与保守的天冬氨酸盐桥)，1-3 个疏水芳环
- **典型阳性参考**: Salmeterol (MW=416, LogP=3.8), Olanzapine (MW=312, LogP=3.0)

### C. PPI 靶点 (蛋白-蛋白相互作用，如 Bcl-2/Bcl-xl、MDM2/p53)
- **规则体系**: bRo5 (Beyond Rule-of-Five)
- **MW**: [150, 1000] Da (PPI 结合面通常 750-1500 Å²，需要较大分子覆盖)
- **cLogP**: [-2, 8] (PPI 结合面常含大型疏水区域)
- **HBD** ≤ 6, **HBA** ≤ 12
- **可旋转键** ≤ 15
- **TPSA**: [40, 250]
- **关键特征**: 大型刚性疏水锚点 + 外围氢键增强选择性
- **典型阳性参考**: Venetoclax (MW=868, LogP=6.2), Navitoclax (MW=974, LogP=8.3)

### D. PROTAC/分子胶 (靶向蛋白降解)
- **规则体系**: bRo5 甚至 beyond bRo5
- **MW**: [400, 1200]
- **cLogP**: [-1, 8]
- **关键特征**: 三组分结构 (POI 配体 + Linker + E3 连接酶配体)
- **特殊考虑**: Linker 长度和柔性是关键设计参数，本协议不限制 Linker 特征

### E. 中枢神经系统 (CNS) 靶点
- **规则体系**: CNS-Ro5 (更严格的 Ro5)
- **MW**: [150, 450]
- **cLogP**: [1, 4.5]
- **TPSA**: [20, 90] (必须<90 才能穿透 BBB)
- **HBD** ≤ 3 (最严格的限制，每增加一个 HBD 穿透率下降 10 倍)
- **可旋转键** ≤ 8

### F. 抗感染靶点 (抗菌/抗病毒/抗寄生虫)
- **规则体系**: Ro5 但对极性要求更宽松
- **MW**: [150, 600]
- **cLogP**: [-3, 5]
- **TPSA**: [20, 180] (许多抗生素需要较高极性)
- **特殊排除**: 排除人 CYP450 强抑制剂 (避免药物-药物相互作用)

## 2. 药效团特征推断指南

从靶点描述和已知结构中推断必需药效团:

- **激酶铰链区** → hbond_donor + hbond_acceptor (模拟 ATP 腺嘌呤的氢键)
- **蛋白酶催化三联体** → hbond_donor + hbond_acceptor + 亲电弹头 (如果是共价抑制剂)
- **PPI 疏水平面** → hydrophobic_ring (至少 2 个芳环参与 π-π 或疏水堆积)
- **GPCR 保守 Asp** → positive_ion (可质子化胺基)
- **金属酶** → 金属螯合基团 (如异羟肟酸、羧酸、巯基)
- **DNA/RNA 靶点** → 平面芳环体系 (嵌入) + positive_ion (与磷酸骨架静电作用)

## 3. 毒性排雷指南

必须明确排除的化合物类型:

- **PAINS**: 查尔酮、醌、烯酮、硫脲、靛红酸酐、罗丹宁 (假阳性惯犯)
- **致突变性**: 亚硝胺前体 (二级胺+亚硝酸盐条件)、芳香胺、环氧前体
- **心脏毒性**: 已知 hERG 结合药效团 (碱性胺+疏水芳环，间距 5-7 Å)
- **肝毒性**: 反应性代谢产物前体 (如噻唑烷二酮、呋喃环)
- **光毒性**: 大共轭体系 (如卟啉类、补骨脂素类)

## 4. 输出质量标准

- 每一个阈值都必须在 rationale 中提供科学依据
- pharmacophore_features 列表必须是具体靶点相关的，不得用通用占位符
- 所有 SMARTS 字符串必须语法正确 (可使用 RDKit 验证)
- 如果无法确定某个阈值，宁可设置宽松的默认值，并在 rationale 中标注不确定性
- 引用已知的阳性对照或上市药物作为参考基准
"""


# =============================================================================
# 用户提示词模板
# =============================================================================

def build_user_prompt(target_info: dict) -> str:
    """根据 TargetInfo 构建用户 Prompt。

    将靶点结构/生化信息组装为结构化的查询文本，供 LLM 推理生成 Rulebook。
    """
    return f"""\
## 虚拟筛选任务: 制定动态药化过滤协议

请根据以下靶点信息，生成一份科学严谨的《动态药化过滤规则协议》。

### 靶点基本信息
- **靶点名称**: {target_info.get('target_name', 'Unknown')}
- **UniProt ID**: {target_info.get('uniprot_id', 'N/A')}
- **PDB ID**: {target_info.get('pdb_id', 'N/A')}
- **靶点类别**: {target_info.get('target_class', 'Unknown')}
- **物种来源**: {target_info.get('organism', 'Homo sapiens')}

### 靶点功能与疾病描述
{target_info.get('description', '无额外描述。')}

### 结合位点信息
- **关键残基**: {', '.join(target_info.get('key_residues', [])) or '未提供'}
- **结合口袋中心**: {target_info.get('binding_site_center', '未提供')}
- **结合口袋尺寸**: {target_info.get('binding_site_size', '未提供')}

### 任务要求
1. 根据靶点类别 `{target_info.get('target_class', 'Unknown')}` 选择正确的规则体系 (Ro5 / bRo5 / custom)。
2. 为每一个理化参数设定合理的阈值范围。
3. 推断该靶点必需的药效团特征 (pharmacophore_required) —— 请基于靶点描述和结合位点性质，不要使用通用占位符。
4. 列出必须排除的 PAINS/毒性/反应性基团 SMARTS。
5. 在 rationale 中详细阐述每个阈值的科学依据。
6. 引用相关文献 (PDB 条目、PMID)。
"""


# =============================================================================
# 3. Strategy Agent 类
# =============================================================================

class StrategyAgent:
    """Strategy Agent — 虚拟筛选的战前情报官。

    使用 OpenAI-compatible API (DeepSeek) 调用 LLM，
    通过 with_structured_output / tool_choice 机制强制输出结构化 Rulebook。
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        Args:
            model: LLM 模型名称 (deepseek-chat / deepseek-v3 / gpt-4o)。
            api_key: API Key (None 则从环境变量 DEEPSEEK_API_KEY 读取)。
            api_base: API Base URL (None 则从环境变量读取)。
            temperature: 策略推理需要一定探索性但不宜过高。
            max_tokens: 最大输出 token 数。
        """
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_base = api_base or os.getenv(
            "DEEPSEEK_API_BASE", "https://api.deepseek.com"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 延迟初始化客户端
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """惰性初始化 OpenAI 兼容客户端。"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=f"{self.api_base}/v1",
            )
        return self._client

    # -------------------------------------------------------------------------
    # 主入口: 生成动态过滤协议
    # -------------------------------------------------------------------------

    def generate_protocol(
        self,
        target_info: dict,
        knowledge_base: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Step 1 主方法: 生成《动态药化过滤协议》。

        流程:
          1. 从 target_info 构建 User Prompt
          2. 拼接 System Prompt + User Prompt
          3. 调用 LLM API (带 JSON 结构化输出约束)
          4. 用 Pydantic 校验解析响应
          5. 若闭环知识库存在，融合先验教训
          6. 返回结构化协议 dict

        Args:
            target_info: 靶点蛋白信息 (TargetInfo TypedDict)。
            knowledge_base: 前轮闭环累积的知识库 (可选)。

        Returns:
            {"protocol": DynamicFilterProtocol dict, "raw_llm_response": str}
        """
        # ---- Step 1a: 构建消息 ----
        system_prompt = STRATEGY_AGENT_SYSTEM_PROMPT
        user_prompt = build_user_prompt(target_info)

        # 如果有闭环知识，追加到 system prompt
        if knowledge_base:
            system_prompt += self._build_knowledge_augmentation(knowledge_base)

        # ---- Step 1b: 调用 LLM ----
        # 优先使用原生 structured output (如 gpt-4o-2024-08-06+)
        # 降级方案: 用 JSON mode + Pydantic 后校验
        rulebook = self._call_llm_structured(system_prompt, user_prompt)

        # ---- Step 1c: 融合闭环洞察 ----
        if knowledge_base and knowledge_base.get("unfavorable_patterns"):
            rulebook = self._apply_closed_loop_insights(rulebook, knowledge_base)

        # ---- Step 1d: 版本号 ----
        rulebook["version"] = 1
        rulebook["generated_at"] = datetime.now(timezone.utc).isoformat()

        return {
            "protocol": rulebook,
        }

    # -------------------------------------------------------------------------
    # LLM 调用 (结构化输出)
    # -------------------------------------------------------------------------

    def _call_llm_structured(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """调用 LLM 并强制输出符合 RulebookSchema 的 JSON。

        策略:
          1. 尝试使用原生 with_structured_output (部分模型支持)
          2. 使用 JSON mode + response_format 作为降级方案
          3. 使用工具调用 (tool_choice="required") 作为兜底方案

        Returns:
            解析后的 RulebookSchema dict。
        """
        # 方案 A: 使用 chat completions + response_format (DeepSeek 支持)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "system",
                        "content": (
                            "请严格按照以下 JSON Schema 输出。不要输出任何非 JSON 内容。"
                            "输出必须是合法 JSON 对象，所有字段必须填写。\n\n"
                            f"JSON Schema:\n{json.dumps(RulebookSchema.model_json_schema(), indent=2, ensure_ascii=False)}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = json.loads(raw_content)
            # Pydantic 校验
            validated = RulebookSchema.model_validate(parsed)
            return validated.model_dump()

        except (json.JSONDecodeError, Exception) as e:
            # 方案 B: JSON 解析失败，重试一次带更强约束
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,  # 降为 0 温度提高确定性
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {
                            "role": "system",
                            "content": (
                                f"上一轮输出 JSON 解析失败: {str(e)}\n"
                                "请仅输出合法 JSON，不要包含 markdown 代码块标记，"
                                "不要添加任何解释文本。"
                            ),
                        },
                    ],
                    response_format={"type": "json_object"},
                )
                raw_content = response.choices[0].message.content.strip()
                # 移除可能的 markdown 标记
                if raw_content.startswith("```"):
                    raw_content = raw_content.split("\n", 1)[-1].rsplit("```", 1)[0]
                parsed = json.loads(raw_content)
                validated = RulebookSchema.model_validate(parsed)
                return validated.model_dump()

            except Exception as e2:
                # 方案 C: 返回 RulebookSchema 的默认值作为兜底
                # 下游节点应检测到默认值并触发人工审核
                default = RulebookSchema()
                default.rationale = (
                    f"[AUTO-FALLBACK] LLM 结构化输出失败:\n"
                    f"  第一次尝试: {str(e)}\n"
                    f"  第二次尝试: {str(e2)}\n"
                    f"  已使用保守默认值。强烈建议人工审核此规则协议。"
                )
                return default.model_dump()

    # -------------------------------------------------------------------------
    # 闭环知识融合
    # -------------------------------------------------------------------------

    def _build_knowledge_augmentation(self, knowledge_base: dict) -> str:
        """将闭环知识库中的教训转化为 System Prompt 的上下文提醒。

        使 LLM 在制定新规则时"记住"前轮迭代的教训。
        """
        parts = ["\n\n## 闭环知识库 (前轮迭代教训)\n"]

        if knowledge_base.get("unfavorable_patterns"):
            parts.append("### 已确认的不利子结构模式 (请纳入排除列表):")
            for pattern in knowledge_base["unfavorable_patterns"]:
                parts.append(f"  - {pattern}")

        if knowledge_base.get("privileged_scaffolds"):
            parts.append("### 已验证的优势骨架 (可适当放宽对此类骨架的理化限制):")
            for scaffold in knowledge_base["privileged_scaffolds"][:5]:
                parts.append(f"  - {scaffold}")

        if knowledge_base.get("md_derived_insights"):
            parts.append("### MD 动力学教训 (用于调整药效团需求):")
            for insight in knowledge_base["md_derived_insights"][:3]:
                parts.append(f"  - {insight}")

        if knowledge_base.get("false_positive_patterns"):
            parts.append("### 对接假阳性模式 (包含此类特征的分子请降权):")
            for pattern in knowledge_base["false_positive_patterns"][:5]:
                parts.append(f"  - {pattern}")

        return "\n".join(parts)

    def _apply_closed_loop_insights(
        self,
        rulebook: dict,
        knowledge_base: dict,
    ) -> dict:
        """后处理: 确保闭环知识的不利模式已纳入排除列表。

        此步骤在 LLM 输出之后执行，确保关键教训不被遗漏。
        """
        # 合并不利子结构
        existing_excluded = set(rulebook.get("excluded_substructures", []))
        kb_unfavorable = set(knowledge_base.get("unfavorable_patterns", []))
        rulebook["excluded_substructures"] = sorted(
            existing_excluded | kb_unfavorable
        )

        # 合并假阳性模式
        existing_reactive = set(rulebook.get("reactive_groups", []))
        kb_false_pos = set(knowledge_base.get("false_positive_patterns", []))
        rulebook["reactive_groups"] = sorted(
            existing_reactive | kb_false_pos
        )

        return rulebook

    # -------------------------------------------------------------------------
    # 靶点背景调研 (辅助)
    # -------------------------------------------------------------------------

    def research_target(self, target_info: dict) -> Dict[str, Any]:
        """对靶点进行快速背景调研 (增强知识, 非必需)。

        TODO: 集成 UniProt API / PubMed 检索 / PDB 数据拉取。
        """
        # 占位: 可集成检索增强生成 (RAG)
        return {
            "uniprot_summary": "",
            "known_ligands": [],
            "related_pdbs": [],
        }


# =============================================================================
# 4. LangGraph 节点函数
# =============================================================================

def scouting_node(state: dict) -> dict:
    """LangGraph 节点函数 —— Step 1: Strategy Agent 战前侦察。

    从 MACVSState 中提取 target_info，调用 StrategyAgent 生成 Rulebook，
    将结果写回 state 的 filter_protocol 字段。

    调用方式:
      workflow.add_node("strategy", scouting_node)

    Args:
        state: MACVSState (LangGraph 状态字典)。

    Returns:
        dict partial state update:
          - pipeline_stage: "strategy"
          - filter_protocol: DynamicFilterProtocol dict
          - protocol_version: int (递增)
          - event_log: 追加日志条目
    """
    # ---- 1. 提取靶点信息 ----
    target_info = state.get("target_info", {})
    if not target_info:
        return {
            "pipeline_stage": "error",
            "errors": [{
                "node": "strategy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Missing target_info in state. Cannot run Strategy Agent.",
            }],
            "event_log": ["[Strategy] ERROR: No target_info provided."],
        }

    # ---- 2. 获取已有的闭环知识库 (如果是第 N 轮迭代) ----
    knowledge_base = state.get("knowledge_base", None)

    # ---- 3. 初始化 Agent 并生成协议 ----
    agent = StrategyAgent()
    result = agent.generate_protocol(
        target_info=target_info,
        knowledge_base=knowledge_base,
    )

    protocol = result.get("protocol", {})

    # ---- 4. 更新 protocol_version (递增) ----
    prev_version = state.get("protocol_version", 0)
    protocol["version"] = prev_version + 1

# ---- 5. 写回状态 ----
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "pipeline_stage": "strategy",           # 恢复为 workflow 路由期望的 "strategy"
        "filter_protocol": protocol,            # 恢复为 state.py 期望的 "filter_protocol"
        "protocol_version": prev_version + 1,
        "updated_at": timestamp,
        "event_log": [                          # 恢复为 state.py 中定义的 "event_log"
            f"[{timestamp}] [Strategy] Protocol v{prev_version + 1} generated. "
            f"Category: {protocol.get('rule_category', 'N/A')}. "
            f"MW range: {protocol.get('mw_range', 'N/A')}."
        ],
    }