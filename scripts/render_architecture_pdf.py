#!/usr/bin/env python3
"""Render the current AutoVS-Agent architecture as a two-page Chinese PDF."""
from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "AutoVS-Agent_当前架构图.pdf"
PAGE = landscape(A3)

GREEN = HexColor("#D8F3DC")
GREEN_LINE = HexColor("#2D6A4F")
BLUE = HexColor("#DBEAFE")
BLUE_LINE = HexColor("#2563EB")
YELLOW = HexColor("#FFF3BF")
YELLOW_LINE = HexColor("#E67700")
RED = HexColor("#FFE3E3")
RED_LINE = HexColor("#C92A2A")
INK = HexColor("#172033")
MUTED = HexColor("#5B6578")
PANEL = HexColor("#F7F9FC")


def register_fonts() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def text(c: canvas.Canvas, value: str, x: float, y: float, size: float = 10,
         color=INK, align: str = "center", leading: float | None = None) -> None:
    c.setFillColor(color)
    c.setFont("STSong-Light", size)
    lines = value.split("\n")
    leading = leading or size * 1.25
    for index, line in enumerate(lines):
        yy = y - index * leading
        if align == "left":
            c.drawString(x, yy, line)
        elif align == "right":
            c.drawRightString(x, yy, line)
        else:
            c.drawCentredString(x, yy, line)


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str,
        fill=GREEN, stroke=GREEN_LINE, size: float = 10, radius: float = 10) -> tuple[float, float, float, float]:
    c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(1.3)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    lines = label.split("\n")
    total = (len(lines) - 1) * size * 1.25
    text(c, label, x + w / 2, y + h / 2 + total / 2 - size * 0.35, size=size)
    return x, y, w, h


def group(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str) -> None:
    c.setFillColor(PANEL); c.setStrokeColor(HexColor("#C8D0DD")); c.setLineWidth(1)
    c.roundRect(x, y, w, h, 12, fill=1, stroke=1)
    text(c, title, x + 12, y + h - 23, size=13, color=INK, align="left")


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
          label: str = "", dashed: bool = False, color=HexColor("#667085")) -> None:
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.5)
    c.setDash(5, 4) if dashed else c.setDash()
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 8
    for delta in (2.55, -2.55):
        c.line(x2, y2, x2 + length * math.cos(angle + delta), y2 + length * math.sin(angle + delta))
    c.setDash()
    if label:
        text(c, label, (x1 + x2) / 2, (y1 + y2) / 2 + 7, size=8, color=MUTED)


def title(c: canvas.Canvas, heading: str, subtitle: str, page_no: int) -> None:
    width, height = PAGE
    text(c, heading, 38, height - 42, size=22, align="left")
    text(c, subtitle, 38, height - 66, size=10, color=MUTED, align="left")
    text(c, f"AutoVS-Agent  |  第 {page_no} 页", width - 38, 24, size=8, color=MUTED, align="right")


def architecture_page(c: canvas.Canvas) -> None:
    width, height = PAGE
    title(c, "AutoVS-Agent 当前系统架构", "绿色=已实现；黄色=已部署但待完整验证；红色=接口预留或环境未就绪", 1)

    group(c, 28, 100, 160, 640, "1. 用户入口")
    web = box(c, 45, 620, 126, 72, "Web界面\n上传PDB、分子库、任务")
    cli = box(c, 45, 515, 126, 72, "CLI\nrun/status/resume/report")
    ext = box(c, 45, 410, 126, 72, "外部MCP客户端\n智能体或调试工具", fill=BLUE, stroke=BLUE_LINE)

    group(c, 210, 100, 285, 640, "2. 应用与决策层")
    pipeline = box(c, 232, 650, 240, 58, "PipelineService：唯一主流水线", fill=BLUE, stroke=BLUE_LINE, size=11)
    scout = box(c, 232, 555, 108, 58, "TargetScout\n靶点调研")
    generator = box(c, 364, 555, 108, 58, "策略生成\n结构化JSON")
    vote = box(c, 232, 465, 108, 58, "全排列投票\n三位评审官")
    evolve = box(c, 364, 465, 108, 58, "策略进化\nTop策略修复")
    compiler = box(c, 232, 365, 240, 62, "Workflow Compiler\n转换为严格WorkflowPlan v1", fill=BLUE, stroke=BLUE_LINE)
    valid = box(c, 232, 260, 240, 62, "可执行性校验\n只允许已注册Action", fill=BLUE, stroke=BLUE_LINE)
    select = box(c, 232, 155, 240, 62, "选择排名最高的可执行策略\n不可执行则自动尝试下一名")

    group(c, 518, 100, 230, 640, "3. MCP与工具控制")
    mcp = box(c, 540, 630, 186, 70, "autovs_tools_mcp\n127.0.0.1:8765/mcp", fill=BLUE, stroke=BLUE_LINE)
    cap = box(c, 540, 530, 186, 62, "能力发现与健康检查", fill=BLUE, stroke=BLUE_LINE)
    jobs = box(c, 540, 430, 186, 62, "提交步骤 / 查询状态\n日志 / 产物 / 取消", fill=BLUE, stroke=BLUE_LINE)
    manager = box(c, 540, 315, 186, 72, "ToolManager\n统一受控工具调度", size=11)
    security = box(c, 540, 205, 86, 65, "安全层\n路径白名单")
    checkpoint = box(c, 640, 205, 86, 65, "Checkpoint\n输入参数哈希")
    note = box(c, 540, 125, 186, 48, "禁止任意Shell命令", fill=RED, stroke=RED_LINE)

    group(c, 772, 100, 248, 640, "4. 计算工具与环境")
    py = box(c, 794, 635, 204, 68, "autovs-core / Python\nRDKit、口袋、排名、报告")
    conda = box(c, 794, 535, 204, 68, "专用Conda环境\nOpenBabel、smina、PLIP")
    gnina = box(c, 794, 435, 204, 68, "GNINA + Slurm GPU\n已部署，待GPU恢复验证", fill=YELLOW, stroke=YELLOW_LINE)
    admet = box(c, 794, 335, 204, 68, "autovs-admet / ADMET-AI\n环境定义完成，尚未安装", fill=RED, stroke=RED_LINE)
    md = box(c, 794, 220, 204, 82, "GROMACS Apptainer + Slurm\n短MD / 100ns / MMGBSA\n镜像存在，生产适配待验证", fill=RED, stroke=RED_LINE)
    cpu = box(c, 794, 125, 204, 58, "当前真实CPU链路已跑通")

    group(c, 1045, 100, 118, 640, "5. 状态与产物")
    sqlite = box(c, 1060, 610, 88, 72, "SQLite\n任务和作业", fill=BLUE, stroke=BLUE_LINE)
    files = box(c, 1060, 485, 88, 88, "任务目录\nPDB/SDF/CSV\n日志与校验和", fill=BLUE, stroke=BLUE_LINE)
    report = box(c, 1060, 350, 88, 88, "最终报告\nMarkdown/HTML\nTop候选", fill=BLUE, stroke=BLUE_LINE)
    gaps = box(c, 1060, 220, 88, 88, "证据缺口\nADMET/MD\n明确标记", fill=YELLOW, stroke=YELLOW_LINE)

    arrow(c, 171, 656, 232, 679)
    arrow(c, 171, 551, 232, 679)
    arrow(c, 171, 446, 540, 665, label="MCP调用")
    arrow(c, 352, 650, 286, 613)
    arrow(c, 340, 584, 364, 584)
    arrow(c, 418, 555, 286, 523)
    arrow(c, 340, 494, 364, 494)
    arrow(c, 418, 465, 352, 427)
    arrow(c, 352, 365, 352, 322)
    arrow(c, 352, 260, 352, 217)
    arrow(c, 472, 186, 540, 351, label="直接调用")
    arrow(c, 633, 630, 633, 592)
    arrow(c, 633, 530, 633, 492)
    arrow(c, 633, 430, 633, 387)
    arrow(c, 633, 315, 583, 270)
    arrow(c, 633, 315, 683, 270)
    arrow(c, 726, 351, 794, 669)
    arrow(c, 726, 351, 794, 569)
    arrow(c, 726, 351, 794, 469, dashed=True)
    arrow(c, 726, 351, 794, 369, dashed=True)
    arrow(c, 726, 351, 794, 261, dashed=True)
    arrow(c, 998, 669, 1060, 646)
    arrow(c, 998, 569, 1060, 529)
    arrow(c, 998, 159, 1060, 394)
    arrow(c, 998, 261, 1060, 264, dashed=True)
    c.showPage()


def flow_page(c: canvas.Canvas) -> None:
    width, height = PAGE
    title(c, "一次任务的真实执行流程", "当前CPU基线已经运行到PLIP和排名；ADMET与分级MD作为明确证据缺口保留", 2)

    y1, y2, y3, y4 = 660, 505, 350, 195
    xs = [38, 182, 326, 470, 614, 758, 902, 1046]
    labels1 = [
        "1 用户提交\n自然语言+PDB+分子库", "2 输入暂存\nSHA256与Task ID", "3 运行模式\n正式或CPU诊断",
        "4 靶点调研", "5 结构化策略生成", "6 全排列三专家投票", "7 Top策略进化", "8 可执行性校验",
    ]
    row1 = []
    for x, label in zip(xs, labels1):
        row1.append(box(c, x, y1, 120, 72, label, fill=BLUE if "运行模式" in label or "校验" in label else GREEN, stroke=BLUE_LINE if "运行模式" in label or "校验" in label else GREEN_LINE, size=9))
    for a, b in zip(row1, row1[1:]): arrow(c, a[0] + a[2], a[1] + a[3] / 2, b[0], b[1] + b[3] / 2)

    labels2 = [
        "9 WorkflowPlan v1", "10 输入与口袋校验", "11 RDKit标准化/去重\nPAINS/显式氢",
        "12 ETKDGv3构象\nMMFF94s或UFF", "13 OpenBabel\n蛋白PDB/PDBQT", "14 smina CPU真实对接",
        "15 最佳Affinity姿态", "16 构建蛋白-配体复合物",
    ]
    row2 = []
    for x, label in zip(reversed(xs), labels2):
        row2.append(box(c, x, y2, 120, 72, label, size=9))
    arrow(c, row1[-1][0] + row1[-1][2] / 2, row1[-1][1], row2[0][0] + row2[0][2] / 2, row2[0][1] + row2[0][3])
    for a, b in zip(row2, row2[1:]): arrow(c, a[0], a[1] + a[3] / 2, b[0] + b[2], b[1] + b[3] / 2)

    labels3 = [
        "17 PLIP相互作用分析", "18 合并Affinity与PLIP", "19 方向归一化排名", "20 骨架多样性限制",
        "21 候选Top 20", "22 ADMET-AI", "23 Top 10短MD", "24 Top 3做100ns\n70-100ns MMGBSA",
    ]
    row3 = []
    for index, (x, label) in enumerate(zip(xs, labels3)):
        pending = index >= 5
        row3.append(box(c, x, y3, 120, 72, label, fill=RED if pending else GREEN, stroke=RED_LINE if pending else GREEN_LINE, size=9))
    arrow(c, row2[-1][0] + row2[-1][2] / 2, row2[-1][1], row3[0][0] + row3[0][2] / 2, row3[0][1] + row3[0][3])
    for index, (a, b) in enumerate(zip(row3, row3[1:])):
        arrow(c, a[0] + a[2], a[1] + a[3] / 2, b[0], b[1] + b[3] / 2, dashed=index >= 4)

    checkpoint = box(c, 130, y4, 250, 76, "每一步生成Checkpoint\n输入文件+参数+工具版本哈希", fill=BLUE, stroke=BLUE_LINE)
    resume = box(c, 470, y4, 250, 76, "服务重启后恢复\n相同输入不重复计算", fill=BLUE, stroke=BLUE_LINE)
    final = box(c, 810, y4, 250, 76, "最终可复现报告\n候选、分数、失败、环境和证据缺口", fill=BLUE, stroke=BLUE_LINE)
    arrow(c, 380, y4 + 38, 470, y4 + 38)
    arrow(c, 720, y4 + 38, 810, y4 + 38)
    arrow(c, 962, y3, 935, y4 + 76, dashed=True, label="计算证据汇总")

    text(c, "当前已验证", 45, 112, size=10, align="left")
    box(c, 120, 98, 48, 24, "", fill=GREEN, stroke=GREEN_LINE)
    text(c, "状态/控制", 205, 112, size=10, align="left")
    box(c, 280, 98, 48, 24, "", fill=BLUE, stroke=BLUE_LINE)
    text(c, "尚未接通", 365, 112, size=10, align="left")
    box(c, 440, 98, 48, 24, "", fill=RED, stroke=RED_LINE)
    text(c, "虚线表示待完成或可选证据", 530, 112, size=10, color=MUTED, align="left")
    c.showPage()


def main() -> None:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=PAGE, pageCompression=1)
    c.setTitle("AutoVS-Agent 当前架构图")
    c.setAuthor("AutoVS-Agent")
    architecture_page(c)
    flow_page(c)
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
