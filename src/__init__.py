"""
src — AutoVS-Agent 核心源码
============================
基于 LangGraph 的多智能体协作虚拟筛选系统。

模块:
  - src.graph:  全局状态定义 + LangGraph 工作流拓扑
  - src.agents: 6 个核心智能体 + Proxy MLP 代理模型
  - src.tools:  分子工具库 (RDKit, PLIP, GNINA, GROMACS 封装)
"""
