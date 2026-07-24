#!/bin/bash
# AutoVS Phase 2 验证 — 统一入口脚本
# 用法:
#   bash tests/phase2_validation/run_all.sh              # 运行全部测试
#   bash tests/phase2_validation/run_all.sh --quick      # 仅运行快速测试(无GPU)
#   bash tests/phase2_validation/run_all.sh --gpu        # 仅运行GPU测试

set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=/users_home/wangpengzheng/miniforge3/bin/python
MODE="${1:-all}"

echo "============================================================"
echo "  AutoVS Phase 2 验证测试"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  模式: $MODE"
echo "============================================================"

# ─── 快速测试 (无GPU需求) ───────────────────────────────────
run_quick_tests() {
    echo ""
    echo "--- Test 00: 配置验证 ---"
    $PYTHON tests/phase2_validation/test_00_config_validation.py

    echo ""
    echo "--- Test 04: 引擎选择逻辑 ---"
    $PYTHON tests/phase2_validation/test_04_engine_selection.py

    echo ""
    echo "--- Test 03: 多样性选择 ---"
    $PYTHON tests/phase2_validation/test_03_diversity_selection.py
}

# ─── GPU测试 (提交Slurm作业并等待) ──────────────────────────
run_gpu_tests() {
    echo ""
    echo "--- Test 01: GNINA GPU Slurm对接 ---"
    echo "提交GNINA作业到gpu_long分区..."
    $PYTHON tests/phase2_validation/test_01_gnina_gpu.py
    GNINA_EXIT=$?

    echo ""
    echo "--- Test 02: DiffDock GPU Slurm PPI对接 ---"
    echo "提交DiffDock作业到gpu_long分区..."
    $PYTHON tests/phase2_validation/test_02_diffdock_gpu.py
    DIFFDOCK_EXIT=$?

    if [ $GNINA_EXIT -eq 0 ] && [ $DIFFDOCK_EXIT -eq 0 ]; then
        echo ""
        echo "✅ 全部GPU测试通过"
    else
        echo ""
        echo "⚠️ 部分GPU测试未通过 (GNINA=$GNINA_EXIT, DiffDock=$DIFFDOCK_EXIT)"
    fi
}

# ─── 报告 ────────────────────────────────────────────────────
run_report() {
    echo ""
    echo "--- 生成验证报告 ---"
    $PYTHON tests/phase2_validation/phase2_report.py
}

case "$MODE" in
    --quick)
        run_quick_tests
        run_report
        ;;
    --gpu)
        run_gpu_tests
        run_report
        ;;
    all|*)
        run_quick_tests
        run_gpu_tests
        run_report
        ;;
esac

echo ""
echo "============================================================"
echo "  Phase 2 验证完成"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
