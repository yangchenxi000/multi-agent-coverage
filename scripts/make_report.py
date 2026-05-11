"""Generate top-level reproduction report from outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def main() -> None:
    smoke_metrics = _load_json(ROOT / "outputs" / "smoke_single" / "metrics.json")
    single_df = pd.read_csv(ROOT / "outputs" / "single_obstacle" / "metrics.csv") if (ROOT / "outputs" / "single_obstacle" / "metrics.csv").exists() else pd.DataFrame()
    two_df = pd.read_csv(ROOT / "outputs" / "two_agents" / "comparison_metrics.csv") if (ROOT / "outputs" / "two_agents" / "comparison_metrics.csv").exists() else pd.DataFrame()
    three_df = pd.read_csv(ROOT / "outputs" / "three_agents" / "comparison_metrics.csv") if (ROOT / "outputs" / "three_agents" / "comparison_metrics.csv").exists() else pd.DataFrame()
    abl_df = pd.read_csv(ROOT / "outputs" / "ablation" / "ablation_results.csv") if (ROOT / "outputs" / "ablation" / "ablation_results.csv").exists() else pd.DataFrame()

    lines: list[str] = []
    lines.append("# REPORT_reproduction")
    lines.append("")
    lines.append("## 1. 复现目标与论文背景")
    lines.append("复现论文 Multi-Agent Coordinated Ergodic Coverage and Obstacle Avoidance with Stein Variational Gradient Flow，核心是轨迹点上的 Stein 方向更新、SDF 障碍感知密度、多智能体耦合分工。")
    lines.append("")
    lines.append("## 2. 方法实现说明")
    lines.append("- 数据结构是 trajectory-first（`[N,T+1,2]`），不是随机散点云。")
    lines.append("- 初始轨迹为直线（可配置为两条/三条平行横线）。")
    lines.append("- Stein 在轨迹离散时刻点上计算 task/self/coupling 三项。")
    lines.append("- 使用 SDF + sigmoid 形成 obstacle-aware density。")
    lines.append("- 轨迹更新采用可运行近似式：`p_new = p + eta*dir + lambda*Laplacian(p)`。")
    lines.append("")
    lines.append("## 3. baseline 与 ours 的定义")
    lines.append("- baseline: 标准 Stein（self term），无 cross-trajectory coupling。")
    lines.append("- ours: baseline + obstacle-aware density + coupling term。")
    lines.append("")
    lines.append("## 4. trajectory-based Stein 关键解释")
    lines.append("每个 agent 的每个轨迹点 `p_i(t)` 是 Stein 样本。每轮迭代都在整条轨迹上更新中间点，保留轨迹结构和时序语义。")
    lines.append("")
    lines.append("## 5. 与原文一致之处")
    lines.append("- 轨迹经验分布 / team occupancy 思路")
    lines.append("- Fourier ergodic metric")
    lines.append("- obstacle-aware density")
    lines.append("- cross-trajectory coupling")
    lines.append("")
    lines.append("## 6. 上游论文补齐之处")
    lines.append("借用了 FMEC 的 flow-matching 迭代思想（reference direction 驱动轨迹/控制更新）。")
    lines.append("")
    lines.append("## 7. 工程近似之处")
    lines.append("严格 LQFM 闭式解未完整恢复；当前主版本为稳定可运行 trajectory-update 近似，并加入障碍投影与分段修复保证可用性。")
    lines.append("")
    lines.append("## 8. 各实验总表")
    if smoke_metrics:
        lines.append("### Experiment 0: smoke single")
        lines.append(_md_table(pd.DataFrame([smoke_metrics])))
    if not single_df.empty:
        lines.append("### Experiment 1: single obstacle")
        lines.append(_md_table(single_df))
    if not two_df.empty:
        lines.append("### Experiment 2: two agents baseline vs ours")
        lines.append(_md_table(two_df))
    if not three_df.empty:
        lines.append("### Experiment 3: three agents baseline vs ours")
        lines.append(_md_table(three_df))
    if not abl_df.empty:
        lines.append("### Experiment 4: ablation")
        lines.append(_md_table(abl_df))
    lines.append("")
    lines.append("## 9. 主要结果图")
    lines.append("- `outputs/smoke_single/iter_000.png`, `iter_010.png`, `iter_050.png`, `iter_100.png`")
    lines.append("- `outputs/single_obstacle/no_obstacle_aware/trajectory_overlay.png`")
    lines.append("- `outputs/single_obstacle/with_obstacle_aware/trajectory_overlay.png`")
    lines.append("- `outputs/two_agents/baseline_vs_ours_2agents.png`")
    lines.append("- `outputs/three_agents/baseline_vs_ours_3agents.png`")
    lines.append("- `outputs/ablation/ablation_ergodic.png`, `outputs/ablation/ablation_overlap.png`")
    lines.append("")
    lines.append("## 10. 复现结论与局限")
    lines.append("工程上已完成可运行复现，并能自动产出图表和报告；局限在于 LQFM 闭式控制更新还未完全严格化。")
    lines.append("")
    lines.append("## 复现结论")
    lines.append("1. 这篇论文是否已经被工程上成功复现：是，已完成 trajectory-based Stein + obstacle-aware + coupling 的可运行复现。")
    lines.append("2. 哪些部分严格按论文实现：轨迹点 Stein 结构、SDF 障碍感知密度、多智能体耦合项、baseline/ours 对比流程。")
    lines.append("3. 哪些部分依赖上游 FMEC 论文补齐：reference flow 到 trajectory/control 迭代匹配的流程化实现思路。")
    lines.append("4. 哪些部分仍属于工程近似：主优化器使用稳定的 trajectory-update 近似，而非完整 LQFM 闭式解。")
    lines.append("5. 下一步若要进一步逼近原文，还差什么：恢复并验证严格 LQFM 推导，按论文同款参数和场景做更细量化对齐。")

    (ROOT / "REPORT_reproduction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Report generated:", ROOT / "REPORT_reproduction.md")


if __name__ == "__main__":
    main()
