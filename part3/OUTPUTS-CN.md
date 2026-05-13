# Part 3 输出布局

现在的正式输出规则是：

- 所有最终实验结果统一放在 `part3/results/`
- 所有非最终 smoke 检查和归档日志统一放在 `part3/artifacts_debug/`

当前布局如下：

- `results/Wild_Video/`
  - 报告中使用的最终完整实验结果
  - 包含 `sam3/`、`sam3_1/` 和 `summary/`
- `results_debug/ablation/`
  - `SAM 3` vs `SAM 3.1` 的 smoke 消融输出
- `results/DAVIS_Dataset/`
  - 预留给未来的全 DAVIS 重跑结果
- `artifacts_debug/checks/`
  - 类似 `wild_video1_short33_check` 这类短片 smoke 检查
- `artifacts_debug/legacy_outputs/`
  - 清理前留下的旧本地运行结果
- `artifacts_debug/slurm_job_logs/`
  - 归档的 SLURM `.out` / `.err` 日志

这样分开之后，正式结果会更容易找，同时历史调试材料也还在，方便追溯。
