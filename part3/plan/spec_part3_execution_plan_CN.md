# Part 3 最终执行规格书

## Summary

本文档是 Part 3 的最终执行规格书，基于以下已确认输入整理而成，用于指导实现，但不替换任何现有计划文档：

- `part3/plan/create_part3.md`
- `part3/plan/part3_codex_execution_plan.md`
- `part3/plan/Notes_for_Attention.md`
- `part3/plan/Slurm_Partition_Selection_Strategy.md`
- `data/Wild_Video/README.md`

Part 3 在前两个阶段的基础上，进一步聚焦于视频中由目标物体引起的副作用消除，尤其是阴影、倒影、镜像和残留痕迹。主技术路线保持为 `SAM 3 + DiffuEraser + ROSE`，并以 `part2` 作为实现参考和基线来源。

本文档规定了 `part3/` 下实现必须产出的内容、在 SLURM 集群上的运行规则、必须优先实现的方法、DAVIS 与 paired wild video 的评估边界，以及允许和不允许的降级策略。

## Environment and HPC Constraints

- GPU 作业必须在计算节点上运行，不要在登录节点直接进行 GPU 推理。
- smoke test 和环境检查默认首先考虑 `debug` 分区。
- 如果正式任务预计也能在 `debug` 限制内完成，也应先考虑 `debug`，而不是只把它留给 smoke test。
- 如果 `debug` 因实时排队情况或用户级 QOS 限制明显吃亏，则按照 `part3/plan/Slurm_Partition_Selection_Strategy.md` 切换到更合适的分区。
- 分区选择必须遵循以下固定顺序：
  - 先判断任务是否适合 `debug`
  - 再看低优先级共享分区
  - 再看中优先级独占分区
  - 最后看高优先级应急分区
- 不要默认使用 `long_cpu` 或 `long_gpu`。只有在运行时间确实需要更长上限，或实时队列明显更优时才使用。
- 如果 A40 已足够，不要默认切到 A800。
- 如果任务可以用 CPU 完成，不要默认抢 GPU。
- 能放进 `debug` 的短正式任务，除非实时队列压力使其他合适分区明显更优，否则仍应优先使用 `debug`。
- 实现时应假定 `debug` 虽然免费，但同时受到运行时长和 QOS 限制，不能假设它支持无限并发。

### 资产下载规则

- 如果需要下载额外模型，先在 `/hpc2hdd/home/ckwong627/workdir/models` 下创建专用子目录。
- 下载命令使用：

```bash
hf download <repo-or-file> --local-dir /hpc2hdd/home/ckwong627/workdir/models/<target_dir>
```

- 如果需要下载额外数据集，同样遵守这条规则。
- 任何下载步骤都必须记录：
  - 下载了什么
  - 大概占用多少存储空间
  - 运行时大概需要多少资源
- 大体积 checkpoint 不得提交进仓库。

## Implementation Changes

- 将 `part3/plan/Notes_for_Attention.md` 视为正式需求来源，不再把它当作未确认备注。
- 保持 `part3` 是独立实现区域，不要通过悄悄修改 `part2` 行为来完成 Part 3。
- 复用 `part2` 的设计模式，特别是 CLI 组织方式、pipeline 编排风格、evaluation 风格和输出约定，同时保持 `part1/` 和 `part2/` 不被破坏。
- 保留 Part 3 的主方法叙事：
  - 基线对比方法：`part2_sam2_propainter`
  - 用于建立 diffusion-based object removal 的重点方法：`sam3_diffueraser_object`
  - 主要主推方法：`sam3_diffueraser_side_effect`、`sam3_rose_object`、`sam3_rose_side_effect`
- 最终报告中的 Part 3 主方法应从 ROSE 相关方案中选出，除非有明确记录说明为什么不能这样做。
- 当 SAM 3 checkpoint 不可用时，不允许静默从 SAM 3 回退到 SAM 2。
- 对于缺失 SAM 3 checkpoint 的情况，唯一允许的降级方式是由用户或 CLI 显式提供 existing-mask 路径。

### 数据与评估边界

- DAVIS 初始只关注 `bmx-trees` 和 `tennis`。
- paired wild evaluation 使用当前 `data/Wild_Video` 下的视频。仓库里的说明目前只确认“这些是新的 wild videos，要使用它们”，并没有建立比现有计划文档更完整的数据规范。
- `data/Wild_Video/clean_gt_no_person` 下的 clean wild ground truth 仅用于评估。
- clean wild ground truth 不得用于 mask 生成、side-effect mask 扩张、prompt 构造或视频修复推理。
- DAVIS 的评估以 mask 为中心：
  - 计算 `JM`
  - 计算 `JR`
- paired wild video 的评估以视频质量为中心：
  - 计算 `PSNR`
  - 计算 `SSIM`

## Public Interfaces / CLI / Config

### `run_part3.py`

- 作为 Part 3 的主执行入口。
- 负责按照 sequence、method、prompt 和 mask source 组织执行。
- 支持配置文件默认值以及 CLI override。
- 在 SLURM 集群上，涉及 GPU 的使用示例应优先给出 `sbatch` 方式。
- 直接 `python` 执行 GPU 任务只应在非集群环境，或者已经进入分配好的计算节点环境时文档化。

### `evaluate_part3.py`

- DAVIS 只评估 `JM` 和 `JR`。
- paired wild videos 只评估 `PSNR` 和 `SSIM`。
- 在计算指标前，可以选择性地对预测帧和 ground-truth wild 帧做对齐。
- 必须明确写出：推理阶段禁止使用 clean GT。

### `SAM3MaskGenerator`

- 负责基于 prompt 的 SAM 3 视频 mask 生成。
- 如果 SAM 3 checkpoint 不可用，必须报出清晰错误。
- 不允许静默用 SAM 2 替代 SAM 3。
- 只有在配置或 CLI 明确要求时，才允许使用 existing-mask fallback。

### `build_side_effect_mask(...)`

- 负责从 object mask 构建 side-effect-aware mask。
- 默认必须启用 morphological expansion。
- 默认必须启用 downward shadow expansion。
- 可以通过配置可选支持 reflection ROI 和 mirror-axis heuristic。

### `DiffuEraserRunner`

- 封装 DiffuEraser 的执行。
- 必须保存 stdout 和 stderr 日志。
- 如果上游 CLI 形状与预期参数不匹配，必须报出有帮助的错误。
- 不允许静默失败。

### `ROSERunner`

- 封装 ROSE 的执行。
- v1 默认策略必须写死为通过 padding 满足 `16n + 1` 帧长约束。
- 后续可以增加 chunking 作为优化，但它不是 v1 的默认行为。
- 必须保存 stdout 和 stderr 日志。

### 默认配置

- 初始 DAVIS 序列：`bmx-trees`, `tennis`
- Wild 视频根目录：
  - `data/Wild_Video/input_with_person`
  - `data/Wild_Video/clean_gt_no_person`
- smoke test 默认优先 `debug`
- 正式任务默认遵循 `part3/plan/Slurm_Partition_Selection_Strategy.md` 中定义的有序分区策略

## Test Plan

### 文件检查

- `part3/plan/spec_part3_execution_plan.md` 存在。
- `part3/plan/spec_part3_execution_plan_CN.md` 存在。
- 两个文件都为 UTF-8 编码。

### 结构检查

- 两份文档包含相同的章节结构。
- 中文版必须是完整等价翻译，而不是摘要版。

### 内容检查

- 两份文档都明确引用：
  - `Notes_for_Attention.md`
  - `Slurm_Partition_Selection_Strategy.md`
- 两份文档都写明 `debug` 不仅用于 smoke test，也可用于符合其限制的正式任务。
- 两份文档都定义了分区升级顺序，并说明了基于队列和 QOS 的切换条件。
- 两份文档都定义了资产下载目录和资源记录要求。
- 两份文档都写明 wild clean GT 仅用于评估。
- 两份文档都定义了 `part2` 的复用边界，并明确 `part3` 仍是独立实现区域。

### 仓库一致性检查

- 路径、类名、方法名和脚本名与当前仓库计划和命名保持一致。
- 不再保留“`Notes_for_Attention.md` 缺失”的旧假设。
- 不夸大 `data/Wild_Video/README.md` 的作用，因为它目前只提供了一个非常简短的确认，说明应使用这批新的 wild videos。

## Assumptions and Defaults

- 本英文稿是主稿，中文稿必须保持完整等价。
- 这两份文件是最终执行规格，不替换以下现有文档：
  - `create_part3.md`
  - `part3_codex_execution_plan.md`
  - `Notes_for_Attention.md`
  - `Slurm_Partition_Selection_Strategy.md`
- 如果更早的计划表述与 `Notes_for_Attention.md` 冲突，以已确认的注意事项为准。
- 由于 `data/Wild_Video/README.md` 目前只包含极少信息，Wild 数据相关假设主要依赖仓库目录结构和 Part 3 主计划文档。
- 本规格书定义的是计划文档中必须写明并长期保持的内容，它本身不直接实现 Part 3 的代码流水线。
