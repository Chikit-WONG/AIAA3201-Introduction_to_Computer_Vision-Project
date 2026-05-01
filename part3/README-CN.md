# Part 3

Part 3 是本课程项目的最终视频目标移除流水线实现。该流水线使用 `SAM 3` 或 `SAM 3.1` 进行视频掩码生成，再接以下两类修复后端之一：

- `DiffuEraser`
- `ROSE`

同时，它还支持针对阴影、反射、镜像残影和边缘伪影的 side-effect-aware mask 扩张阶段。

## 目标

Part 3 的目标是在 Part 2 基线之上进一步提升视频中的人物移除效果，主要改进点包括：

- 将分割器从 `SAM 2` 升级到 `SAM 3` / `SAM 3.1`
- 引入基于 diffusion 的视频修复后端
- 显式处理被移除目标周围的副作用区域

## 已实现方法

主入口是 [run_part3.py](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/run_part3.py)。当前支持的方法包括：

- `part2_sam2_propainter`
- `sam3_propainter`
- `sam3_diffueraser_object`
- `sam3_diffueraser_side_effect`
- `sam3_rose_object`
- `sam3_rose_side_effect`

在最终 Part 3 实验中，重点使用的方法是：

- `sam3_diffueraser_object`
- `sam3_diffueraser_side_effect`
- `sam3_rose_object`
- `sam3_rose_side_effect`

同样的完整流程也通过独立配置在 `SAM 3.1` 上运行了一遍。

## 环境配置

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate cv2
```

如果你更想手动安装，也可以用：

```bash
conda create -n cv2 python=3.10 -y
conda activate cv2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_part3.txt
```

这一部分现在与 Part 1 和 Part 2 的环境配置风格保持一致。如果你的本地 CUDA / PyTorch 环境不同，可以按实际情况调整 PyTorch 安装命令。

### 2. 配置外部依赖仓库

```bash
bash scripts/setup_external_repos.sh
```

该脚本会准备 Part 3 所需的外部仓库：

- `SAM 3`
- `DiffuEraser`
- `ROSE`
- 来自 Part 2 的 `ProPainter` 复用路径

### 3. 准备模型权重

模型路径在 YAML 中使用绝对路径配置。当前环境下默认放在：

- `/hpc2hdd/home/ckwong627/workdir/models/sam3`
- `/hpc2hdd/home/ckwong627/workdir/models/sam3.1`
- `/hpc2hdd/home/ckwong627/workdir/models/diffuEraser`
- `/hpc2hdd/home/ckwong627/workdir/models/sd-vae-ft-mse`
- `/hpc2hdd/home/ckwong627/workdir/models/Wan2.1-Fun-1.3B-InP`
- `/hpc2hdd/home/ckwong627/workdir/models/ROSE_transformer`

## 数据

### DAVIS

DAVIS 用于以下序列的 mask 评估：

- `bmx-trees`
- `tennis`

标注读取路径为：

- `../data/DAVIS`

### Wild Video

paired wild videos 用于修复质量评估：

- `Wild_Video1`
- `Wild_Video2`

输入与 GT 在 [configs/default.yaml](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/configs/default.yaml) 中配置：

- 输入视频：`../data/Wild_Video/input_with_person`
- clean GT：`../data/Wild_Video/clean_gt_no_person`

clean ground truth 只用于评估，不参与推理。

## 目录结构

核心文件和目录如下：

- [configs/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/configs)
  - 默认配置、完整实验配置和消融配置
- [src/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/src)
  - pipeline 实现、wrapper、评估和工具代码
- [slurm_scripts/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts)
  - smoke test、方法运行和评估的 SLURM 脚本
- [outputs_ablation/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/outputs_ablation)
  - `SAM 3` vs `SAM 3.1` 的 smoke 消融结果
- [outputs_full/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/outputs_full)
  - 最终完整实验输出和汇总表
- [plan/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/plan)
  - 规划文档和执行规格书

## 外部依赖

Part 3 依赖以下外部仓库：

- `SAM 3`：`external/repository/sam3`
- `DiffuEraser`：`external/repository/DiffuEraser`
- `ROSE`：`external/repository/ROSE`
- `ProPainter`：复用 `../part2/external/ProPainter`

仓库准备脚本为：

- [scripts/setup_external_repos.sh](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/scripts/setup_external_repos.sh)

本项目是在 HPC 环境中开发的，部分后端可能需要独立环境或后端专用解释器。例如完整 `ROSE` 配置会通过 `rose.python_bin` 指向一个单独的 Python。

## 运行方式

### 本地运行

示例：

```bash
python run_part3.py \
  --config configs/sam3_full.yaml \
  --method sam3_diffueraser_side_effect \
  --sequence Wild_Video1 \
  --input ../data/Wild_Video/input_with_person/Wild_Video1.mp4
```

### SLURM 运行

推荐使用以下 HPC 入口脚本：

- [slurm_scripts/run_part3_method.slurm](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts/run_part3_method.slurm)
- [slurm_scripts/eval_part3_metrics.slurm](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts/eval_part3_metrics.slurm)

示例：

```bash
CONFIG=configs/sam3_full.yaml \
METHOD=sam3_rose_side_effect \
SEQUENCE=Wild_Video1 \
INPUT_VIDEO=../data/Wild_Video/input_with_person/Wild_Video1.mp4 \
sbatch slurm_scripts/run_part3_method.slurm
```

## 评估

使用 [evaluate_part3.py](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/evaluate_part3.py)。

### DAVIS

DAVIS 评估输出：

- `JM`
- `JR`

示例：

```bash
python evaluate_part3.py \
  --config configs/sam3_full.yaml \
  --method sam3_diffueraser_object \
  --evaluate-davis \
  --sequence bmx-trees tennis \
  --metrics-tag sam3_diffueraser_object
```

### Wild Video

Wild Video 评估输出：

- `PSNR`
- `SSIM`

示例：

```bash
python evaluate_part3.py \
  --config configs/sam3_full.yaml \
  --method sam3_rose_side_effect \
  --evaluate-wild \
  --no-align \
  --metrics-tag sam3_rose_side_effect
```

## 输出结构

主要输出根目录：

- `outputs/`
  - 本地 smoke 和中间调试运行
- `outputs_ablation/`
  - `SAM 3` vs `SAM 3.1` 的 smoke 消融
- `outputs_full/`
  - 最终完整实验结果

在 `outputs_full/<variant>/` 下面，最重要的是：

- `videos/`
- `masks/`
- `metrics/`

最终汇总表位于：

- [outputs_full/summary/part3_full_summary.md](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/outputs_full/summary/part3_full_summary.md)
- [outputs_full/summary/part3_results_table.md](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/outputs_full/summary/part3_results_table.md)

同目录下也提供了中文版。

## 最终结果

最终结果概括如下：

- 最好的 DAVIS mask 指标：`sam3.1` 系列
- 最好的 wild-video `PSNR`：`sam3_diffueraser_side_effect`
- 最好的 wild-video `SSIM`：`sam3_rose_side_effect`
- 最均衡的最终推荐方法：`sam3.1_rose_side_effect`

当前最终表如下：

| variant | method | avg_davis_JM | avg_davis_JR | avg_wild_PSNR | avg_wild_SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| sam3 | sam3_diffueraser_object | 0.6953 | 0.8625 | 14.3225 | 0.2270 |
| sam3 | sam3_diffueraser_side_effect | 0.6953 | 0.8625 | **14.3236** | 0.2271 |
| sam3 | sam3_rose_object | 0.6953 | 0.8625 | 14.2992 | 0.2294 |
| sam3 | sam3_rose_side_effect | 0.6953 | 0.8625 | 14.3121 | **0.2303** |
| sam3.1 | sam3_diffueraser_object | **0.6969** | **0.8750** | 14.3219 | 0.2270 |
| sam3.1 | sam3_diffueraser_side_effect | **0.6969** | **0.8750** | 14.3231 | 0.2271 |
| sam3.1 | sam3_rose_object | **0.6969** | **0.8750** | 14.3081 | 0.2302 |
| sam3.1 | sam3_rose_side_effect | **0.6969** | **0.8750** | 14.3179 | 0.2302 |

## 说明

- 同一 variant 内 DAVIS 指标完全一致，是因为 DAVIS 评估复用了同一份 `object_mask`。
- `PSNR` 是像素级重建误差相关指标。
- `SSIM` 是局部结构相似性指标。
- 一些历史 smoke-check 目录，例如 `wild_video1_short33_check` 和 `wild_video1_short33_chunked_check`，主要用于调试与验证，不属于最终 full-run 结果。
