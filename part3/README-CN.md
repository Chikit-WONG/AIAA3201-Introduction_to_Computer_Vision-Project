# Part 3

Part 3 是本课程项目的最终视频目标移除流水线实现。正式主线使用 `SAM 3` 进行视频掩码生成，再接以下两类修复后端之一：

- `DiffuEraser`
- `ROSE`

同时，它还支持针对阴影、反射、镜像残影和边缘伪影的 side-effect-aware mask 扩张阶段。

## 目标

Part 3 的目标是在 Part 2 基线之上进一步提升视频中的人物移除效果，主要改进点包括：

- 将分割器从 `SAM 2` 升级到 `SAM 3`
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

`SAM 3.1` 现在只保留为可选消融路径，不再属于默认必跑主线。

## 环境配置

如果要在另一台机器上运行 Part 3，先修改 `../local_paths.sh`。项目里大多数路径已经改成相对路径；这个文件专门集中管理少数仍然必须依赖本机环境的绝对路径，比如 `conda.sh` 位置、环境名，以及可选的 legacy shared model 根目录。

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
source ../local_paths.sh
conda activate "$PART3_CONDA_ENV"
```

如果你更想手动安装，也可以用：

```bash
conda create -n cv2 python=3.10 -y
conda activate cv2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_part3.txt
```

这一部分现在与 Part 1 和 Part 2 的环境配置风格保持一致。如果你的本地 CUDA / PyTorch 环境不同，可以按实际情况调整 PyTorch 安装命令。

### 2. 一键配置

现在最推荐的配置方式是直接运行：

```bash
conda env create -f environment.yml
source ../local_paths.sh
conda activate "$PART3_CONDA_ENV"
bash setup.sh
```

这个脚本会：

- clone 所有需要的外部仓库
- 安装辅助下载所需的 Python 包
- 把模型权重下载到 `models/`
- 默认优先使用 `ModelScope`
- 默认会下载 `SAM 3` 主线权重
- 对当前项目里还没有确认 `ModelScope` 镜像的权重，自动回退到 `Hugging Face`
- 默认假设你已经激活了 Part 3 的 conda 环境

如果你还想保留 `SAM 3.1` 这条可选消融路径，可以执行：

```bash
source ../local_paths.sh
conda activate "$PART3_CONDA_ENV"
bash setup.sh --include-sam3-1
```

如果你想强制优先走 `Hugging Face`，可以运行：

```bash
source ../local_paths.sh
conda activate "$PART3_CONDA_ENV"
bash setup.sh --source hf
```

如果你想保留自动回退策略，也可以显式写成：

```bash
source ../local_paths.sh
conda activate "$PART3_CONDA_ENV"
bash setup.sh --source auto
```

### 3. 分开配置外部依赖仓库

```bash
bash scripts/setup_external_repos.sh
```

该脚本会把 Part 3 所需的外部仓库 clone 到当前配置实际使用的位置：

- `SAM 3` -> `part3/external/repository/sam3`
- `DiffuEraser` -> `part3/external/repository/DiffuEraser`
- `ROSE` -> `part3/external/repository/ROSE`
- `ProPainter` -> `part2/external/ProPainter`

外部仓库地址如下：

- `SAM 3`：`https://github.com/facebookresearch/sam3.git`
- `DiffuEraser`：`https://github.com/lixiaowen-xw/DiffuEraser.git`
- `ROSE`：`https://github.com/Kunbyte-AI/ROSE.git`
- `ProPainter`：`https://github.com/sczhou/ProPainter.git`

如果你想手动 clone，也可以直接执行：

```bash
git clone https://github.com/facebookresearch/sam3.git \
  part3/external/repository/sam3

git clone https://github.com/lixiaowen-xw/DiffuEraser.git \
  part3/external/repository/DiffuEraser

git clone https://github.com/Kunbyte-AI/ROSE.git \
  part3/external/repository/ROSE

git clone https://github.com/sczhou/ProPainter.git \
  part2/external/ProPainter
```

### 4. 分开准备模型权重

自动下载脚本：

```bash
bash scripts/download_models.sh
```

如果你还想一起准备 `SAM 3.1` 的可选消融权重：

```bash
bash scripts/download_models.sh --include-sam3-1
```

现在默认模型根目录是 `part3/` 下的相对路径 `models/`。
该脚本会**优先使用 ModelScope** 下载本项目里已经明确确认过有 ModelScope 镜像的模型。

模型路径在 YAML 中使用相对路径配置。当前默认放在：

- `models/sam3`
- `models/sam3.1` 可选，仅用于消融
- `models/diffuEraser`
- `models/sd-vae-ft-mse`
- `models/stable-diffusion-v1-5`
- `models/Wan2.1-Fun-1.3B-InP`
- `models/ROSE_transformer`

模型来源与目标目录如下：

| 组件 | 下载地址 | 下载到哪里 | 说明 |
| --- | --- | --- | --- |
| `SAM 3` checkpoint bundle | `https://modelscope.cn/models/facebook/sam3` | `models/sam3` | 本项目默认下载源。`Hugging Face` 仍可选：`https://huggingface.co/facebook/sam3`。 |
| `SAM 3.1` checkpoint bundle | `https://modelscope.cn/models/facebook/sam3.1` | `models/sam3.1` | 可选的消融权重。`Hugging Face` 仍可选：`https://huggingface.co/facebook/sam3.1`。 |
| `DiffuEraser` 权重 | `https://www.modelscope.cn/models/xingzi/diffuEraser` | `models/diffuEraser` | 可由 `scripts/download_models.sh` 自动下载。 |
| `sd-vae-ft-mse` | `https://huggingface.co/stabilityai/sd-vae-ft-mse` | `models/sd-vae-ft-mse` | `DiffuEraser` 所需。需要手动走 upstream 下载。 |
| `stable-diffusion-v1-5` 基础模型 | `https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5` | `models/stable-diffusion-v1-5` | `DiffuEraser` 所需，体积较大。需要手动走 upstream 下载。 |
| `Wan2.1-Fun-1.3B-InP` 基础模型 | `https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP` | `models/Wan2.1-Fun-1.3B-InP` | 可由 `scripts/download_models.sh` 自动下载。体积较大。 |
| `ROSE` transformer 权重 | `https://huggingface.co/Kunbyte/ROSE` | `models/ROSE_transformer` | 当前这套项目里属于 upstream-only。 |

手动下载命令如下：

```bash
pip install modelscope

python - <<'PY'
from modelscope import snapshot_download
snapshot_download('facebook/sam3', local_dir='models/sam3')
snapshot_download('xingzi/diffuEraser', local_dir='models/diffuEraser')
snapshot_download('PAI/Wan2.1-Fun-1.3B-InP', local_dir='models/Wan2.1-Fun-1.3B-InP')
PY
```

如果要补 `SAM 3.1` 的可选消融权重，再额外执行：

```bash
python - <<'PY'
from modelscope import snapshot_download
snapshot_download('facebook/sam3.1', local_dir='models/sam3.1')
PY
```

其余模型的 upstream 手动下载命令如下：

```bash
hf auth login

hf download stabilityai/sd-vae-ft-mse --local-dir models/sd-vae-ft-mse

hf download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir models/stable-diffusion-v1-5

hf download Kunbyte/ROSE --local-dir models/ROSE_transformer
```

如果你仍然想手动从 `Hugging Face` 下载 `SAM 3.1` 的可选消融权重，可以额外执行：

```bash
hf auth login
hf download facebook/sam3.1 --local-dir models/sam3.1
```

说明：

- 默认优先下载源是 ModelScope，前提是本项目里已经明确确认到对应的 ModelScope 镜像。
- 在当前这套配置里，`SAM 3` 可以直接走 ModelScope，因此默认路径下不需要先申请 Hugging Face 权限。
- `SAM 3.1` 仍然保留支持，但只建议在可选消融实验里使用。
- `stable-diffusion-v1-5` 和 `Wan2.1-Fun-1.3B-InP` 体积较大，下载前请先确认磁盘空间足够。
- 本项目中的 `ROSE` 配置虽然通过 `rose.python_bin` 指向单独解释器，但模型文件仍统一下载到共享模型目录。

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
- [outputs/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/results)
  - 唯一正式输出根目录，下面按 `full/`、`ablation/` 和未来的 `davis_full/` 分类
- `artifacts_debug/`
  - 非最终 smoke 检查、历史本地运行结果和归档的 SLURM 日志
- [plan/](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/plan)
  - 规划文档和执行规格书

## 外部依赖

Part 3 依赖以下外部仓库：

- `SAM 3`：`external/repository/sam3`
- `DiffuEraser`：`external/repository/DiffuEraser`
- `ROSE`：`external/repository/ROSE`
- `ProPainter`：复用 `../part2/external/ProPainter`

推荐的一键入口脚本为：

- [setup.sh](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/setup.sh)

只负责仓库 clone 的脚本为：

- [scripts/setup_external_repos.sh](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/scripts/setup_external_repos.sh)

模型下载脚本为：

- [scripts/download_models.sh](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/scripts/download_models.sh)

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

对于 sequence-matrix 任务：

- 短时间 `debug` 分区检查请使用 [slurm_scripts/run_part3_sequence_matrix_debug.slurm](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts/run_part3_sequence_matrix_debug.slurm)。
- 完整 DAVIS 和其它较长任务请使用 [slurm_scripts/run_part3_sequence_matrix.slurm](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/slurm_scripts/run_part3_sequence_matrix.slurm)。

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

- `results/Wild_Video/`
  - 最终完整实验结果和最终汇总表
- `results_debug/ablation/`
  - 可选的 `SAM 3` vs `SAM 3.1` smoke 消融
- `results/DAVIS_Dataset/`
  - 预留给全 DAVIS 重跑结果
- `artifacts_debug/`
  - 非最终 smoke 检查、归档的本地实验和 SLURM 日志

在 `results/Wild_Video/<variant>/` 下面，最重要的是：

- `videos/`
- `masks/`
- `metrics/`

最终汇总表位于：

- [results/Wild_Video/summary/part3_full_summary.md](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/results/Wild_Video/summary/part3_full_summary.md)
- [results/Wild_Video/summary/part3_results_table.md](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/part3/results/Wild_Video/summary/part3_results_table.md)

同目录下也提供了中文版。

## 最终结果

最终结果概括如下：

- 主线最终推荐方法：`sam3_rose_side_effect`
- 主线里 wild-video `PSNR` 最好的是：`sam3_diffueraser_side_effect`
- 主线里 wild-video `SSIM` 最好的是：`sam3_rose_side_effect`
- `SAM 3.1` 只保留为可选消融，不再纳入默认必跑和默认汇总

当前最终表如下：

| variant | method | avg_davis_JM | avg_davis_JR | avg_wild_PSNR | avg_wild_SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| sam3 | sam3_diffueraser_object | 0.6953 | 0.8625 | 14.3225 | 0.2270 |
| sam3 | sam3_diffueraser_side_effect | 0.6953 | 0.8625 | **14.3236** | 0.2271 |
| sam3 | sam3_rose_object | 0.6953 | 0.8625 | 14.2992 | 0.2294 |
| sam3 | sam3_rose_side_effect | 0.6953 | 0.8625 | 14.3121 | **0.2303** |

## 说明

- 同一 variant 内 DAVIS 指标完全一致，是因为 DAVIS 评估复用了同一份 `object_mask`。
- `PSNR` 是像素级重建误差相关指标。
- `SSIM` 是局部结构相似性指标。
- 默认报告和默认重跑路径现在只使用 `SAM 3`。`SAM 3.1` 仅作为可选消融材料保留。
- 一些历史 smoke-check 目录，例如 `wild_video1_short33_check` 和 `wild_video1_short33_chunked_check`，主要用于调试与验证，不属于最终 full-run 结果。
