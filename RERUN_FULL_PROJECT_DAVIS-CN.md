# 全量 DAVIS 重跑说明

English version: [RERUN_FULL_PROJECT_DAVIS.md](RERUN_FULL_PROJECT_DAVIS.md)

这份文档用于指导同学在不使用 `sbatch` 的情况下，直接通过 `python` 命令重跑整个项目。

## 范围

- 重跑 `part1`、`part2`、`part3`，覆盖 `data/DAVIS/JPEGImages/480p/` 下当前已经解压出的完整 DAVIS 数据。
- 最终项目指标只保留 DAVIS 的 `JM` 和 `JR`。
- 旧的 Wild Video 结果不再纳入最终对比总表。

## 为什么要重跑

- 之前 Part 1 和 Part 2 使用的 Wild Video 设置不够可靠，不能继续作为最终报告依据。
- Part 3 里 `Wild_Video1` 存在 input video 和 ground-truth 对调的问题。
- 因此这次干净的复现实验目标改为只重跑完整 DAVIS。

## 重要规则

- 这些命令应该在已经分配好 GPU 的机器上运行，或者在已经进入的计算节点里运行。
- 不要在 HPC 登录节点上直接跑重型 GPU 作业。
- 这次重跑的最终汇总表只保留 DAVIS 的 `JM` 和 `JR`。
- 旧版 `part1/README.md` 和 `part2/README.md` 里关于 Wild Video 的段落，这次复现时不要再作为最终实验口径。
- 如果换机器重跑，先修改项目根目录下的 `local_paths.sh`。这个文件统一收口了少数仍然必须依赖本机环境的绝对路径，比如 `conda.sh`、环境名和共享模型根目录。

## 输出目录

- `part1/results_davis_full/`
- `part2/results_davis_full/`
- `part3/results/DAVIS_Dataset/`
- `results_davis_summary/project_davis_summary.md`
- `results_davis_summary/project_davis_summary.csv`
- `results_davis_summary/project_davis_summary.json`

## Part 1

创建 Part 1 环境：

```bash
cd part1
source ../local_paths.sh
conda env create -f environment.yml
conda activate "$PART1_CONDA_ENV"
```

运行 Part 1 的三种条件，并覆盖完整 DAVIS：

```bash
python run.py --davis --config configs/davis_all_temporal_aligned.yaml --output results_davis_full/temporal_aligned
python evaluate.py --pred results_davis_full/temporal_aligned --davis-root ../data/DAVIS --save-json results_davis_full/temporal_aligned_metrics.json

python run.py --davis --config configs/davis_all_temporal_no_align.yaml --output results_davis_full/temporal_no_align
python evaluate.py --pred results_davis_full/temporal_no_align --davis-root ../data/DAVIS --save-json results_davis_full/temporal_no_align_metrics.json

python run.py --davis --config configs/davis_all_spatial_only.yaml --output results_davis_full/spatial_only
python evaluate.py --pred results_davis_full/spatial_only --davis-root ../data/DAVIS --save-json results_davis_full/spatial_only_metrics.json
```

说明：

- 这三份配置会自动发现 `../data/DAVIS/JPEGImages/480p/` 下的全部序列。
- 更新后的 Part 1 评估脚本现在只保留 `JM` 和 `JR`。

## Part 2

创建 Part 2 环境：

```bash
cd part2
source ../local_paths.sh
conda env create -f environment.yml
conda activate "$PART2_CONDA_ENV"
bash setup.sh
```

运行 Part 2 的两种方法，并覆盖完整 DAVIS：

```bash
python run.py --config configs/davis_all.yaml --method sam2 --output results_davis_full/sam2 --gpu 0
python evaluate.py --pred results_davis_full/sam2 --davis-root ../data/DAVIS --save-json results_davis_full/sam2_metrics.json

python run.py --config configs/davis_all.yaml --method vggt4d --output results_davis_full/vggt4d --gpu 0
python evaluate.py --pred results_davis_full/vggt4d --davis-root ../data/DAVIS --save-json results_davis_full/vggt4d_metrics.json
```

说明：

- `configs/davis_all.yaml` 是这次整库重跑专用配置，会自动发现当前所有 DAVIS 序列。
- 更新后的 Part 2 评估脚本现在只保留 `JM` 和 `JR`。

## Part 3

创建 Part 3 环境：

```bash
cd part3
source ../local_paths.sh
conda env create -f environment.yml
conda activate "$PART3_CONDA_ENV"
```

先确认外部仓库和权重都已经准备好：

- `external/repository/sam3`
- `external/repository/DiffuEraser`
- `external/repository/ROSE`
- `../part2/external/ProPainter`
- `models/sam3/`
- `models/sam3.1/` 可选，仅在你要补消融时才需要
- `models/diffuEraser/`
- `models/Wan2.1-Fun-1.3B-InP/`
- `models/ROSE_transformer/`
- `models/sd-vae-ft-mse/`
- `models/stable-diffusion-v1-5/`

模型下载默认规则：

- 默认优先使用 `ModelScope`
- 在当前这套配置里，`SAM 3` 可以直接从 `ModelScope` 下载，因此如果走默认路径，就不需要 `Hugging Face` 访问审批
- `SAM 3.1` 是可选项，只有你想补消融时才需要准备
- 如果某个权重当前仍没有被这套 `ModelScope` helper 覆盖，再使用 `part3/README-CN.md` 里写的可选 `Hugging Face` 下载方式

推荐的模型准备命令：

```bash
bash setup.sh
```

这是推荐路径。它会先 clone 所需仓库，再下载 `SAM 3` 主线和当前已经确认好的 `ModelScope` 模型，并在 `part3/models/` 下把其余目录也准备好。

如果你还想保留 `SAM 3.1` 的可选消融路径，再额外执行：

```bash
bash setup.sh --include-sam3-1
```

如果仓库已经 clone 好，只想单独下载模型，也可以执行：

```bash
bash scripts/download_models.sh
```

如果还想一起准备 `SAM 3.1` 的可选消融权重，再额外执行：

```bash
bash scripts/download_models.sh --include-sam3-1
```

先把 DAVIS 帧目录转换成 mp4：

```bash
python scripts/prepare_davis_videos.py --davis-root ../data/DAVIS --output-dir inputs/davis_videos
```

运行 SAM 3 的完整 DAVIS 实验：

```bash
python scripts/run_all_davis_methods.py --config configs/sam3_davis_all.yaml --gpu 0
```

SLURM 说明：

- 短时间 `debug` 分区检查请使用 `slurm_scripts/run_part3_sequence_matrix_debug.slurm`。
- 完整 DAVIS 和其它较长的 sequence-matrix 任务请使用 `slurm_scripts/run_part3_sequence_matrix.slurm`。

说明：

- 这个脚本会使用 DAVIS 第一帧的 GT mask，自动构造 bbox 和 point 来初始化 SAM 3。
- 这次 DAVIS-only 重跑里，Part 3 的 `JM/JR` 只从 mask 输出上评估。
- Part 3 的四种方法仍然都会真正执行，并各自生成输出文件，统一写到 `part3/results/DAVIS_Dataset/`。
- 但是 DAVIS 打分仍然只看 object mask 输出，inpainting backend 不参与 DAVIS `JM/JR` 计算。
- 模型下载默认优先使用 `ModelScope`。`SAM 3` 属于默认重跑主线；`SAM 3.1` 只保留为可选消融。

如果你确实还想补 `SAM 3.1` 的可选消融，再单独运行：

```bash
python scripts/run_all_davis_methods.py --config configs/sam3_1_davis_all.yaml --gpu 0
```

## 生成跨 Part 最终总表

在项目根目录运行：

```bash
python scripts/build_project_davis_summary.py
```

会生成：

- `results_davis_summary/project_davis_summary.md`
- `results_davis_summary/project_davis_summary.csv`
- `results_davis_summary/project_davis_summary.json`

## 以后如果还要重跑 Wild Video，先做人工核对

在任何新的 Wild Video 实验开始前，先确认：

1. `data/Wild_Video/input_with_person/Wild_Video1.mp4` 是否真的是输入视频。
2. `data/Wild_Video/clean_gt_no_person/Wild_Video1-Ground_Truth.mp4` 是否真的是干净 GT。
3. 只有把对调问题修正后，才继续做新的 Wild Video 重跑。

## 建议的重跑顺序

1. `part1`
2. `part2`
3. `part3` 的 `sam3`
4. `scripts/build_project_davis_summary.py`

可选：

5. `part3` 的 `sam3.1` 消融
6. `python scripts/build_project_davis_summary.py --include-sam3-1`
