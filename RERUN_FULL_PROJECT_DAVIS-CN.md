# 全量 DAVIS 重跑说明

English version: [RERUN_FULL_PROJECT_DAVIS.md](/hpc2hdd/home/ckwong627/workdir/Class/AIAA3201_L01_Introduction_to_Computer_Vision/Project/Group-Project/AIAA3201-Introduction_to_Computer_Vision-Project/RERUN_FULL_PROJECT_DAVIS.md)

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

## 输出目录

- `part1/results_davis_full/`
- `part2/results_davis_full/`
- `part3/outputs_davis_full/`
- `results_davis_summary/project_davis_summary.md`
- `results_davis_summary/project_davis_summary.csv`
- `results_davis_summary/project_davis_summary.json`

## Part 1

创建 Part 1 环境：

```bash
cd part1
conda env create -f environment.yml
conda activate cv
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
conda env create -f environment.yml
conda activate cv-2
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
conda env create -f environment.yml
conda activate cv3
```

先确认外部仓库和权重都已经准备好：

- `external/repository/sam3`
- `external/repository/DiffuEraser`
- `external/repository/ROSE`
- `../part2/external/ProPainter`
- `/hpc2hdd/home/ckwong627/workdir/models/sam3/`
- `/hpc2hdd/home/ckwong627/workdir/models/sam3.1/`
- `/hpc2hdd/home/ckwong627/workdir/models/diffuEraser/`
- `/hpc2hdd/home/ckwong627/workdir/models/Wan2.1-Fun-1.3B-InP/`
- `/hpc2hdd/home/ckwong627/workdir/models/ROSE_transformer/`
- `/hpc2hdd/home/ckwong627/workdir/models/sd-vae-ft-mse/`

先把 DAVIS 帧目录转换成 mp4：

```bash
python scripts/prepare_davis_videos.py --davis-root ../data/DAVIS --output-dir inputs/davis_videos
```

运行 SAM 3 的完整 DAVIS 实验：

```bash
python scripts/run_all_davis_methods.py --config configs/sam3_davis_all.yaml --gpu 0
```

运行 SAM 3.1 的完整 DAVIS 实验：

```bash
python scripts/run_all_davis_methods.py --config configs/sam3_1_davis_all.yaml --gpu 0
```

说明：

- 这个脚本会使用 DAVIS 第一帧的 GT mask，自动构造 bbox 和 point 来初始化 SAM 3。
- 这次 DAVIS-only 重跑里，Part 3 的 `JM/JR` 只从 mask 输出上评估。
- 因此 Part 3 里四种方法在 DAVIS 上会复用同一套 object masks，inpainting backend 不参与 DAVIS `JM/JR` 计算。

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
4. `part3` 的 `sam3.1`
5. `scripts/build_project_davis_summary.py`
