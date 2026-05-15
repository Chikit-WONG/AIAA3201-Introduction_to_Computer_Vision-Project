# Part 2：高级视频目标移除

## 概述

本部分实现两条高级视频目标移除流水线，并在 DAVIS 2017 数据集上比较：

| 流水线 | 掩码提取 | 视频修复 |
|---|---|---|
| **Pipeline A** | VGGT4D | ProPainter |
| **Pipeline B** | SAM 2 | ProPainter |

评估指标包括：`JM`、`JR`、`PSNR`、`SSIM`、`PSNR_masked`、`SSIM_masked`。

## 环境配置

```bash
conda env create -f environment.yml
conda activate cv2
```

或手动创建环境：

```bash
conda create -n cv2 python=3.10 -y
conda activate cv2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 外部依赖配置

```bash
bash setup.sh
```

## 运行方式

### DAVIS 正式结果

```bash
python run.py --method sam2
python run.py --method vggt4d
python run.py --method sam2 --sequences bmx-trees tennis
```

### Wild Video 正式结果

```bash
sbatch slurm_sam2_wild.sh
sbatch slurm_vggt4d_wild.sh
```

### 评估

```bash
python evaluate.py --pred results/results_davis_full/sam2 --davis-root ../data/DAVIS
python evaluate.py --pred results/results_davis_full/vggt4d --davis-root ../data/DAVIS
python evaluate.py --pred results/results_davis_full/vggt4d --pred2 results/results_davis_full/sam2 --davis-root ../data/DAVIS
python evaluate.py --pred results/results_davis_full/sam2 --save-json results/results_davis_full/sam2_metrics.json
```

## 输出目录统一规则

现在所有正式输出统一为：

```
results/
├── results_davis_full/
│   └── <method>/<sequence>/
├── results_sample_data/
│   └── <method>/<sequence>/
└── results_wild_video/
    └── <method>/<sequence>/
```

说明：

- `results/results_davis_full/`：完整 DAVIS 正式结果
- `results/results_sample_data/`：课程 sample data，也就是 `bmx-trees` 和 `tennis`，从正式 DAVIS 结果复制而来
- `results/results_wild_video/`：Wild Video 正式结果

每个序列目录下现在包含：

```
masks/
frames/
inpainted.mp4
```

也就是说，除了逐帧 PNG，现在代码会自动生成 `inpainted.mp4`。

如果你需要给旧结果批量补视频，可以运行：

```bash
python pack_result_videos.py --root results
```

## Wild Video 说明

当前重跑中，paired wild-video 集为：

- `Wild_Video1`
- `Wild_Video2`

对应正式结果路径为：

- `results/results_wild_video/sam2/`
- `results/results_wild_video/vggt4d/`

## 备注

`results/results_wild_video/sam2_smoke` 和 `results/results_wild_video/vggt4d_smoke` 属于早期 smoke/debug 结果，不属于正式结果目录。正式提交或整理结果时一般不需要保留；除非你还想保留早期快速验证痕迹，否则可以删除或移到单独 debug 归档目录。
