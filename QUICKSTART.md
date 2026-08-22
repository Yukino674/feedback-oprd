# 快速开始

以下步骤用于在普通 Linux 多 GPU 服务器上运行 Step-wise Feedback-Guided OPRD-Bridge，
不需要 Slurm。

## 1. 安装环境

```bash
git clone https://github.com/Yukino674/feedback-oprd.git
cd feedback-oprd
git lfs install
git lfs pull

conda env create -f environment.yml
conda activate atod-oprd
```

如果 Conda 环境已经存在，可以跳过 `conda env create`。

## 2. 准备 ALFWorld 数据

建议将完整的 ALFWorld 数据解压到当前用户的 `$HOME/data/alfworld`。例如，
如果用户名是 `alice`，实际目录就是 `/home/alice/data/alfworld`：

```text
$HOME/data/alfworld/
├── json_2.1.1/
├── logic/
└── detectors/
```

脚本默认使用这个位置，因此通常不需要修改 `ALFWORLD_DATA`。如果数据放在其他
目录，再把脚本顶部的 `ALFWORLD_DATA` 改成实际目录。

## 3. 设置路径

打开 `stepwise_feedback/run_formal.sh` 顶部的 `USER SETTINGS` 区域，填写：

```bash
STUDENT_MODEL_PATH="/path/to/Qwen3-1.7B"
TEACHER_MODEL_PATH="/path/to/Qwen3-8B"
ALFWORLD_DATA="/path/to/alfworld"
CONDA_ENV="atod-oprd"
CONDA_SH="/path/to/miniconda3/etc/profile.d/conda.sh"
```

仓库内的 `ps_bank.pt`、`train.parquet` 和 `test.parquet` 已经有默认路径，完成
`git lfs pull` 后通常不需要修改。

## 4. 登录 WandB

```bash
wandb login
```

脚本默认使用 WandB online 模式。

## 5. 启动训练

准备好 8 张可见 GPU 后执行：

```bash
bash stepwise_feedback/run_formal.sh
```
