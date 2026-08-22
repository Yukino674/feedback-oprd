# ALFWorld 上的 OPRD-Bridge

本目录包含 Qwen3 ALFWorld 实验所需的训练代码、ATOD 框架和 Slurm 启动脚本。

仓库根目录已经包含运行所需的 ATOD 核心包（`verl`、`agent_system`）、ALFWorld
skills 以及相关示例。自定义的 OPRD-Bridge 模块已经放在正确的 Python 导入路径下，
安装环境后可以从仓库根目录启动训练。

## 目录说明

- `stepwise_feedback/`：逐步教师反馈引导的 OPRD-Bridge。学生先提出动作，教师给出
  反馈，学生重写当前 response，环境只执行重写后的动作，并使用重写后的 response
  进行 hidden-state 监督。
- `hidden_only/`：不使用逐步反馈的 OPRD-Bridge hidden-only 基线。
- `bridge_bank/`：构建 all-layer bridge bank 所需的脚本和配置。
- `configs/`：训练脚本使用的实验配置。
- `patches/`：实验所需的 ATOD/verl 补丁快照。
- `examples/`：OPD、SOD 和 ATOD 训练示例脚本。
- `data/`：较小的训练/验证 parquet 文件；原始 ALFWorld 数据不包含在仓库中。
- `artifacts/bridge_bank/`：rank-64 的 `ps_bank.pt` bridge bank。
- `ENVIRONMENT.md`：从实际 `atod-oprd` 环境导出的关键依赖说明。

## Bridge Bank 构建

`ps_bank.pt` 基于 ALFWorld 的 TCOD/SDAR turn-level experience buffer 构建：先收集
训练 step 的 student/teacher response hidden states，再提取所有 decoder
层的 response-token 表示。对 teacher 表示拟合 PCA 子空间，并为每一对 student/teacher
层训练线性投影器，使 student hidden 能对齐到 teacher 的低秩表示。构建脚本会生成多个
rank 版本；本仓库使用的是 rank-64 的 `artifacts/bridge_bank/ps_bank.pt`。

对应脚本和配置位于 `bridge_bank/`：

- `build_all_layers_bridge_bank_formal.py`
- `oprd_bridge_construction_sdar_clean_b16tb16_r8_f250.yaml`


## 外部依赖

训练需要准备以下外部资源：

- Qwen3-1.7B 学生模型
- Qwen3-8B 教师模型
- 原始 ALFWorld 数据目录
- CUDA GPU、Ray、vLLM 和 Flash-Attention 等运行依赖

原始 ALFWorld 数据较大，因此没有上传到仓库。运行前请设置：

```bash
export ALFWORLD_DATA=/path/to/alfworld
```

该目录应包含标准 ALFWorld 的 `json_2.1.1`、`detectors` 和 `logic` 子目录。

## 安装环境

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate atod-oprd
```

`environment.yml` 是当前实验实际使用的环境清单；完整 pip 清单保存在
`requirements_atod_oprd_actual.txt`。原 ATOD 环境快照另存为
`environment_atod_upstream.yml`，不作为本实验的默认安装文件。

训练入口从 ATOD 环境启动，例如：

```bash
python3 -m verl.trainer.main_sod_oprd_bridge_stepwise_feedback
```

具体 Hydra 参数保存在对应的 `.sbatch` 启动脚本中。

如果当前集群无法访问 `environment.yml` 中的内部 Conda 镜像，可以根据
`requirements.txt` 在已有的 PyTorch/CUDA 环境中安装 Python 依赖，并确保
PyTorch、vLLM、Ray 和 Flash-Attention 版本相互兼容。

## 运行训练

通用 Slurm 脚本支持以下环境变量：

- `ATOD_REPO`
- `CONDA_ENV`
- `STUDENT_MODEL_PATH`
- `TEACHER_MODEL_PATH`
- `BRIDGE_BANK_PATH`
- `TRAIN_FILE`
- `VAL_FILE`
- `ALFWORLD_DATA`

parquet 文件和 bridge bank 默认使用仓库内的路径。设置好 ALFWorld 数据和模型路径后，
可以提交：

```bash
sbatch stepwise_feedback/run_formal.sbatch
```

或者运行 hidden-only 基线：

```bash
sbatch hidden_only/run_formal.sbatch
```


