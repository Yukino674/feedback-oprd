# 实际训练环境

`environment.yml` 是从服务器当前使用的 `atod-oprd` Conda 环境导出的版本清单，
不是根据 ATOD 示例猜测的依赖。`requirements_atod_oprd_actual.txt` 是同一环境的
Python pip 包清单，便于排查 Conda 导出或 CUDA 安装问题。

## 关键版本

- Python 3.12.13
- CUDA toolkit 12.8
- PyTorch 2.8.0
- vLLM 0.11.0
- Ray 2.50.0
- Transformers 4.57.3
- Flash-Attention 2.7.4.post1
- xformers 0.0.32.post1
- tensordict 0.10.0
- hydra-core 1.3.5
- alfworld 0.4.2
- SwanLab 0.9.4

## 安装

```bash
conda env create -f environment.yml
conda activate atod-oprd
```

该清单面向 Linux + NVIDIA CUDA 环境。GPU 驱动需要支持 CUDA 12.8；如果集群的
驱动或 GPU 架构不同，应保留 Python 依赖版本，同时按照本机 CUDA 版本调整
PyTorch、Flash-Attention、xformers 和 vLLM。

原来的仓库环境快照保存在 `environment_atod_upstream.yml`，仅作为对照，不是本实验
的推荐安装入口。
