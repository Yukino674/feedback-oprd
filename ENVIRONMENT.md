# 实际训练环境

`requirements_repro.txt` 是面向复现者的精简安装清单。`environment.yml` 和
`requirements_atod_oprd_actual.txt` 是从服务器当前使用的 `atod-oprd` 环境导出的
版本快照，仅用于记录和排查，不建议直接在其他机器上创建环境。

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
conda create -n atod-oprd python=3.12 pip -y
conda activate atod-oprd
pip install -r requirements_repro.txt
pip install --no-build-isolation flash-attn==2.7.4.post1
pip install --no-deps -e .
```

该清单面向 Linux + NVIDIA CUDA 环境。GPU 驱动需要支持 CUDA 12.8；如果集群的
驱动或 GPU 架构不同，应保留 Python 依赖版本，同时按照本机 CUDA 版本调整
PyTorch、Flash-Attention、xformers 和 vLLM。

原来的atod仓库环境保存在 `environment_atod_upstream.yml`，仅作为对照，不是本实验
的推荐安装入口。
