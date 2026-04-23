# FlashAttention on Chipyard

English Version: [README.md](README.md)。

## 目录

1. [项目概述](#项目概述)
2. [项目实现内容](#项目实现内容)
3. [技术细节](#技术细节)
4. [仓库结构](#仓库结构)
5. [环境与依赖](#环境与依赖)
6. [使用方法](#使用方法)
7. [实验数据](#实验数据)
8. [范围与限制](#范围与限制)

## 项目概述

本仓库实现了面向边缘端部署的 Flash Attention 前向计算算子，目标平台为基于 Chipyard 的 RISC-V 异构系统。

整体计算分工如下：

- Gemmini NPU 负责矩阵乘法。
- Saturn RVV 负责向量化 softmax 相关计算。
- Rocket 核负责控制流与裸机执行。

项目支持两条 softmax 精度路径：

- FP32 softmax。
- FP16 softmax。

两条路径都已集成到同一前向流程中，可通过编译宏切换。

## 项目实现内容

本项目的核心实现包括：

1. 边缘端 Int Flash Attention 前向计算（含分块与 causal mask）。
2. Q/K/V 的量化数据流（int8 + 缩放因子），以及反量化与归一化流程。
3. 基于 RVV 的 Online Safe Softmax。
4. Schraudolph 指数近似（FP32 与 FP16 两套实现）。
5. 单头前向全流程及分阶段周期统计。
6. Python 参考模型与 golden 对比脚本，支持精度复现。

## 技术细节

### 1. 硬件配置入口

Chipyard 配置类位于 [config/CustomConfigs.scala](config/CustomConfigs.scala)：

- `GemminiRocketSaturnConfig`：Gemmini + Saturn RVV + Rocket 基础配置。
- `GemminiRocketSaturnConfigWithFP16`：在基础配置上启用 Rocket FP16 支持。

### 2. C 端核心实现

核心实现位于 [source/flash_attention.c](source/flash_attention.c) 和 [source/flash_attention.h](source/flash_attention.h)：

- `rvv_softmax_fp32`：FP32 的 online safe softmax。
- `rvv_softmax_fp16`：FP16 的 online safe softmax。
- `flash_attention_forward_inner`：分块内层核心计算。
- `flash_attention_forward_single_head`：单头前向流程。
- `flash_attention_forward`：多头封装接口。

[source/flash_attention.h](source/flash_attention.h) 中的关键编译宏：

- `SEQ_LEN`、`HEAD_DIM`：序列长度与头维度。
- `BR`、`BC`：分块大小。
- `USE_FP16`：softmax 精度路径开关。

### 3. 单头运行流程

每个分块按照以下顺序执行：

1. 在 Gemmini 上执行 `QK^T`（int8 x int8 -> int32）。
2. 在 RVV 上执行缩放/反量化、causal mask 与 online safe softmax。
3. 在 Gemmini 上执行 `P*V`。
4. 执行输出累加、反量化与最终归一化。

### 4. Python 参考与验证脚本

- [golden/flash_attention.py](golden/flash_attention.py)：Python C-model、随机输入生成、输出误差对比。
- [golden/llama_python.py](golden/llama_python.py)：模型级验证流程。
- [golden/llama_c.py](golden/llama_c.py)：C 侧联调导出流程。

## 仓库结构

```text
.
├── source/
│   ├── flash_attention.c
│   ├── flash_attention.h
│   ├── matmul_cycle_cnt.c
│   ├── softmax_cpu_cycle_cnt.c
│   ├── softmax_rvv_cycle_cnt.c
│   ├── random_data.h
│   └── llama_data.h
├── config/
│   └── CustomConfigs.scala
├── golden/
│   ├── flash_attention.py
│   ├── llama_c.py
│   └── llama_python.py
├── analysis/
│   ├── single_head_acc_evaluation.ipynb
│   ├── single_head_cycles_cnt.ipynb
│   └── data/
│       ├── single_head_cycles_cnt_fp16.xlsx
│       ├── single_head_cycles_cnt_fp32.xlsx
│       └── single_head_outputs/
│           ├── fp16/
│           └── fp32/
└── others/
    ├── clip_issue.ipynb
    ├── expp/
    └── quantized_flash_attention_with_recomputation/
```

## 环境与依赖

本项目推荐在 Chipyard 环境中完成硬件配置与裸机程序编译，再在 Verilator 中运行仿真。

需要的环境条件：

1. 可用的 Chipyard 工作环境（含 Gemmini 和 Saturn RVV 支持）。
2. 使用 [config/CustomConfigs.scala](config/CustomConfigs.scala) 里的自定义配置类生成硬件。
3. RISC-V 裸机编译工具链：
   - FP32 softmax：可使用 Chipyard 默认 GCC。
   - FP16 softmax：***需要 GCC 15.2***。
4. Chipyard 的 Verilator 仿真流程可正常运行。

## 使用方法

考虑到每个 Chipyard 工程目录、脚本入口与工具链路径不同，本节只提供可复用的操作流程。

### 步骤 1：准备 Chipyard 环境

确保 Chipyard 环境可用，且包含 Gemmini 与 RVV 相关支持。

### 步骤 2：选择硬件配置

在 Chipyard 构建流程中选择 [config/CustomConfigs.scala](config/CustomConfigs.scala) 中的配置类：

- 基础流程可使用 `GemminiRocketSaturnConfig`。
- 需要 FP16 softmax 时，使用 `GemminiRocketSaturnConfigWithFP16`。

### 步骤 3：设置精度模式与编译参数

在裸机程序编译设置中传入 [source/flash_attention.h](source/flash_attention.h) 使用的宏参数，例如：

- `USE_FP16=false`：FP32 softmax 路径。
- `USE_FP16=true`：FP16 softmax 路径。
- `SEQ_LEN`、`HEAD_DIM`、`BR`、`BC`：按实验配置设置。

### 步骤 4：编译裸机程序

以 [source/flash_attention.c](source/flash_attention.c)为目标编译裸机二进制文件。

工具链建议：

- FP32 Softmax 可使用 Chipyard 默认 GCC。
- FP16 Softmax 编译前需切换到 GCC 15.2。

### 步骤 5：在 Verilator 中运行

将裸机程序放入 Chipyard Verilator 流程运行，采集串口/控制台输出。

程序会输出：

- 总周期数。
- 分阶段周期（`QK^T`、softmax、`P*V`、累加反量化、归一化）。

### 步骤 6：与 Python/Golden 结果对比

使用 [golden](golden) 与 [analysis](analysis) 下脚本和 notebook，完成 C 输出与 Python、PyTorch golden 的精度对比。

## 实验数据

### 1. 精度指标（single head，seq_len=1024，head_dim=64）

来源：[analysis/single_head_acc_evaluation.ipynb](analysis/single_head_acc_evaluation.ipynb)

| 指标（对比 PyTorch Golden） | FP32 Softmax | FP16 Softmax |
| --- | ---: | ---: |
| Mean Cosine Similarity | 0.999844 | 0.999844 |
| Tensor Relative Error | 0.012122 | 0.012125 |
| Tensor Max Absolute Error | 0.019475 | 0.012210 |

### 2. C 与 Python Model 一致性（seq_len=1024，br=128，bc=256）

来源：[analysis/single_head_acc_evaluation.ipynb](analysis/single_head_acc_evaluation.ipynb)

| 指标（C vs Python） | FP16 Softmax | FP32 Softmax |
| --- | ---: | ---: |
| Mean Cosine Similarity | 1.000000 | 1.000000 |
| Tensor Relative Error | 0.000031 | 0.000001 |
| Tensor Max Absolute Error | 0.000099 | 0.000001 |

### 3. 周期对比（single head，seq_len=1024，BRxBC=128x256）

来源：[analysis/single_head_cycles_cnt.ipynb](analysis/single_head_cycles_cnt.ipynb)

| 阶段 | FP32 cycles | FP16 cycles | Delta (FP16-FP32) | 相对 FP32 变化 |
| --- | ---: | ---: | ---: | ---: |
| QK^T | 281,932 | 303,016 | +21,084 | +7.478399% |
| PV | 270,789 | 272,396 | +1,607 | +0.593451% |
| softmax(S)（含 mask 与 dequant） | 5,576,350 | 4,198,285 | -1,378,065 | -24.712670% |
| output accum and dequant | 259,796 | 255,323 | -4,473 | -1.721736% |
| normalization | 73,674 | 74,464 | +790 | +1.072291% |
| TOTAL | 6,778,361 | 5,420,339 | -1,358,022 | -20.034666% |

结论：

- 在该测试设置下，FP16 路径的主要加速来源是 softmax 阶段。
- 单头总周期降低约 20.03%。

### 4. 仓库中已包含的实验结果

- 输出文本：
  [analysis/data/single_head_outputs/fp16](analysis/data/single_head_outputs/fp16) 与 [analysis/data/single_head_outputs/fp32](analysis/data/single_head_outputs/fp32)
- 周期统计表：
  [analysis/data/single_head_cycles_cnt_fp16.xlsx](analysis/data/single_head_cycles_cnt_fp16.xlsx)、[analysis/data/single_head_cycles_cnt_fp32.xlsx](analysis/data/single_head_cycles_cnt_fp32.xlsx)
- 分析 notebook：
  [analysis/single_head_acc_evaluation.ipynb](analysis/single_head_acc_evaluation.ipynb)、[analysis/single_head_cycles_cnt.ipynb](analysis/single_head_cycles_cnt.ipynb)

## 范围与限制

1. 本仓库重点覆盖 Flash Attention 前向路径与实验分析流程。
2. Chipyard 构建与运行命令依赖本地工程组织方式，本 README 采用流程化描述而非固定绝对命令。
3. FP16 softmax 路径在本项目流程中依赖 GCC 15.2。
