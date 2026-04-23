# FlashAttention on Chipyard

Chinese version: [README_zh-CN.md](README_zh-CN.md).

## Table of Contents

1. [Project Overview](#project-overview)
2. [What Is Implemented](#what-is-implemented)
3. [Technical Details](#technical-details)
4. [Repository Structure](#repository-structure)
5. [Environment and Dependencies](#environment-and-dependencies)
6. [How to Use](#how-to-use)
7. [Experiment Data](#experiment-data)
8. [Known Scope and Limitations](#known-scope-and-limitations)

## Project Overview

This repository implements an edge-oriented Flash Attention forward pipeline on a Chipyard-based RISC-V platform.

The implementation targets a heterogeneous architecture:

- Gemmini NPU for matrix multiplications.
- Saturn RVV for vectorized softmax-related kernels.
- Rocket core for control logic and bare-metal execution.

The project supports two softmax accumulation modes:

- FP32 softmax path.
- FP16 softmax path.

Both are integrated into the same Flash Attention forward flow and selectable at compile time.

## What Is Implemented

This project includes the following core work:

1. Edge-side Int Flash Attention forward computation with tiling and causal masking.
2. Quantized Q/K/V flow (int8 tensors with scaling factors), including dequantization and normalization stages.
3. Online Safe Softmax implementation on RVV.
4. Schraudolph-style exponential approximation in both FP32 and FP16 variants.
5. End-to-end single-head forward path with cycle-level performance breakdown.
6. Python golden/reference models for validation and reproducible accuracy analysis.

## Technical Details

### 1. Hardware configuration entry

Main Chipyard config classes are defined in [config/CustomConfigs.scala](config/CustomConfigs.scala):

- `GemminiRocketSaturnConfig`: Gemmini + Saturn RVV + Rocket baseline.
- `GemminiRocketSaturnConfigWithFP16`: same as above, with Rocket FP16 support enabled.

### 2. Core C implementation

Core kernels are in [source/flash_attention.c](source/flash_attention.c) and [source/flash_attention.h](source/flash_attention.h):

- `rvv_softmax_fp32`: FP32 online safe softmax path.
- `rvv_softmax_fp16`: FP16 online safe softmax path.
- `flash_attention_forward_inner`: inner tiled kernel orchestration.
- `flash_attention_forward_single_head`: single-head forward pipeline.
- `flash_attention_forward`: multi-head wrapper.

Key compile-time knobs in [source/flash_attention.h](source/flash_attention.h):

- `SEQ_LEN`, `HEAD_DIM`: sequence/head dimensions.
- `BR`, `BC`: tile sizes.
- `USE_FP16`: precision mode switch for softmax path.

### 3. Runtime pipeline (single head)

For each tile block, the implementation performs:

1. `QK^T` on Gemmini (int8 x int8 -> int32 accumulation).
2. Scale/dequant + causal mask + online safe softmax on RVV.
3. `P*V` on Gemmini.
4. Output accumulation/dequantization and final normalization.

Cycle counters in [source/flash_attention.c](source/flash_attention.c) track these stages separately:

- `t_gemm1`, `t_softmax`, `t_gemm2`, `t_accum`, `t_norm`, `t_total`.

### 4. Golden/reference and validation scripts

- [golden/flash_attention.py](golden/flash_attention.py): Python C-model, random input generation, output comparison.
- [golden/llama_python.py](golden/llama_python.py): model-level validation workflow.
- [golden/llama_c.py](golden/llama_c.py): C-side integration/export workflow.

## Repository Structure

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

## Environment and Dependencies

This project is designed to run in a Chipyard environment and then execute bare-metal binaries on Verilator simulation.

Required environment:

1. Chipyard workspace with Gemmini and Saturn RVV support available.
2. Custom hardware configuration classes from [config/CustomConfigs.scala](config/CustomConfigs.scala).
3. RISC-V bare-metal toolchain:
   - FP32 softmax mode: Chipyard default GCC toolchain is acceptable.
   - FP16 softmax mode: ***GCC 15.2 is required*** (for the FP16 code path).
4. Verilator flow enabled in your Chipyard setup.

## How to Use

This section intentionally describes an operational workflow instead of hard-coded local commands, because Chipyard project paths and scripts vary by environment.

### Step 1: Prepare Chipyard environment

Set up a working Chipyard environment with Gemmini and RVV-enabled configuration support.

### Step 2: Select hardware config

Use [config/CustomConfigs.scala](config/CustomConfigs.scala) in your Chipyard build flow:

- Choose `GemminiRocketSaturnConfig` for baseline flow.
- Choose `GemminiRocketSaturnConfigWithFP16` when running FP16 softmax path.

### Step 3: Select precision mode and compile-time parameters

In your bare-metal build settings, configure macro values used by [source/flash_attention.h](source/flash_attention.h), for example:

- `USE_FP16=false` for FP32 softmax path.
- `USE_FP16=true` for FP16 softmax path.
- `SEQ_LEN`, `HEAD_DIM`, `BR`, `BC` as your experiment settings.

### Step 4: Compile bare-metal program

Build the bare-metal program from [source/flash_attention.c](source/flash_attention.c).

Toolchain guidance:

- FP32 softmax: use Chipyard default GCC.
- FP16 softmax: update to GCC 15.2 before compilation.

### Step 5: Run on Verilator

Run the generated bare-metal binary in your Chipyard Verilator simulation flow and capture console logs.

The program prints:

- Total cycle count.
- Per-stage cycle breakdown (`QK^T`, softmax, `P*V`, accumulation/dequantization, normalization).

### Step 6: Validate against Python golden outputs

Use scripts and notebooks under [golden](golden) and [analysis](analysis) to compare C outputs with Python and PyTorch golden references.

## Experiment Data

### 1. Accuracy metrics (single head, seq_len=1024, head_dim=64)

Source: [analysis/single_head_acc_evaluation.ipynb](analysis/single_head_acc_evaluation.ipynb)

| Metric (vs PyTorch Golden) | FP32 Softmax | FP16 Softmax |
| --- | ---: | ---: |
| Mean Cosine Similarity | 0.999844 | 0.999844 |
| Tensor Relative Error | 0.012122 | 0.012125 |
| Tensor Max Absolute Error | 0.019475 | 0.012210 |

### 2. C vs Python consistency (seq_len=1024, br=128, bc=256)

Source: [analysis/single_head_acc_evaluation.ipynb](analysis/single_head_acc_evaluation.ipynb)

| Metric (C vs Python) | FP16 Softmax | FP32 Softmax |
| --- | ---: | ---: |
| Mean Cosine Similarity | 1.000000 | 1.000000 |
| Tensor Relative Error | 0.000031 | 0.000001 |
| Tensor Max Absolute Error | 0.000099 | 0.000001 |

### 3. Cycle comparison (single head, seq_len=1024, BRxBC=128x256)

Source: [analysis/single_head_cycles_cnt.ipynb](analysis/single_head_cycles_cnt.ipynb)

| Stage | FP32 cycles | FP16 cycles | Delta (FP16-FP32) | Change% vs FP32 |
| --- | ---: | ---: | ---: | ---: |
| QK^T | 281,932 | 303,016 | +21,084 | +7.478399% |
| PV | 270,789 | 272,396 | +1,607 | +0.593451% |
| softmax(S) including mask and dequant | 5,576,350 | 4,198,285 | -1,378,065 | -24.712670% |
| output accum and dequant | 259,796 | 255,323 | -4,473 | -1.721736% |
| normalization | 73,674 | 74,464 | +790 | +1.072291% |
| TOTAL | 6,778,361 | 5,420,339 | -1,358,022 | -20.034666% |

Interpretation:

- The major cycle reduction comes from the softmax stage in FP16 mode.
- End-to-end single-head runtime is reduced by about 20.03% in this tested setup.

### 4. Data artifacts included in this repository

- Output tensors: [analysis/data/single_head_outputs/fp16](analysis/data/single_head_outputs/fp16) and [analysis/data/single_head_outputs/fp32](analysis/data/single_head_outputs/fp32).
- Cycle spreadsheets: [analysis/data/single_head_cycles_cnt_fp16.xlsx](analysis/data/single_head_cycles_cnt_fp16.xlsx), [analysis/data/single_head_cycles_cnt_fp32.xlsx](analysis/data/single_head_cycles_cnt_fp32.xlsx).
- Notebooks: [analysis/single_head_acc_evaluation.ipynb](analysis/single_head_acc_evaluation.ipynb), [analysis/single_head_cycles_cnt.ipynb](analysis/single_head_cycles_cnt.ipynb).

## Known Scope and Limitations

1. This repository focuses on Flash Attention forward path and related analysis workflows.
2. Build commands are environment-specific to your Chipyard setup, so this README provides a process-oriented guide instead of fixed local command lines.
3. FP16 softmax path depends on GCC 15.2 in this project workflow.
