"""
Chipyard Int Flash Attention的Python C-Model

包括：
    - flash_attention_forward_inner_calc等函数的Python Golden实现
    - 将C端Flash Attention Forward的输出与Pytorch标准实现进行对比验证的脚本，得到相关误差指标并输出为纯文本文件
    - 生成随机Query、Key、Value张量的工具脚本，相关矩阵输出为./source/下的C头文件，可直接用于C端测试
"""

import torch
from typing import Tuple
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path

# Hyperparameters
SEQ_LEN = 250
HEAD_DIM = 64
BR = 128
BC = 256
SEED = 23
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Python C-Model Implementation of Flash Attention Forward on Chipyard
def expp(num_2_calc: torch.Tensor) -> torch.Tensor:
    """
    Returns the EXPP Method results that is a more accurate approximation of standard exponentiation.

    Args:
        num_2_calc: The number to be calculated.
    """
    # Parameter
    ln2_inversed = 1.442695040889634
    alpha = 0.21875
    beta = 0.4375
    gamma_1 = 3.296875
    gamma_2 = 2.171875
    P = []
    
    x = num_2_calc * ln2_inversed
    x_int = torch.floor(x)
    x_frac = x - x_int

    # The correction polynomial P(x)
    p_high = 1 - beta * (1 - x_frac) * (x_frac + gamma_2) # For x >= 0.5
    p_low = alpha * x_frac * (x_frac + gamma_1)           # For x < 0.5
    P = torch.where(condition = x_frac >= 0.5,
                    input = p_high,
                    other = p_low)

    # The final results
    expp = (2 ** x_int) * (1 + P)
    expp = torch.where(num_2_calc < -87.0, torch.zeros_like(expp), expp)
    return expp

def flash_attention_forward_inner(q_int8: torch.Tensor, 
                                  k_int8: torch.Tensor, 
                                  v_int8: torch.Tensor, 
                                  s_q: torch.Tensor,
                                  s_k: torch.Tensor,
                                  m_old: torch.Tensor,
                                  l_old: torch.Tensor,
                                  o_old: torch.Tensor,
                                  Br: int = BR,
                                  Bc: int = BC,
                                  d: int = HEAD_DIM,
                                  r_glob: int = 0,
                                  c_glob: int = 0,
                                  is_causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Returns: (max, l, output)
    """
    参照 C 端 flash_attention_forward_inner_calc 的 Python 版本（用于验证）

    与C保持完全一致

    Args:
        q_int8 (torch.Tensor): (Br, d) 查询张量
        k_int8 (torch.Tensor): (Bc, d) 键张量
        v_int8 (torch.Tensor): (Bc, d) 值张量
        s_q (torch.Tensor): (Br,) Query 的 Token Level 缩放因子
        s_k (torch.Tensor): (Bc,) Key 的 Token Level 缩放因子
        m_old (torch.Tensor): (Br,) 旧的行最大值
        l_old (torch.Tensor): (Br,) 旧的行和
        o_old (torch.Tensor): (Br, d) 旧的输出
        Br (int): 分块行数
        Bc (int): 分块列数
        d (int): 头维度
        r_glob (int): 全局行偏移
        c_glob (int): 全局列偏移
        is_causal (bool): 是否应用因果掩码

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - max (torch.Tensor): (Br,) 最新行最大值
            - l (torch.Tensor): (Br,) 最新行和
            - output (torch.Tensor): (Br, d) 最新输出
    """
    device = q_int8.device
    # Step 1: Q @ K^T
    # s_int32 = torch.matmul(q_int8.to(torch.int32), k_int8.t().to(torch.int32))
    s_int32 = torch.matmul(q_int8.to(torch.float32), k_int8.t().to(torch.float32)).round().to(torch.int32)
    s_float = s_int32.to(torch.float32) * (d ** -0.5)
    s_float = s_float * s_q.unsqueeze(-1)
    s_float = s_float * s_k.unsqueeze(0)

    # Apply Causal Mask if needed
    if is_causal:
        row_idx = torch.arange(Br, device=device) + r_glob
        col_idx = torch.arange(Bc, device=device) + c_glob
        invalid = col_idx.unsqueeze(0) > row_idx.unsqueeze(1)
        s_float = s_float.masked_fill(invalid, -torch.finfo(torch.float32).max)

    # Step 2: Online Safe Softmax
    max = torch.maximum(m_old, torch.max(s_float, dim=1).values)
    p_unnormed = expp(s_float - max.unsqueeze(1)).mul(127.0)
    l = torch.sum(p_unnormed, dim=1) + l_old * expp(m_old - max)
    p_unnormed_int8 = torch.clamp(torch.round(p_unnormed), -128, 127).to(torch.int8)

    # Step 3: O = P @ V
    # O_int32 = torch.matmul(p_unnormed_int8.to(torch.int32), v_int8.to(torch.int32))
    O_int32 = torch.matmul(p_unnormed_int8.to(torch.float32), v_int8.to(torch.float32)).round().to(torch.int32)
    output = O_int32.to(torch.float32) + expp(m_old - max).unsqueeze(-1) * o_old

    return max, l, output

def flash_attention_forward_single_head(Q: torch.Tensor,
                                        K: torch.Tensor,
                                        V: torch.Tensor,
                                        S_q: torch.Tensor,
                                        S_k: torch.Tensor,
                                        S_v: torch.Tensor,
                                        br: int = BR,
                                        bc: int = BC,
                                        seq_len: int = SEQ_LEN,
                                        head_dim: int = HEAD_DIM,
                                        is_causal: bool = True) -> torch.Tensor:
    """
    参考C端 flash_attention_forward_single_head 的 Python 版本（用于验证）

    与C保持一致
    
    :param Q: (seq_len, head_dim) INT8查询张量
    :type Q: torch.Tensor
    :param K: (seq_len, head_dim) INT8键张量
    :type K: torch.Tensor
    :param V: (seq_len, head_dim) INT8值张量
    :type V: torch.Tensor
    :param S_q: (seq_len, ) Float Token Level缩放因子
    :type S_q: torch.Tensor
    :param S_k: (seq_len, ) Float Token Level缩放因子
    :type S_k: torch.Tensor
    :param S_v: Scalar, Float Tensor Level缩放因子
    :type S_v: torch.Tensor
    :param seq_len: 序列长度
    :type seq_len: int
    :param head_dim: 头维度
    :type head_dim: int
    :param is_causal: 是否应用因果掩码
    :type is_causal: bool
    :return: (seq_len, head_dim) 输出张量
    :rtype: Tensor
    """
    device = Q.device
    if head_dim is None:
        head_dim = Q.shape[-1]
    if seq_len is None:
        seq_len = Q.shape[0]

    TR = (seq_len + br - 1) // br
    TC_total = (seq_len + bc - 1) // bc
    O = torch.zeros(size=(seq_len, head_dim), dtype=torch.float32, device=device)

    for i in range(TR):
        r_start = i * br
        r_end = min(seq_len, r_start + br)
        r_len = r_end - r_start

        TC = ((r_end + bc - 1) // bc) if is_causal else TC_total

        m_prev = torch.full(size=(br, ), fill_value=-torch.finfo(torch.float32).max, dtype=torch.float32, device=device)
        m_curr = torch.full(size=(br, ), fill_value=-torch.finfo(torch.float32).max, dtype=torch.float32, device=device)
        l_prev = torch.zeros(size=(br, ), dtype=torch.float32, device=device)
        l_curr = torch.zeros(size=(br, ), dtype=torch.float32, device=device)
        o_prev = torch.zeros(size=(br, head_dim), dtype=torch.float32, device=device)
        o_curr = torch.zeros(size=(br, head_dim), dtype=torch.float32, device=device)

        if r_len == br:
            q_block = Q[r_start:r_end, :]
            s_q_block = S_q[r_start:r_end]
        else:
            q_block = torch.cat([Q[r_start:r_end, :], torch.zeros((br - r_len, head_dim), dtype=Q.dtype, device=device)], dim=0)
            s_q_block = torch.cat([S_q[r_start:r_end], torch.full((br-r_len, ), 1.0, dtype=S_q.dtype, device=device)], dim=0)

        for j in range(TC):
            c_start = j * bc
            c_end = min(seq_len, c_start + bc)
            c_len = c_end - c_start

            if c_len == bc:
                k_block = K[c_start:c_end, :]
                v_block = V[c_start:c_end, :]
                s_k_block = S_k[c_start:c_end]
            else:
                k_block = torch.cat([K[c_start:c_end, :], torch.zeros((bc - c_len, head_dim), dtype=K.dtype, device=device)], dim=0)
                v_block = torch.cat([V[c_start:c_end, :], torch.zeros((bc - c_len, head_dim), dtype=V.dtype, device=device)], dim=0)
                s_k_block = torch.cat([S_k[c_start:c_end], torch.full((bc - c_len, ), 1.0, dtype=S_k.dtype, device=device)], dim=0)

            m_curr, l_curr, o_curr = flash_attention_forward_inner(q_int8 = q_block,
                                                                   k_int8 = k_block,
                                                                   v_int8 = v_block,
                                                                   s_q = s_q_block,
                                                                   s_k = s_k_block,
                                                                   m_old = m_prev,
                                                                   l_old = l_prev,
                                                                   o_old = o_prev,
                                                                   Br = br,
                                                                   Bc = bc,
                                                                   d = head_dim,
                                                                   r_glob = r_start,
                                                                   c_glob = c_start,
                                                                   is_causal = is_causal)

            m_prev = m_curr
            l_prev = l_curr
            o_prev = o_curr

        l_inversed = torch.where(l_curr > 0, 1.0 / l_curr, torch.zeros_like(l_curr))
        O[r_start:r_end, :] = o_curr[0:r_len, :] * l_inversed[0:r_len].unsqueeze(1) * S_v 

    return O

def flash_attention_forward(query: torch.Tensor,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            s_q: torch.Tensor,
                            s_k: torch.Tensor,
                            s_v: torch.Tensor,
                            br: int,
                            bc: int,
                            is_causal: bool = True) -> torch.Tensor:
    """
    多头版本的 Flash Attention Forward 实现，与 Chipyard C 端完全一致
    
    参数形状说明：
    - Q, K, V : (batch_size, num_heads, seq_len, head_dim) INT8
    - S_q, S_k: (batch_size, num_heads, seq_len) FLOAT32
    - S_v     : (batch_size, num_heads) FLOAT32
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    O = torch.zeros(size=(batch_size, num_heads, seq_len, head_dim), dtype=torch.float32, device=query.device)
    
    for b in range(batch_size):
        for h in range(num_heads):
            Q_single_head = query[b, h, :, :]   # shape: (seq_len, head_dim)
            K_single_head = key[b, h, :, :]
            V_single_head = value[b, h, :, :]
            
            s_q_single_head = s_q[b, h, :]  # shape: (seq_len, )
            s_k_single_head = s_k[b, h, :] 
            s_v_single_head = s_v[b, h]     # Scalar tensor
            
            O_single_head = flash_attention_forward_single_head(
                Q = Q_single_head,
                K = K_single_head,
                V = V_single_head,
                S_q = s_q_single_head,
                S_k = s_k_single_head,
                S_v = s_v_single_head,
                br = br,
                bc = bc,
                seq_len = seq_len,
                is_causal = is_causal
            )
            
            O[b, h, :, :] = O_single_head
            
    return O

# Helper functions
def to_c_array(tensor: torch.Tensor, name: str, type_str: str) -> str:
    """
    返回将张量转换为 C 数组表示的字符串（已展平为一维），形式为：static <type_str> <name>[] = { ... };
    
    :param tensor: 要转换的张量
    :type tensor: torch.Tensor
    :param name: 转换后该数组的名称
    :type name: str
    :param type_str: 张量的数据类型字符串（如 "float" 或 "int8_t"）
    :type type_str: str
    """
    flat = tensor.to("cpu").flatten().tolist()
    if "float" in type_str:
        data_str = ", ".join([f"{x}f" for x in flat])
    else:
        data_str = ", ".join([str(int(x)) for x in flat])

    return f"static {type_str} {name}[] = {{{data_str}}};"

def write_c_head(Q_int8: torch.Tensor,
                 K_int8: torch.Tensor,
                 V_int8: torch.Tensor,
                 S_q: torch.Tensor,
                 S_k: torch.Tensor,
                 S_v: torch.Tensor):
    with open("./bareMetalC/random_data.h", "w") as f:
        f.write(f"#ifndef RANDOM_DATA_H\n#define RANDOM_DATA_H\n\n")
        f.write(f"// Generated by Python script. \n// SEQ_LEN = {SEQ_LEN}, BR = {BR}, BC = {BC}, HEAD_DIM = {HEAD_DIM}, SEED = {SEED}\n")
        f.write(f"// The arrays have been flattened already.\n")
        f.write(f"#include \"include/gemmini_params.h\"\n\n")

        f.write(f"#define SEQ_LEN {SEQ_LEN}\n")
        f.write(f"#define BR {BR}\n")
        f.write(f"#define BC {BC}\n")
        f.write(f"#define HEAD_DIM {HEAD_DIM}\n\n")

        f.write(to_c_array(tensor=Q_int8, name="Q_int8_gen", type_str="elem_t") + "\n")
        f.write(to_c_array(tensor=K_int8, name="K_int8_gen", type_str="elem_t") + "\n")
        f.write(to_c_array(tensor=V_int8, name="V_int8_gen", type_str="elem_t") + "\n")
        f.write(to_c_array(tensor=S_q, name="S_q_gen", type_str="float") + "\n")
        f.write(to_c_array(tensor=S_k, name="S_k_gen", type_str="float") + "\n")
        f.write(f"static float S_v_gen = {S_v.item()}f;\n")

        f.write("\n#endif\n")

def init_int8_attention_data(seed = SEED,
                             seq_len: int = SEQ_LEN,
                             head_dim: int = HEAD_DIM,
                             device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    为Attention Forward初始化随机INT8 Q, K, V张量及其缩放因子
    
    :param seed: 种子
    :type seed: int
    :param seq_len: 序列长度
    :type seq_len: int
    :param head_dim: 头部维度
    :type head_dim: int
    :return: 初始化的Q, K, V张量及其缩放因子，形状分别为(Q, K, V): (seq_len, head_dim), (seq_len, head_dim), (seq_len, head_dim); 缩放因子(S_q, S_k, S_v): (seq_len,), (seq_len,), scalar
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    Q_float32 = torch.randn(size=(seq_len, head_dim), dtype=torch.float32).to(device)
    K_float32 = torch.randn(size=(seq_len, head_dim), dtype=torch.float32).to(device)
    V_float32 = torch.randn(size=(seq_len, head_dim), dtype=torch.float32).to(device)

    S_q = torch.max(torch.abs(Q_float32), dim=1).values / 127.0
    S_k = torch.max(torch.abs(K_float32), dim=1).values / 127.0
    S_v = torch.max(torch.abs(V_float32)) / 127.0

    Q_int8 = torch.clamp((Q_float32 / S_q.unsqueeze(1)), -128, 127).to(torch.int8)
    K_int8 = torch.clamp((K_float32 / S_k.unsqueeze(1)), -128, 127).to(torch.int8)
    V_int8 = torch.clamp((V_float32 / S_v), -128, 127).to(torch.int8)

    return Q_int8, K_int8, V_int8, S_q, S_k, S_v

# Standard Attention Forward
def attention_fp32_single_head_golden(Q_int8: torch.Tensor,
                                      K_int8: torch.Tensor,
                                      V_int8: torch.Tensor,
                                      S_q: torch.Tensor,
                                      S_k: torch.Tensor,
                                      S_v: torch.Tensor,
                                      seed = SEED) -> torch.Tensor:
    """
    采用标准Pytorch Scaled Dot Product Attention实现Attention模块，接受输入INT8类型的Query, Key, Value并将其反量化为FP32后进行计算
    
    用于与C代码INT8 Flash Attention相比较

    :param Q_int8: INT8 Query of shape (SEQ_LEN, HEAD_DIM)
    :type Q_int8: torch.Tensor
    :param K_int8: INT8 Key of shape (SEQ_LEN, HEAD_DIM)
    :type K_int8: torch.Tensor
    :param V_int8: INT8 Value of shape (SEQ_LEN, HEAD_DIM)
    :type V_int8: torch.Tensor
    :param S_q: FP32 Token level scaler for Query of shape (SEQ_LEN, )
    :type S_q: torch.Tensor
    :param S_k: FP32 Token level scaler for Key of shape (SEQ_LEN, )
    :type S_k: torch.Tensor
    :param S_v: FP32 Tensor level scaler for Value
    :type S_v: torch.Tensor
    :param seed: Seed
    :type seed: int
    :return: FP32 Output tensor of shape (SEQ_LEN, HEAD_DIM)
    :rtype: torch.Tensor
    """
    if seed is not None:
        torch.manual_seed(seed)

    Q_fp32 = Q_int8.to(torch.float32) * S_q.unsqueeze(1)
    K_fp32 = K_int8.to(torch.float32) * S_k.unsqueeze(1)
    V_fp32 = V_int8.to(torch.float32) * S_v

    O_fp32 = F.scaled_dot_product_attention(query = Q_fp32.unsqueeze(0).unsqueeze(0), # 增加 Batch 和 Head 维度
                                            key = K_fp32.unsqueeze(0).unsqueeze(0),
                                            value = V_fp32.unsqueeze(0).unsqueeze(0),
                                            is_causal = True).squeeze(0).squeeze(0) # 去掉 Batch 和 Head 维度
    
    return O_fp32

if __name__ == "__main__":
    """
    以下脚本用于评估flash_attention_forward_single_head函数的正确性，流程如下：

    1. 调用init_int8_attention_data生成随机的INT8 Q, K, V张量及其缩放因子，并将这些数据写入C头文件（./bareMetalC/random_data.h）以供C端测试使用
    2. 使用attention_fp32_single_head_golden函数计算基于标准Pytorch实现的Attention输出（作为Golden结果）
    3. 使用flash_attention_forward_single_head函数计算基于Python C-Model实现的Flash Attention输出
    4. 从C端模拟器的输出日志中读取Chipyard Flash Attention输出
    5. 将上述三种结果（Golden, Python实现, C端模拟器）输出到./analysis/data/single_head_outputs/目录下的纯文本文件中，供后续分析使用
    """    
    print("The evaluation of flash_attention_forward_single_head starts...")
    print("Current device:", DEVICE)

    # Calculation & Read the C output
    Q_int8, K_int8, V_int8, S_q, S_k, S_v = init_int8_attention_data(device='cpu') # 确保生成数据时设备一致，否则随机数会有细微差异
    Q_int8, K_int8, V_int8, S_q, S_k, S_v = Q_int8.to(DEVICE), K_int8.to(DEVICE), V_int8.to(DEVICE), S_q.to(DEVICE), S_k.to(DEVICE), S_v.to(DEVICE)

    # Write C header file                
    write_c_head(Q_int8, K_int8, V_int8, S_q, S_k, S_v)

    O_golden = attention_fp32_single_head_golden(Q_int8, K_int8, V_int8, S_q, S_k, S_v) # Golden implementation
    O_python = flash_attention_forward_single_head(Q_int8, K_int8, V_int8, S_q, S_k, S_v, br=BR, bc=BC, seq_len=SEQ_LEN, is_causal=True)  # C Model
    O_c = torch.tensor([])

    path2c_output = Path(r"\\wsl.localhost\Ubuntu-Chipyard\home\yenxu\chipyard\sims\verilator\output\chipyard.harness.TestHarness.GemminiRocketSaturnConfig\single_out.log")
    ANCHOR_STR = "The outputs of shape"

    if path2c_output.is_file():
        with path2c_output.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if ANCHOR_STR in line:
                    if idx + 1 < len(lines):
                        data_line = lines[idx + 1]
                        values = [float(x) for x in data_line.strip().split()]
                        expected_len = SEQ_LEN * HEAD_DIM
                        if len(values) != expected_len:
                            print(f"[Warning] Log数据长度 ({len(values)}) 与预期 ({expected_len}) 不符，请检查Log文件完整性。")
                        
                        O_c = torch.tensor(values, dtype=torch.float32).reshape(SEQ_LEN, HEAD_DIM)
    else:
        raise FileNotFoundError(f"{path2c_output}文件不存在。")

    # Write output comparison files
    Path("./analysis/data/single_head_outputs/").mkdir(parents=True, exist_ok=True)

    with open(f"./analysis/data/single_head_outputs/O_golden_seq{SEQ_LEN}_brbc{BR}x{BC}.txt", "w") as f:
        for row in O_golden:
            f.write(" ".join([f"{x:.6f}" for x in row]) + "\n\n")

    with open(f"./analysis/data/single_head_outputs/O_python_seq{SEQ_LEN}_brbc{BR}x{BC}.txt", "w") as f:
        for row in O_python:
            f.write(" ".join([f"{x:.6f}" for x in row]) + "\n\n")
    
    with open(f"./analysis/data/single_head_outputs/O_c_seq{SEQ_LEN}_brbc{BR}x{BC}.txt", "w") as f:
        for row in O_c:
            f.write(" ".join([f"{x:.6f}" for x in row]) + "\n\n")

    print(f"Successfully generated \"./bareMetalC/random_data.h\"(parameters: SEQ_LEN={SEQ_LEN}, BR={BR}, BC={BC}, HEAD_DIM={HEAD_DIM}, SEED={SEED}), and runs the PyTorch golden model, outputs are in ./analysis/data/single_head_outputs/ file.")