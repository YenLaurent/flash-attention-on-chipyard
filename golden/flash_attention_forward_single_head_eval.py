"""
用于验证C代码中flash_attention_forward_single_head函数的正确性
包括：
    - flash_attention_forward_single_head 的 Python Golden 实现（并非完全一比一复现，可能会有一些误差）
    - 生成随机Query、Key、Value张量的脚本文件，相关矩阵输出为./source/single_head_data.h头文件，可直接用于C端测试
"""
from flash_attention_forward_inner_eval import flash_attention_forward_inner, to_c_array
import torch

BR = 64
BC = 64
HEAD_DIM = 64
SEQ_LEN = 120
SEED = 42

def flash_attention_forward_single_head(Q: torch.Tensor,
                                        K: torch.Tensor,
                                        V: torch.Tensor,
                                        S_q: torch.Tensor,
                                        S_k: torch.Tensor,
                                        S_v: torch.Tensor,
                                        seq_len: int = SEQ_LEN,
                                        is_causal: bool = True) -> torch.Tensor:
    """
    参考C端 flash_attention_forward_single_head 的 Python 版本（用于验证）
    
    :param Q: (seq_len, HEAD_DIM) INT8查询张量
    :type Q: torch.Tensor
    :param K: (seq_len, HEAD_DIM) INT8键张量
    :type K: torch.Tensor
    :param V: (seq_len, HEAD_DIM) INT8值张量
    :type V: torch.Tensor
    :param S_q: (seq_len, ) Float Token Level缩放因子
    :type S_q: torch.Tensor
    :param S_k: (seq_len, ) Float Token Level缩放因子
    :type S_k: torch.Tensor
    :param S_v: Scalar, Float Tensor Level缩放因子
    :type S_v: torch.Tensor
    :param seq_len: 序列长度
    :type seq_len: int
    :param is_causal: 是否应用因果掩码
    :type is_causal: bool
    :return: (seq_len, HEAD_DIM) 输出张量
    :rtype: Tensor
    """
    TR = (seq_len + BR - 1) // BR
    TC = (seq_len + BC - 1) // BC
    O = torch.zeros(size=(seq_len, HEAD_DIM), dtype=torch.float32)

    for i in range(TR):
        r_start = i * BR
        r_end = min(seq_len, r_start + BR)
        r_len = r_end - r_start

        TC = ((r_end + BC - 1) // BC) if is_causal else TC

        m_prev = torch.full(size=(BR, ), fill_value=float('-inf'), dtype=torch.float32)
        m_curr = torch.full(size=(BR, ), fill_value=float('-inf'), dtype=torch.float32)
        l_prev = torch.zeros(size=(BR, ), dtype=torch.float32)
        l_curr = torch.zeros(size=(BR, ), dtype=torch.float32)
        o_prev = torch.zeros(size=(BR, HEAD_DIM), dtype=torch.float32)
        o_curr = torch.zeros(size=(BR, HEAD_DIM), dtype=torch.float32)

        if r_len == BR:
            q_block = Q[r_start:r_end, :]
            s_q_block = S_q[r_start:r_end]
        else:
            q_block = torch.cat([Q[r_start:r_end, :], torch.zeros((BR - r_len, HEAD_DIM), dtype=Q.dtype)], dim=0)
            s_q_block = torch.cat([S_q[r_start:r_end], torch.full((BR-r_len, ), 1.0, dtype=S_q.dtype)], dim=0)

        for j in range(TC):
            c_start = j * BC
            c_end = min(seq_len, c_start + BC)
            c_len = c_end - c_start

            if c_len == BC:
                k_block = K[c_start:c_end, :]
                v_block = V[c_start:c_end, :]
                s_k_block = S_k[c_start:c_end]
            else:
                k_block = torch.cat([K[c_start:c_end, :], torch.zeros((BC - c_len, HEAD_DIM), dtype=K.dtype)], dim=0)
                v_block = torch.cat([V[c_start:c_end, :], torch.zeros((BC - c_len, HEAD_DIM), dtype=V.dtype)], dim=0)
                s_k_block = torch.cat([S_k[c_start:c_end], torch.full((BC - c_len, ), 1.0, dtype=S_k.dtype)], dim=0)

            m_curr, l_curr, o_curr = flash_attention_forward_inner(q_int8 = q_block,
                                                                   k_int8 = k_block,
                                                                   v_int8 = v_block,
                                                                   s_q = s_q_block,
                                                                   s_k = s_k_block,
                                                                   m_old = m_prev,
                                                                   l_old = l_prev,
                                                                   o_old = o_prev,
                                                                   Br = BR,
                                                                   Bc = BC,
                                                                   d = HEAD_DIM,
                                                                   r_glob = r_start,
                                                                   c_glob = c_start,
                                                                   is_causal = is_causal)

            m_prev = m_curr
            l_prev = l_curr
            o_prev = o_curr

        O[r_start:r_end, :] = o_curr[0:r_len, :] * S_v / torch.max(l_curr[0:r_len].unsqueeze(1), torch.tensor(1e-9))

    return O

if __name__ == "__main__":
    torch.manual_seed(SEED)

    print("The evaluation of flash_attention_forward_single_head starts...")
    print(f"Parameters: SEQ_LEN={SEQ_LEN}, BR={BR}, BC={BC}, HEAD_DIM={HEAD_DIM}, SEED={SEED}")

    Q = torch.randint(-128, 127, (SEQ_LEN, HEAD_DIM)).to(torch.int8)
    K = torch.randint(-128, 127, (SEQ_LEN, HEAD_DIM)).to(torch.int8)
    V = torch.randint(-128, 127, (SEQ_LEN, HEAD_DIM)).to(torch.int8)

    S_q = torch.rand(SEQ_LEN)
    S_k = torch.rand(SEQ_LEN)
    S_v = torch.rand(1)

    O = flash_attention_forward_single_head(Q = Q,
                                            K = K,
                                            V = V,
                                            S_q = S_q,
                                            S_k = S_k,
                                            S_v = S_v,
                                            seq_len = SEQ_LEN,
                                            is_causal = True)

    print(f"\nGolden Model Output of First 5 elements: \n{O.flatten()[:5]}")

    with open("./source/forward_single_head_data.h", "w") as f:
        f.write(f"#ifndef FORWARD_SINGLE_HEAD_DATA_H\n#define FORWARD_SINGLE_HEAD_DATA_H\n\n")
        f.write(f"// Generated by Python script. \n// SEQ_LEN = {SEQ_LEN}, BR = {BR}, BC = {BC}, HEAD_DIM = {HEAD_DIM}, SEED = {SEED}\n")
        f.write(f"// The arrays have been flattened already.\n")
        f.write(f"#include \"include/gemmini_params.h\"\n\n")

        f.write(f"#define SEQ_LEN {SEQ_LEN}\n")
        f.write(f"#define BR {BR}\n")
        f.write(f"#define BC {BC}\n")
        f.write(f"#define HEAD_DIM {HEAD_DIM}\n\n")

        f.write(to_c_array(tensor=Q, name="Q_int8_gen", type_str="elem_t") + "\n")
        f.write(to_c_array(tensor=K, name="K_int8_gen", type_str="elem_t") + "\n")
        f.write(to_c_array(tensor=V, name="V_int8_gen", type_str="elem_t") + "\n")
        f.write(to_c_array(tensor=S_q, name="S_q_gen", type_str="float") + "\n")
        f.write(to_c_array(tensor=S_k, name="S_k_gen", type_str="float") + "\n")
        f.write(f"static float S_v_gen = {S_v.item()};\n")

        f.write("\n#endif\n")

    print("\nSuccess! \"forward_single_head_data.h\" generated.")