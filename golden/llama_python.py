"""
截取Tiny LLaMA模型第一层的真实Query、Key、Value，将其量化为INT8格式，导入至Python C Model中进行快速Flash Attention Chipyard仿真

同时实际运行前向传播，获得模型标准困惑度，与Python C Model中使用Flash Attention前向传播获得的困惑度进行对比分析
"""
import torch
from pathlib import Path
from flash_attention import flash_attention_forward
from typing import Callable, Tuple, Optional
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)

BR = 128
BC = 256
SEED = 42
MODE = "GOLDEN"  # 可选 "GOLDEN", "PYTHON"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_int8_attention_data(seq_len: int,
                             head_dim: int,
                             query: Optional[torch.Tensor] = None,
                             key: Optional[torch.Tensor] = None,
                             value: Optional[torch.Tensor] = None,
                             seed: int = SEED,
                             is_random: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    为Attention Forward初始化随机/给定INT8 Q, K, V张量及其缩放因子
    
    :param query: 可选的查询张量，如果为None则随机生成
    :type query: Optional[torch.Tensor]
    :param key: 可选的键张量，如果为None则随机生成
    :type key: Optional[torch.Tensor]
    :param value: 可选的值张量，如果为None则随机生成
    :type value: Optional[torch.Tensor]
    :param seed: 种子
    :type seed: int
    :param seq_len: 序列长度
    :type seq_len: int
    :param head_dim: 头部维度
    :type head_dim: int
    :param is_random: 是否随机生成数据，如果为False则使用给定的query, key, value张量
    :type is_random: bool
    :return: 初始化的Q, K, V张量及其缩放因子，若为随机生成，则不包含Batch与Head维度，否则与输入张量维度一致
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    if is_random:
        if seed is not None:
            torch.manual_seed(seed)
        Q_float32 = torch.randn(size=(seq_len, head_dim), dtype=torch.float32)
        K_float32 = torch.randn(size=(seq_len, head_dim), dtype=torch.float32)
        V_float32 = torch.randn(size=(seq_len, head_dim), dtype=torch.float32)
    else:
        assert query is not None and key is not None and value is not None, "When is_random is False, query, key, value must be provided."
        Q_float32 = query.float()
        K_float32 = key.float()
        V_float32 = value.float()

    S_q = torch.max(torch.abs(Q_float32), dim=-1).values / 127.0
    S_k = torch.max(torch.abs(K_float32), dim=-1).values / 127.0
    S_v = torch.max(torch.max(torch.abs(V_float32), dim=-1).values, dim=-1).values / 127.0

    Q_int8 = torch.clamp((Q_float32 / S_q.unsqueeze(-1)), -128, 127).to(torch.int8)
    K_int8 = torch.clamp((K_float32 / S_k.unsqueeze(-1)), -128, 127).to(torch.int8)
    V_int8 = torch.clamp((V_float32 / S_v.unsqueeze(-1).unsqueeze(-1)), -128, 127).to(torch.int8)

    return Q_int8, K_int8, V_int8, S_q, S_k, S_v
    
def custom_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        用于硬件验证的自定义Forward，只修改第一层
        """
        if self.layer_idx != 0:
             return LlamaAttention.original_forward(    # type: ignore[attr-defined]
                  self,
                  hidden_states,
                  position_embeddings,
                  attention_mask,
                  past_key_values,
                  cache_position,
                  **kwargs
             )
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        global MODE

        if MODE == "PYTHON":
            Q_int8, K_int8, V_int8, S_q, S_k, S_v = init_int8_attention_data(query=query_states,
                                                                             key=repeat_kv(key_states, self.num_key_value_groups),
                                                                             value=repeat_kv(value_states, self.num_key_value_groups),
                                                                             seq_len=query_states.shape[2],
                                                                             head_dim=query_states.shape[3],
                                                                             is_random=False)
            attn_output = flash_attention_forward(query=Q_int8,
                                                  key=K_int8,
                                                  value=V_int8,
                                                  s_q=S_q,
                                                  s_k=S_k,
                                                  s_v=S_v,
                                                  br=BR,
                                                  bc=BC,
                                                  is_causal=True)
            
            # attn_output_duplicate, attn_weights = attention_interface(
            #     self,
            #     query_states,
            #     key_states,
            #     value_states,
            #     attention_mask,
            #     dropout=0.0 if not self.training else self.attention_dropout,
            #     scaling=self.scaling,
            #     **kwargs,
            # )

            attn_weights = torch.empty_like(attn_output)  # 占位
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            
        elif MODE == "GOLDEN":
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

def evaluate_dataset_ppl(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    text_column: str = "text",
    max_length: int = 512,
    stride: int = 256,
    limit_samples: int | None = None,
    min_text_len: int = 1,
    device: str | None = None,
    verbose: bool = True,
):
    """
    在同一数据集上分别评估：
    - MODE = "GOLDEN"（原始注意力前向）
    - MODE = "PYTHON"（第一层替换为 Python C-Model Flash Attention）
    并比较两者 PPL。

    Returns:
        dict:
        {
            "dataset": "...",
            "split": "...",
            "num_docs": int,
            "num_tokens": int,
            "golden": {"ppl": float, "avg_nll": float},
            "python": {"ppl": float, "avg_nll": float},
            "delta": {"ppl_abs": float, "ppl_rel_percent": float}
        }
    """
    import time
    from datasets import load_dataset

    global MODE

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if verbose:
        print(f"Evaluation on {device} with {model.__class__.__name__} begins...")

    if max_length <= 1:
        raise ValueError(f"max_length必须大于1，当前是{max_length}")
    if stride <= 0:
        raise ValueError(f"stride必须大于0，当前是{stride}")
    if stride > max_length:
        if verbose:
            print(f"[Info] stride({stride})大于max_length({max_length})，自动截断为max_length")
        stride = max_length

    if verbose:
        print(f"\n[Dataset] Loading {dataset_name}/{dataset_config} ({split})...")
    ds = load_dataset(dataset_name, dataset_config, split=split)

    if limit_samples is not None:
        limit_samples = min(limit_samples, len(ds))
        ds = ds.select(range(limit_samples))
        if verbose:
            print(f"[Dataset] Sample number: {limit_samples}")

    texts = []
    for x in ds[text_column]:
        if isinstance(x, str) and len(x.strip()) >= min_text_len:
            texts.append(x.strip())

    if len(texts) == 0:
        raise ValueError("数据集文本为空，请检查text_column或筛选条件")

    corpus = "\n\n".join(texts)

    if verbose:
        print("[Tokenize] Encoding corpus...")
    enc = tokenizer(corpus, return_tensors="pt")
    input_ids_all = enc["input_ids"].to(device)
    total_len = input_ids_all.size(1)

    if total_len < 2:
        raise ValueError(f"token数过少: {total_len}")

    if verbose:
        print(f"[Tokenize] num_docs={len(texts)}, total_tokens={total_len}, max_length={max_length}, stride={stride}")

    def _eval_one_mode(eval_mode: str):
        global MODE
        MODE = eval_mode

        nll_sum = torch.zeros((), dtype=torch.float64, device=device)
        n_tokens = 0
        n_steps = 0

        t0 = time.time()

        with torch.inference_mode():
            for i in range(0, total_len - 1, stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, total_len)
                trg_len = end_loc - i

                input_ids = input_ids_all[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                outputs = model(input_ids=input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss.double() * trg_len

                nll_sum += neg_log_likelihood
                n_tokens += trg_len
                n_steps += 1

                if end_loc == total_len:
                    break

        avg_nll = (nll_sum / max(n_tokens, 1)).item()
        ppl = float(torch.exp(torch.tensor(avg_nll)).item())
        elapsed = time.time() - t0

        if verbose:
            print(f"[{eval_mode}] steps={n_steps}, valid_tokens={n_tokens}, avg_nll={avg_nll:.6f}, ppl={ppl:.6f}, time={elapsed:.1f}s")

        return {
            "mode": eval_mode,
            "ppl": ppl,
            "avg_nll": avg_nll,
            "steps": n_steps,
            "valid_tokens": n_tokens,
            "elapsed_sec": elapsed,
        }

    if verbose:
        print("\n[Run] MODE=GOLDEN")
    golden = _eval_one_mode("GOLDEN")

    if verbose:
        print("[Run] MODE=PYTHON")
    python_res = _eval_one_mode("PYTHON")

    ppl_abs = abs(python_res["ppl"] - golden["ppl"])
    ppl_rel_percent = (ppl_abs / max(golden["ppl"], 1e-12)) * 100.0

    MODE = "GOLDEN"

    result = {
        "dataset": f"{dataset_name}/{dataset_config}",
        "split": split,
        "num_docs": len(texts),
        "num_tokens": int(total_len),
        "golden": {
            "ppl": golden["ppl"],
            "avg_nll": golden["avg_nll"],
        },
        "python": {
            "ppl": python_res["ppl"],
            "avg_nll": python_res["avg_nll"],
        },
        "delta": {
            "ppl_abs": ppl_abs,
            "ppl_rel_percent": ppl_rel_percent,
        },
    }

    if verbose:
        print("\n[Summary]")
        print(f"Dataset: {result['dataset']} ({split})")
        print(f"Docs: {result['num_docs']}\nTokens: {result['num_tokens']}")
        print(f"GOLDEN PPL: {result['golden']['ppl']:.6f}")
        print(f"PYTHON PPL: {result['python']['ppl']:.6f}")
        print(f"ABS Diff: {result['delta']['ppl_abs']:.6f}")
        print(f"REL Diff: {result['delta']['ppl_rel_percent']:.6f} %")

    return result

def evaluate_words_ppl(text: str,
                       model,
                       tokenizer,
                       device: str = DEVICE):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    labels = input_ids.clone()
    print(f"Int Flash Attention LlaMA Evaluation Starts...")
    print(f"Input Text: \"{text}\", Sequence Length: {input_ids.shape[1]}")

    global MODE
    MODE = "GOLDEN"
    print(f"\n>>> Running {MODE} Mode")
    
    with torch.inference_mode():
        outputs_golden = model(input_ids=input_ids, labels=labels)
        loss_golden = outputs_golden.loss
        ppl_golden = torch.exp(loss_golden)

    print(f"原始前向传播结束，Golden PPL: {ppl_golden.item():.4f}, Golden Loss: {loss_golden.item():.4f}")

    MODE = "PYTHON"
    print(f"\n>>> Running {MODE} Mode")
    with torch.inference_mode():
        outputs_c = model(input_ids=input_ids, labels=labels)
        loss_c = outputs_c.loss
        ppl_c = torch.exp(loss_c)

    print(f"Flash Attention前向传播结束，Hardware (C Model Simulated) PPL: {ppl_c.item():.4f}, Hardware (C Model Simulated) Loss: {loss_c.item():.4f}")
    print(f"PPL Difference: {abs(ppl_c.item() - ppl_golden.item()):.4f}")
    print("\n>>> Evaluation finished. Please check the PPL difference and analyze any discrepancies.")

if __name__ == "__main__":
    if not hasattr(LlamaAttention, "original_forward"):
        setattr(LlamaAttention, "original_forward", LlamaAttention.forward)

    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(DEVICE)   # type: ignore[call-arg]
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    LlamaAttention.forward = custom_forward

    # evaluate_words_ppl(
    #     text="多年以后，面对行刑队，奥雷里亚诺·布恩迪亚上校将会回想起父亲带他去见识冰块的那个遥远的下午。那时的马孔多是一个二十户人家的村落，泥巴和芦苇盖成的屋子沿河岸排开，湍急的河水清澈见底，河床里卵石洁白光滑宛如史前巨蛋。世界新生伊始，许多事物还没有名字，提到的时候尚需用手指指点点。",
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=DEVICE
    # )

    results = evaluate_dataset_ppl(
        model=model,
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test",
        text_column="text",
        max_length=512,
        stride=256,
        limit_samples=None,
        device=DEVICE,
        verbose=True
    )
