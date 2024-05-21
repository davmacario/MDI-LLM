import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    """
    Python implementation of torch.nn.functional.scaled_dot_product_attention from
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.squeeze()
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    DTYPE = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    DTYPE_TORCH = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[DTYPE]

    size = (1, 1, 10, 10)
    L, S, _, _ = size
    query = torch.rand(size, dtype=DTYPE_TORCH, device=DEVICE)
    key = torch.rand(size, dtype=DTYPE_TORCH, device=DEVICE)
    value = torch.rand(size, dtype=DTYPE_TORCH, device=DEVICE)
    torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)

    out_original = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    out_custom = scaled_dot_product_attention(query, key, value, is_causal=True)

    # print(out_original)
    # print(out_custom)

    print("Original:")
    print(f"- Device: {out_original.device}")

    print("Custom:")
    print(f"- Device: {out_custom.device}")

    print(torch.sum(out_original - out_custom))
