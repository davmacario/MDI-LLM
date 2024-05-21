import torch
import torch.nn.functional as F
from sub import functional as my_F

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

if __name__ == "__main__":
    size = (1, 1, 10, 10)
    L, S, _, _ = size
    query = torch.rand(size, dtype=DTYPE_TORCH, device=DEVICE)
    key = torch.rand(size, dtype=DTYPE_TORCH, device=DEVICE)
    value = torch.rand(size, dtype=DTYPE_TORCH, device=DEVICE)
    torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)

    out_original = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    out_custom = my_F.scaled_dot_product_attention(query, key, value, is_causal=True)

    # print(out_original)
    # print(out_custom)

    print("Original:")
    print(f"- Device: {out_original.device}")

    print("Custom:")
    print(f"- Device: {out_custom.device}")

    print(torch.sum(out_original - out_custom))
