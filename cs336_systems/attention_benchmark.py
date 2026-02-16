import time
import torch
import torch.nn.functional as F


def benchmark_attention(
    batch_size: int = 8,
    num_heads: int = 8,
    seq_len: int = 256,
    head_dim: int = 64,
    device: str = "cuda",
    n_warmup: int = 10,
    n_steps: int = 50,
    use_amp: bool = False,
):
    # Create random Q, K, V
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Warmup
    for _ in range(n_warmup):
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = F.scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()

    # Timed loop
    start = time.time()

    for _ in range(n_steps):
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = F.scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / n_steps
    print(f"Seq Len {seq_len} → {avg_time:.6f} seconds")

    return avg_time


if __name__ == "__main__":
    for seq in [128, 256, 512, 1024]:
        benchmark_attention(seq_len=seq)