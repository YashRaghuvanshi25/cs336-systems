import time
import torch
from cs336_basics.model import BasicsTransformerLM


def benchmark_model(
    batch_size: int = 8,
    seq_len: int = 128,
    vocab_size: int = 10000,
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    device: str = "cuda",
    n_warmup: int = 10,
    n_steps: int = 50,
    forward_only: bool = False,
    use_amp: bool = False,
):
    # ------------------------
    # Model
    # ------------------------
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)

    if forward_only:
        model.eval()
    else:
        model.train()

    # ------------------------
    # Dummy Input
    # ------------------------
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------------------
    # Warmup
    # ------------------------
    for _ in range(n_warmup):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = logits.mean()

        if not forward_only:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    torch.cuda.synchronize()

    # ------------------------
    # Timed Loop
    # ------------------------
    start = time.time()

    for _ in range(n_steps):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = logits.mean()

        if not forward_only:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / n_steps
    print(f"Average step time: {avg_time:.6f} seconds")

    return avg_time


if __name__ == "__main__":
    print("FP32 Forward + Backward:")
    benchmark_model()

    print("\nAMP Forward + Backward:")
    benchmark_model(use_amp=True)

    print("\nForward Only FP32:")
    benchmark_model(forward_only=True)

    print("\nForward Only AMP:")
    benchmark_model(forward_only=True, use_amp=True)