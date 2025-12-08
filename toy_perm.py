"""
Permutation‑learning toy task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Learns the unknown permutation `perm_true` that maps A -> C using a
differentiable Sinkhorn relaxation plus a straight‑through top‑k sparsifier.
"""

import os
import time 
import torch




BATCHES   = int(os.getenv("BATCHES", 10))
N         = int(os.getenv("N", 100))
MAX_STEPS = int(os.getenv("MAX_STEPS", 10_000))
LR        = float(os.getenv("LR", 5e-2))
SINK_ITERS = int(os.getenv("SINK_ITERS", 5))
TAU_START  = float(os.getenv("TAU_START", 1.5))
USE_SINKHORN = os.getenv("USE_SINKHORN", "1") == "1"


def sinkhorn(logits: torch.Tensor, n_iters: int = SINK_ITERS, tau: float = TAU_START,
             eps: float = 1e-9) -> torch.Tensor:
    """Differentiable doubly‑stochastic projection."""
    P = torch.exp(logits / tau)
    for _ in range(n_iters):
        P = P / (P.sum(-1, keepdim=True) + eps)   # row norm
        P = P / (P.sum(-2, keepdim=True) + eps)   # col norm
    return P

device = "mps" if torch.backends.mps.is_available() else "cuda"

torch.manual_seed(0)
A = torch.randn(BATCHES, N, N, device=device)


perm_true = torch.randperm(N, device=device)
B_true = torch.eye(N, device=device)[perm_true]                     # (N,N)
C_target = torch.bmm(A, B_true.expand(BATCHES, -1, -1))            # (B,N,N)

log_alpha = torch.zeros(N, N, device=device, requires_grad=True)
opt = torch.optim.Adam([log_alpha], lr=LR)

if USE_SINKHORN:
    permf = sinkhorn
else:
    permf = lambda x: x  # identity

t_start = time.time()
for step in range(MAX_STEPS):
    P_soft = permf(log_alpha)  # dense, differentiable
    C_soft = torch.bmm(A, P_soft.expand(BATCHES, -1, -1))
    loss = (C_soft - C_target).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    acc = (P_soft.argmax(-1) == perm_true).float().mean().item()

    if step % 5 == 0:
        print(f"{step:5d}  loss={loss:.3e}  perm‑acc={acc:.3f}")

    if acc == 1.0:
        print("✓ learned exact permutation in", time.time() - t_start, "seconds and ", step+1, "steps.")
        break

else:
    print("Did not reach exact permutation within max steps.")

P_final =  permf(log_alpha)
perm_final = P_final.argmax(-1)

if torch.equal(perm_final, perm_true):
    print("Final permutation:", perm_final.tolist())
else:
    print("⚠️  Argmax changed after extra normalisation.")
    print("learned:", perm_final.tolist())
    print("true   :", perm_true.tolist())
