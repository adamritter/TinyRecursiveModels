"""
Substitution-learning toy task
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
USE_SINKHORN = os.getenv("USE_SINKHORN", "0") == "1"
USE_GUMBEL = os.getenv("USE_GUMBEL", "0") == "1"
USE_GUMBEL_TOPK = os.getenv("USE_GUMBEL_TOPK", "0") == "1"


def sinkhorn(logits: torch.Tensor, n_iters: int = SINK_ITERS, tau: float = TAU_START,
             eps: float = 1e-9) -> torch.Tensor:
    """Differentiable doubly‑stochastic projection."""
    P = torch.exp(logits / tau)
    for _ in range(n_iters):
        P = P / (P.sum(-1, keepdim=True) + eps)   # row norm
        P = P / (P.sum(-2, keepdim=True) + eps)   # col norm
    return P

def sinkhorn_sparse(
        logits: torch.Tensor,
        n_iters: int = 20,
        tau: float = 0.1,
        k: int = 1,                     # keep k non-zeros per row
        eps: float = 1e-9
) -> torch.sparse.Tensor:
    """
    Sparse Sinkhorn:  doubly-stochastic projection ➊,  row-wise top-k ➋,
    returned as a COO tensor whose *values* remain differentiable.

    logits : (N,N) or (B,N,N) —   real-valued
    """
    # ➊ Dense Sinkhorn (same as before, works batch-wise too)
    P = torch.exp(logits / tau)
    for _ in range(n_iters):
        P = P / (P.sum(-1, keepdim=True) + eps)      # rows
        P = P / (P.sum(-2, keepdim=True) + eps)      # cols

    # ➋ Keep top-k entries per row
    vals, cols = P.topk(k, dim=-1)                   # (…,N,k)
    rows = torch.arange(P.size(-2), device=P.device)\
                 .unsqueeze(-1).expand_as(cols)      # (…,N,k)

    # flatten batch dims (if any) into the 1st COO index
    leading = P.shape[:-2]                           # e.g. (B,)
    if leading:
        # batch offset so each (batch,row) pair has unique row-id
        flat_rows = rows + (torch.arange(
            P.numel() // (P.size(-1)*P.size(-2)),
            device=P.device)
            .view(*leading, 1, 1) * P.size(-2))
        shape = (flat_rows.max().item() + 1, P.size(-1))
        indices = torch.stack([flat_rows.reshape(-1), cols.reshape(-1)])
    else:
        shape = P.shape[-2:]
        indices = torch.stack([rows.reshape(-1), cols.reshape(-1)])

    values = vals.reshape(-1)                        # keep grads!

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

def soft_sub(log_S, tau=0.1):
    gumbel = -torch.empty_like(log_S).exponential_().log()
    return torch.softmax((log_S + gumbel) / tau, dim=-1)   # (K,K)

def gumbel_topk(logits, k=1, tau=0.1):
    g = -torch.empty_like(logits).exponential_().log()   # Gumbel noise
    y = (logits + g) / tau                               # noisy scores

    vals, idx = y.topk(k, dim=-1)                        # keep best k
    hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)

    # straight-through: forward = hard, backward = soft
    soft = torch.softmax(y, dim=-1)
    return hard.detach() - soft.detach() + soft          # (…, k non-zeros)

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
elif USE_GUMBEL:
    permf = soft_sub
elif USE_GUMBEL_TOPK:
    permf = gumbel_topk
else:
    permf = lambda x: x  # identity

# 123*123=
# 123
#  246
#.  369


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

    if loss < 1e-3 and acc == 1.0:
        print("✓ learned exact permutation in", time.time() - t_start, "seconds and ", step+1, "steps, loss=", loss.item())
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
