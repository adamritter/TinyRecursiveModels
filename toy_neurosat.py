# toy_neurosat.py
# These don't help: 2/3 layer MLP, skip connections, layer norm didn't really matter
# Harder problem didn't help
# python toy_neurosat.py --iterations 25000 --num-vars 7 --num-clauses 40 --lr 1e-3 --batch-size 64 --epochs 10 --dim 512 --muon --num-layers 16 --test-every-s 10 --test-layer-multiplier 12 --train-layer-multiplier 6
# python toy_neurosat.py --iterations 25000 --num-vars 20 --num-clauses 150 --lr 1e-3 --batch-size 64 --epochs 10 --dim 512 --muon --num-layers 16 --test-every-s 10 --test-layer-multiplier 12 --train-layer-multiplier 6 --save model20
# large model was trained for 7 hours on GH200 by: python toy_neurosat.py --iterations 250000 --num-vars 20 --num-clauses 150 --lr 1e-3 --batch-size 64 --epochs 80 --dim 512 --muon --num-layers 16 --test-every-s 10 --test-layer-multiplier 12 --train-layer-multiplier 6 --save large_model
# it overfits, test_exact_acc: 0 :)
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sat_utils import cadical_solve, random_3sat


class OneLayerNeuroSAT(nn.Module):
    def __init__(self, d=32, use_act=False):
        super().__init__()
        # initial embeddings for all clauses and literals (shared)
        self.clause_init = nn.Parameter(torch.randn(d))
        self.literal_init = nn.Parameter(torch.randn(d))
        # three learnable linear maps
        self.Wc2l = nn.Linear(d, d, bias=True)
        self.Wl2c = nn.Linear(d, d, bias=True)
        self.Wflip = nn.Linear(d, d, bias=True)
        # read-out: scalar score per literal
        self.readout = nn.Linear(d, 1, bias=True)
        self.use_act = use_act
        self.halt = nn.Linear(d, 1, bias=True) if use_act else None
        self.d = d
        # learnable scales for residual updates
        self.scale_lit = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.scale_clause = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # mandatory normalization for more stable training
        self.norm_c = nn.LayerNorm(d)
        self.norm_l = nn.LayerNorm(d)
        # GRU-style recurrence over message-passing steps
        self.gru_lit = nn.GRUCell(d, d)
        self.gru_clause = nn.GRUCell(d, d)

    def forward(self, Hc, Hl, Ci, Lj, flip_index):
        """
        Single message-passing step.

        Hc : (m, d) clause embeddings
        Hl : (2n, d) literal embeddings
        Ci -> Lj edges: two tensors (src, dst) of equal length Ecl
        flip_index : (2n,) tensor giving the index of ¬ℓ for each literal ℓ
        """
        # clause -> literal with skip connection
        msg_c2l = Hc + self.Wc2l(Hc)                     # (m, d)
        agg_c2l = torch.zeros_like(Hl)
        agg_c2l.index_add_(0, Lj, msg_c2l[Ci])           # sum messages onto literals

        # literal -> neg-literal ("flip") with skip connection
        flip_in = Hl[flip_index]
        agg_flip = flip_in + self.Wflip(flip_in)         # (2n, d)

        # update literals via GRU cell
        lit_in = self.scale_lit * (agg_c2l + agg_flip)
        Hl = self.gru_lit(lit_in, Hl)

        # literal -> clause with skip connection
        msg_l2c = Hl + self.Wl2c(Hl)                     # (2n, d)
        agg_l2c = torch.zeros_like(Hc)
        agg_l2c.index_add_(0, Ci, msg_l2c[Lj])           # aggregate onto clauses

        # update clauses via GRU cell
        clause_in = self.scale_clause * agg_l2c
        Hc = self.gru_clause(clause_in, Hc)

        # layer normalization
        Hc = self.norm_c(Hc)
        Hl = self.norm_l(Hl)

        return Hl, Hc


# ---------- utility: build tiny SAT problems ----------


def build_graph(clauses):
    """
    Build a bipartite clause–literal graph for a single CNF instance.

    Returns:
        n (int): number of distinct Boolean variables.
        m (int): number of clauses.
        Ci (LongTensor): shape (E,), clause indices for each literal occurrence.
        Lj (LongTensor): shape (E,), literal indices in [0, 2*n) matching `Ci`.
        flip (LongTensor): shape (2*n,), for each literal index ℓ gives index of ¬ℓ.
    """
    m = len(clauses)
    vars_ = {abs(int(l)) for c in clauses for l in c}
    n = len(vars_)
    # index literals: x1,¬x1,x2,¬x2,...
    lit_index = { (v, s): 2*(v-1)+(0 if s==1 else 1) for v in vars_ for s in (1,-1) }
    Ci, Lj = [], []
    for ci, clause in enumerate(clauses):
        for lit in clause:
            val = int(lit)
            v = abs(val)
            s = 1 if val > 0 else -1
            Ci.append(ci)
            Lj.append(lit_index[(v, s)])
    Ci = torch.tensor(Ci, dtype=torch.long)
    Lj = torch.tensor(Lj, dtype=torch.long)
    flip = [lit_index[(v, -1)] if i % 2 == 0 else lit_index[(v, 1)]
            for v in vars_ for i in range(2)]
    flip = torch.tensor(flip, dtype=torch.long)
    return n, m, Ci, Lj, flip


def random_3sat_graph(num_vars=10, num_clauses=40):
    """
    Sample a random satisfiable 3-SAT instance and return its graph.

    Returns:
        clauses (list[tuple[int]]): CNF clauses for the instance.
        assignment (tuple[int]): one satisfying assignment (0/1 per variable).
        n, m, Ci, Lj, flip: outputs of `build_graph(clauses)`.
    """
    clauses, assignment = random_3sat(num_vars=num_clauses and num_vars, num_clauses=num_clauses)
    n, m, Ci, Lj, flip = build_graph(clauses)
    return clauses, assignment, n, m, Ci, Lj, flip


def _solve_dataset_with_cadical(graphs):
    """Solve all CNF instances in ``graphs`` with cadical_solve and return wall time."""

    if not graphs:
        return 0.0

    def _solve_one(sample):
        Ci, Lj, flip, target, num_clauses_total, num_literals, per_problem = sample
        # Reconstruct clauses from the bipartite graph encoding.
        # Literal index -> (var, sign): var = idx // 2 + 1, sign = +1 if idx % 2 == 0 else -1.
        ci_list = Ci.tolist()
        lj_list = Lj.tolist()
        clauses = [[] for _ in range(num_clauses_total)]
        for ci_idx, lit_idx in zip(ci_list, lj_list):
            var = lit_idx // 2 + 1
            sign = 1 if (lit_idx % 2) == 0 else -1
            clauses[ci_idx].append(sign * var)
        # Use cadical_solve; we ignore the result, only timing matters here.
        _ = cadical_solve(clauses)

    max_workers = min(32, (os.cpu_count() or 1), len(graphs))
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_solve_one, graphs))
    elapsed = time.perf_counter() - start
    print(f"solve_time_s {elapsed:.3f}")
    return elapsed


def _evaluate_dataset_eval_only(
    model,
    device,
    graphs,
    batch_size,
    num_layers,
    test_layer_multiplier,
):
    """Evaluate ``model`` on all graphs and return wall time."""
    if not graphs:
        return 0.0

    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    eval_start = time.perf_counter()
    evaluate_on_loader(
        model=model,
        device=device,
        loader=loader,
        num_layers=num_layers,
        test_layer_multiplier=test_layer_multiplier,
        print_prefix="eval-only | ",
    )
    elapsed = time.perf_counter() - eval_start
    print(f"eval_time_s {elapsed:.3f}")
    return elapsed


# ---------- dataset generation ----------
def generate_dataset(
    num_vars=2,
    num_clauses=4,
    dataset_size=1024,
    cache_dir="dataset",
):
    """Pre-generate (and cache) a fixed dataset of satisfiable 3-SAT instances."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = (
        f"toy_neurosat_dataset_size{dataset_size}_"
        f"vars{num_vars}_clauses{num_clauses}_v3.pt"
    )
    path = os.path.join(cache_dir, filename)

    if os.path.exists(path):
        t0 = time.perf_counter()
        data = torch.load(path)
        load_time = time.perf_counter() - t0
        print(f"Loaded dataset from {path} in {load_time:.3f}s")
        return data

    def _make_one(_):
        clauses, assignment = random_3sat(num_vars, num_clauses)
        n, m, Ci, Lj, flip = build_graph(clauses)
        target = torch.tensor(
            [assignment[i // 2] if i % 2 == 0 else 1 - assignment[i // 2]
             for i in range(2 * n)],
            dtype=torch.float,
        )
        num_literals = 2 * n
        per_problem = num_literals
        return Ci, Lj, flip, target, m, num_literals, per_problem

    max_workers = min((os.cpu_count() or 4), dataset_size)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        dataset = list(ex.map(_make_one, range(dataset_size)))
    create_time = time.perf_counter() - t0
    print(
        f"Created dataset of {dataset_size} problems "
        f"(vars={num_vars}, clauses={num_clauses}) in {create_time:.3f}s, saved to {path}"
    )

    torch.save(dataset, path)
    return dataset


def build_batch_from_samples(samples):
    """Collate a batch of pre-built graphs into one batched graph + targets."""
    all_Ci = []
    all_Lj = []
    all_flip = []
    all_targets = []
    clause_offset = 0
    lit_offset = 0
    per_problem = None

    # samples is a list of tuples from SATDataset.__getitem__:
    # (Ci, Lj, flip, target, num_clauses, num_literals, per_problem)
    for Ci, Lj, flip, target, num_clauses, num_literals, per_problem_single in samples:

        Ci = Ci + clause_offset
        Lj = Lj + lit_offset
        flip = flip + lit_offset

        all_Ci.append(Ci)
        all_Lj.append(Lj)
        all_flip.append(flip)
        all_targets.append(target)

        if per_problem is None:
            per_problem = per_problem_single

        clause_offset += num_clauses
        lit_offset += num_literals

    Ci = torch.cat(all_Ci, dim=0)
    Lj = torch.cat(all_Lj, dim=0)
    flip = torch.cat(all_flip, dim=0)
    target = torch.cat(all_targets, dim=0)

    return Ci, Lj, flip, target, clause_offset, lit_offset, per_problem


def compute_literal_metrics(scores: torch.Tensor, target: torch.Tensor, per_problem: int):
    """
    Compute literal-wise and exact accuracy from logits and targets.

    Args:
        scores: (B * per_problem,) logits for literals.
        target: (B * per_problem,) target labels in {0,1}.
        per_problem: number of literals per SAT instance (2 * num_vars).

    Returns:
        literal_accuracy (float), exact_accuracy (float), exact_per_example (BoolTensor[B])
    """
    with torch.no_grad():
        batch_size_actual = target.size(0) // per_problem
        scores_view = scores.view(batch_size_actual, per_problem)
        target_view = target.view(batch_size_actual, per_problem)

        num_vars = per_problem // 2
        scores_pair = scores_view.view(batch_size_actual, num_vars, 2)
        target_pair = target_view.view(batch_size_actual, num_vars, 2)

        # per-variable decision: choose more confident between (pos, neg)
        pred_true = (scores_pair[..., 0] >= scores_pair[..., 1])
        target_true = target_pair[..., 0] > 0.5

        correct = (pred_true == target_true)
        exact_per_example = correct.all(dim=1)  # (B,)
        literal_accuracy = correct.float().mean().item()
        exact_accuracy = exact_per_example.float().mean().item()
    return literal_accuracy, exact_accuracy, exact_per_example


def evaluate_on_loader(
    model, device, loader, num_layers, test_layer_multiplier, print_prefix=None,
):
    """
    Run evaluation over a DataLoader and return averaged loss and accuracies.

    If print_prefix is provided, print a single summary line of the form
    `{print_prefix}test_loss ... | test_acc ... | test_exact_acc ...`.
    """
    model.eval()
    total_loss = 0.0
    total_literal_acc = 0.0
    total_exact_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            Ci, Lj, flip, target, num_clauses_total, num_literals_total, per_problem = (
                build_batch_from_samples(batch)
            )

            Ci = Ci.to(device)
            Lj = Lj.to(device)
            flip = flip.to(device)
            target = target.to(device)

            Hc0 = model.clause_init.unsqueeze(0).expand(num_clauses_total, -1)
            Hl0 = model.literal_init.unsqueeze(0).expand(num_literals_total, -1)

            # "Cheating" test-time unrolls: per-example early stopping when exact.
            B = target.size(0) // per_problem
            solved = torch.zeros(B, dtype=torch.bool, device=device)
            final_scores_view = torch.zeros(B, per_problem, device=device)

            Hl = Hl0
            Hc = Hc0
            scores_view = None

            attempts = max(1, int(test_layer_multiplier))
            for _attempt in range(attempts):
                # advance all examples by num_layers steps
                for _ in range(num_layers):
                    Hl, Hc = model(Hc, Hl, Ci, Lj, flip)
                scores = model.readout(Hl).squeeze(-1)
                scores_view = scores.view(B, per_problem)

                # compute per-example exactness for this attempt
                _, _, exact_per_example = compute_literal_metrics(
                    scores, target, per_problem
                )

                newly_solved = exact_per_example & ~solved
                if newly_solved.any():
                    final_scores_view[newly_solved] = scores_view[newly_solved]
                    solved = solved | newly_solved
                if solved.all():
                    break

            # any unsolved examples use the last attempt's scores
            if not solved.all():
                final_scores_view[~solved] = scores_view[~solved]

            final_scores = final_scores_view.view(-1)

            loss = F.binary_cross_entropy_with_logits(final_scores, target)
            literal_accuracy, exact_accuracy, _ = compute_literal_metrics(
                final_scores, target, per_problem
            )

            total_loss += loss.item()
            total_literal_acc += literal_accuracy
            total_exact_acc += exact_accuracy
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_lit = total_literal_acc / max(num_batches, 1)
    avg_exact = total_exact_acc / max(num_batches, 1)

    if print_prefix is not None:
        print(
            f"{print_prefix}"
            f"test_loss {avg_loss:.4f} | "
            f"test_acc {avg_lit:.3f} | "
            f"test_exact_acc {avg_exact:.3f}"
        )

    return avg_loss, avg_lit, avg_exact

def create_optimizers(model, lr, use_muon: bool):
    optimizers = []
    if use_muon:
        # Split parameters so 2D parameters can be optimized by Muon (or any
        # 2D-only optimizer) and others by Adam.
        params_2d = []
        params_other = []
        for p in model.parameters():
            if p.ndim == 2:
                params_2d.append(p)
            else:
                params_other.append(p)

        if params_2d:
            # Replace `torch.optim.Muon` with your Muon optimizer class if needed.
            optimizers.append(torch.optim.Muon(params_2d, lr=lr))
        if params_other:
            optimizers.append(torch.optim.Adam(params_other, lr=lr))
    else:
        # Simple Adam on all parameters when not using Muon.
        optimizers.append(torch.optim.Adam(model.parameters(), lr=lr))

    return optimizers


# ---------- tiny training loop (cross-entropy over literal pairs) ----------
def train(
    epochs=10,
    d=32,
    lr=1e-3,
    num_vars=2,
    num_clauses=4,
    batch_size=32,
    dataset_size=1024,
    use_muon=False,
    num_layers=5,
    test_layer_multiplier=1,
    train_layer_multiplier=1,
    test_every_s=0.0,
    increase_multiplier_slowly=False,
    const_train_loss=0.0,
    load_path=None,
    eval_only=False,
    solve_only=False,
    use_act=False,
):

    model = OneLayerNeuroSAT(d, use_act=use_act)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if load_path:
        state = torch.load(load_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model state_dict from {load_path}")

    model.to(device)

    graphs = generate_dataset(num_vars=num_vars, num_clauses=num_clauses, dataset_size=dataset_size)
    if solve_only:
        _solve_dataset_with_cadical(graphs)
        return model

    if eval_only:
        _ = _evaluate_dataset_eval_only(
            model=model,
            device=device,
            graphs=graphs,
            batch_size=batch_size,
            num_layers=num_layers,
            test_layer_multiplier=test_layer_multiplier,
        )
        return model

    optimizers = create_optimizers(model, lr, use_muon)

    num_samples = len(graphs)
    test_size = min(500, num_samples // 2) if num_samples > 1 else 0
    split = num_samples - test_size
    train_graphs = graphs[:split]
    test_graphs = graphs[split:] if test_size > 0 else graphs

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    # Global timer for mid-epoch tests (does not reset each epoch)
    training_start = time.perf_counter()
    last_test_time = training_start

    HcInit = model.clause_init.unsqueeze(0).expand(num_clauses * batch_size, -1)
    HlInit = model.literal_init.unsqueeze(0).expand(num_vars * batch_size * 2, -1)
    Hc = HcInit.clone()
    Hl = HlInit.clone()
    step = torch.zeros(batch_size, device=device, dtype=torch.int32)
    Ci = None
    Lj = None
    flip = None
    target = None
    phalt_total = torch.zeros(batch_size, dtype=torch.float32, device=device)
    # set starting multiplier depending on flag:
    # - if increase_multiplier_slowly: start at 1 and grow
    # - otherwise: start directly at the requested train_layer_multiplier
    current_train_layer_multiplier = 1 if increase_multiplier_slowly else train_layer_multiplier

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()

        total_loss = 0.0
        total_literal_acc = 0.0
        total_exact_acc = 0.0
        num_batches = 0
        compute_time_train = 0.0

        for batch_idx, batch in enumerate(train_loader, start=1):
            CiNew, LjNew, flipNew, targetNew, _, _, per_problem = (
                build_batch_from_samples(batch)
            )

            CiNew = CiNew.to(device)
            LjNew = LjNew.to(device)
            flipNew = flipNew.to(device)
            targetNew = targetNew.to(device)

            if Ci is None:
                Ci = CiNew
                Lj = LjNew
                flip = flipNew
                target = targetNew
            else:
                reset_mask = (step == 0)  # Shape: [batch_size] (Boolean)

                if reset_mask.any():
                    # 2. Expand masks for FLATTENED inputs
                    # For Ci (Clauses): Expand mask to [Batch * num_clauses]
                    reset_mask_clauses = reset_mask.repeat_interleave(num_clauses*3)
                    
                    # For Flip (Variables): Expand mask to [Batch * num_vars]
                    reset_mask_vars = reset_mask.repeat_interleave(num_vars*2)

                    # 3. Apply Updates using the Expanded Masks
                    Ci[reset_mask_clauses] = CiNew[reset_mask_clauses]
                    Lj[reset_mask_clauses] = LjNew[reset_mask_clauses]
                    flip[reset_mask_vars] = flipNew[reset_mask_vars]
                    target[reset_mask_vars] = targetNew[reset_mask_vars]
                    phalt_total[reset_mask] = 0.0

            compute_start = time.perf_counter()
            # apply a stack of num_layers one-step updates
            for _ in range(num_layers):
                Hl, Hc = model(Hc, Hl, Ci, Lj, flip)
            scores = model.readout(Hl).squeeze(-1)
            literal_accuracy, exact_accuracy, exact_per_example = compute_literal_metrics(
                scores, target, per_problem
            )
            if use_act:
                phalt_current = torch.sigmoid(model.halt(Hl.reshape(batch_size, num_vars * 2, model.d).mean(dim=1).squeeze(-1)).squeeze(-1))
                phalt_current_detach = phalt_current.detach()
                overflow_mask = (phalt_total + phalt_current_detach) > 0.99
                phalt_total = (phalt_total + phalt_current).detach()
                losses = F.binary_cross_entropy_with_logits(scores, target, reduction='none').reshape(batch_size, num_vars * 2).mean(dim=1)
                phalt_current_loss = torch.where(overflow_mask, 1.0 - phalt_total, phalt_current)
                loss = ((losses * phalt_current_loss + (1.0 - phalt_current_loss) * const_train_loss ).mean()) * current_train_layer_multiplier
            else:
                losses = F.binary_cross_entropy_with_logits(scores, target, reduction='none').reshape(batch_size, num_vars * 2).mean(dim=1)
                #losses = torch.where(exact_per_example, losses * 10.0, losses)
                #loss = F.binary_cross_entropy_with_logits(scores, target)
                loss = losses.mean()
                if const_train_loss != 0.0:
                    loss = loss + const_train_loss


            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in optimizers:
                opt.step()


            total_loss += loss.item()
            total_literal_acc += literal_accuracy
            total_exact_acc += exact_accuracy
            compute_time_train += time.perf_counter() - compute_start
            num_batches += 1

            step += 1
            is_max = step >= current_train_layer_multiplier
            halt =  is_max | exact_per_example
            if use_act:
                halt = halt | overflow_mask

            if (is_max & exact_per_example).sum() > 0.5 and current_train_layer_multiplier < train_layer_multiplier:
                current_train_layer_multiplier += 1
                print("Increasing train_layer_multiplier to", current_train_layer_multiplier)

            # Create expanded masks matching the flattened sizes
            halt_clauses = halt.repeat_interleave(num_clauses) # Shape: [B * clauses]
            halt_literals = halt.repeat_interleave(num_vars * 2) # Shape: [B * vars * 2]

            # Detach history to stop backprop into past steps
            Hc = Hc.detach()
            Hl = Hl.detach()

            # Reset only the halted parts to Init state
            Hc[halt_clauses] = HcInit[halt_clauses]
            Hl[halt_literals] = HlInit[halt_literals]

            step = torch.where(halt, 0, step)

            # Optional mid-epoch evaluation based on wall-clock time
            if test_every_s > 0.0:
                now = time.perf_counter()
                if now - last_test_time >= test_every_s:
                    train_loss_so_far = total_loss / max(num_batches, 1)
                    train_literal_acc_so_far = total_literal_acc / max(num_batches, 1)
                    train_exact_acc_so_far = total_exact_acc / max(num_batches, 1)

                    mid_test_loss, mid_test_lit_acc, mid_test_exact_acc = evaluate_on_loader(
                        model=model,
                        device=device,
                        loader=test_loader,
                        num_layers=num_layers,
                        test_layer_multiplier=test_layer_multiplier,
                        print_prefix=(
                            f"[mid-test] epoch {epoch} time {now - training_start:.1f}s "
                            f"batch {batch_idx}/{len(train_loader)} | "
                            f"train_loss {train_loss_so_far:.4f} | "
                            f"train_acc {train_literal_acc_so_far:.3f} | "
                            f"train_exact_acc {train_exact_acc_so_far:.3f} | "
                        ),
                    )

                    model.train()
                    last_test_time = now

        train_loss = total_loss / max(num_batches, 1)
        train_literal_acc = total_literal_acc / max(num_batches, 1)
        train_exact_acc = total_exact_acc / max(num_batches, 1)

        # ---- evaluation on last 500 examples ----
        test_loss, test_literal_acc, test_exact_acc = evaluate_on_loader(
            model=model,
            device=device,
            loader=test_loader,
            num_layers=num_layers,
            test_layer_multiplier=test_layer_multiplier,
            print_prefix=(
                f"epoch {epoch} | "
                f"train_loss {train_loss:.4f} | "
                f"train_acc {train_literal_acc:.3f} | "
                f"train_exact_acc {train_exact_acc:.3f} | "
                f"train_compute_s {compute_time_train:.3f} | "
            ),
        )
    return model


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a tiny one-layer NeuroSAT model on random 3-SAT instances."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1024,
        help="Number of SAT problems to generate in the dataset.",
    )
    parser.add_argument(
        "--num-vars",
        type=int,
        default=2,
        help="Number of variables in each random 3-SAT instance.",
    )
    parser.add_argument(
        "--num-clauses",
        type=int,
        default=4,
        help="Number of clauses in each random 3-SAT instance.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Embedding dimension for literals and clauses.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of random SAT problems per optimization step.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of passes over the generated dataset.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=5,
        help="Number of message-passing iterations per batch.",
    )
    parser.add_argument(
        "--muon",
        action="store_true",
        help="Use Muon optimizer for 2D parameters and Adam for the rest.",
    )
    parser.add_argument(
        "--test-layer-multiplier",
        type=int,
        default=1,
        help="Run test unrolls up to this many times per batch, "
             "stopping early per example if an exact match is achieved.",
    )
    parser.add_argument(
        "--train-layer-multiplier",
        type=int,
        default=1,
        help="Target train unroll multiplier for the persistent pool.",
    )
    parser.add_argument(
        "--const-train-loss",
        type=float,
        default=0.0,
        help="Constant value added to the training loss each step.",
    )
    parser.add_argument(
        "--increase-multiplier-slowly",
        action="store_true",
        default=False,
        help="If set, start training at 1 and slowly increase up to train_layer_multiplier.",
    )
    parser.add_argument(
        "--test-every-s",
        type=float,
        default=0.0,
        help="If >0, run evaluation every this many seconds inside each epoch.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, save the trained model's state_dict to this path at the end of training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="If set, load a model state_dict from this path before training.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="If set, skip training and only evaluate on the full generated dataset.",
    )
    parser.add_argument(
        "--solve-only",
        action="store_true",
        help="If set, skip training and model evaluation and instead solve all generated instances with cadical_solve, reporting wall-clock solve_time_s.",
    )
    parser.add_argument(
        "--act",
        action="store_true",
        help="If set, use adaptive computation time (ACT) during training.",
    )
    args = parser.parse_args(argv)

    model = train(
        epochs=args.epochs,
        d=args.dim,
        lr=args.lr,
        num_vars=args.num_vars,
        num_clauses=args.num_clauses,
        batch_size=args.batch_size,
        dataset_size=args.iterations,
        use_muon=args.muon,
        num_layers=args.num_layers,
        test_layer_multiplier=args.test_layer_multiplier,
        train_layer_multiplier=args.train_layer_multiplier,
        test_every_s=args.test_every_s,
        increase_multiplier_slowly=args.increase_multiplier_slowly,
        const_train_loss=args.const_train_loss,
        load_path=args.load,
        eval_only=args.eval_only,
        solve_only=args.solve_only,
        use_act=args.act,
    )
    if args.save:
        save_dir = os.path.dirname(args.save)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Saved model state_dict to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
