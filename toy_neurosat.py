# toy_neurosat.py
import argparse
import os
import sys
import random
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product, combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class OneLayerNeuroSAT(nn.Module):
    def __init__(self, d=32, use_layernorm=True, num_layers=5):
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
        self.d = d
        self.num_layers = num_layers
        # learnable scales for residual updates
        self.scale_lit = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.scale_clause = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # optional normalization for more stable training
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.norm_c = nn.LayerNorm(d)
            self.norm_l = nn.LayerNorm(d)
        # GRU-style recurrence over message-passing steps
        self.gru_lit = nn.GRUCell(d, d)
        self.gru_clause = nn.GRUCell(d, d)

    def forward(self, Hc, Hl, Ci, Lj, flip_index, layer_multiplier=1.0):
        """
        Hc : (m, d) initial clause embeddings
        Hl : (2n, d) initial literal embeddings
        Ci -> Lj edges: two tensors (src, dst) of equal length Ecl
        flip_index : (2n,) tensor giving the index of ¬ℓ for each literal ℓ
        """
        # Initialize hidden states for clauses and literals
        h_clause = Hc
        h_lit = Hl

        for _ in range(round(self.num_layers * layer_multiplier)):
            # -------- clause -> literal --------
            msg_c2l = self.Wc2l(h_clause)                 # (m, d)
            agg_c2l = torch.zeros_like(h_lit)
            agg_c2l.index_add_(0, Lj, msg_c2l[Ci])        # sum messages onto literals

            # -------- literal -> neg-literal ("flip") --------
            agg_flip = self.Wflip(h_lit[flip_index])      # (2n, d)

            # -------- update literals via GRU cell --------
            lit_in = self.scale_lit * (agg_c2l + agg_flip)
            h_lit = self.gru_lit(lit_in, h_lit)

            # -------- literal -> clause --------
            msg_l2c = self.Wl2c(h_lit)                    # (2n, d)
            agg_l2c = torch.zeros_like(h_clause)
            agg_l2c.index_add_(0, Ci, msg_l2c[Lj])        # aggregate onto clauses

            # -------- update clauses via GRU cell --------
            clause_in = self.scale_clause * agg_l2c
            h_clause = self.gru_clause(clause_in, h_clause)

            # -------- layer normalization (optional) --------
            if self.use_layernorm:
                h_clause = self.norm_c(h_clause)
                h_lit = self.norm_l(h_lit)

        # -------- read-out: score each literal from final h_lit --------
        scores = self.readout(h_lit).squeeze(-1)         # (2n,)
        return scores, h_lit, h_clause


# ---------- utility: build tiny SAT problems ----------
def random_3sat(num_vars=10, num_clauses=40):
    """
    Returns: list[tuple[int]] of length num_clauses.
             Each clause is a tuple of signed ints (e.g. -3 means ¬x3).
             Guarantees satisfiable by sampling until it finds one.
    """
    lits = list(range(1, num_vars + 1))
    all_vars = set(lits)
    while True:
        clauses = []
        for _ in range(num_clauses):
            clause = []
            for _ in range(3):
                v = random.choice(lits)
                s = random.choice((1, -1))
                clause.append(s * v)
            clauses.append(tuple(clause))

        used_vars = {abs(l) for c in clauses for l in c}
        if used_vars != all_vars:
            continue

        # brute-force test (only feasible for toy sizes)
        for assignment in product([0, 1], repeat=num_vars):
            if all(any((l > 0) == assignment[abs(l) - 1] for l in c) for c in clauses):
                return clauses, assignment  # satisfiable formula + one witness


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
        f"vars{num_vars}_clauses{num_clauses}_v2.pt"
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

    max_workers = min(32, (os.cpu_count() or 4), dataset_size)
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


# ---------- tiny training loop (cross-entropy over literal pairs) ----------
def train_toy(
    epochs=10,
    d=32,
    lr=1e-3,
    num_vars=2,
    num_clauses=4,
    batch_size=32,
    dataset_size=1024,
    use_muon=False,
    num_layers=5,
    test_layer_multiplier=1.0,
):
    model = OneLayerNeuroSAT(d, num_layers=num_layers)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

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

    graphs = generate_dataset(num_vars=num_vars, num_clauses=num_clauses, dataset_size=dataset_size)
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
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()

        total_loss = 0.0
        total_literal_acc = 0.0
        total_exact_acc = 0.0
        num_batches = 0
        compute_time_train = 0.0

        for batch in train_loader:
            Ci, Lj, flip, target, num_clauses_total, num_literals_total, per_problem = (
                build_batch_from_samples(batch)
            )

            Ci = Ci.to(device)
            Lj = Lj.to(device)
            flip = flip.to(device)
            target = target.to(device)

            Hc = model.clause_init.unsqueeze(0).expand(num_clauses_total, -1)
            Hl = model.literal_init.unsqueeze(0).expand(num_literals_total, -1)

            compute_start = time.perf_counter()
            # apply recurrent OneLayerNeuroSAT (LSTM-style) once with default depth
            scores, Hl, Hc = model(Hc, Hl, Ci, Lj, flip)

            loss = F.binary_cross_entropy_with_logits(scores, target)

            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in optimizers:
                opt.step()

            with torch.no_grad():
                batch_size_actual = target.size(0) // (per_problem)
                scores_view = scores.view(batch_size_actual, per_problem)
                target_view = target.view(batch_size_actual, per_problem)

                num_vars = per_problem // 2
                scores_pair = scores_view.view(batch_size_actual, num_vars, 2)
                target_pair = target_view.view(batch_size_actual, num_vars, 2)

                # per-variable decision: choose more confident between (pos, neg)
                pred_true = (scores_pair[..., 0] >= scores_pair[..., 1])
                target_true = target_pair[..., 0] > 0.5

                correct = (pred_true == target_true)
                literal_accuracy = correct.float().mean().item()
                exact_accuracy = correct.all(dim=1).float().mean().item()

            total_loss += loss.item()
            total_literal_acc += literal_accuracy
            total_exact_acc += exact_accuracy
            compute_time_train += time.perf_counter() - compute_start
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)
        train_literal_acc = total_literal_acc / max(num_batches, 1)
        train_exact_acc = total_exact_acc / max(num_batches, 1)

        # ---- evaluation on last 500 examples ----
        model.eval()
        test_loss = 0.0
        test_literal_acc = 0.0
        test_exact_acc = 0.0
        test_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                Ci, Lj, flip, target, num_clauses_total, num_literals_total, per_problem = (
                    build_batch_from_samples(batch)
                )

                Ci = Ci.to(device)
                Lj = Lj.to(device)
                flip = flip.to(device)
                target = target.to(device)

                Hc = model.clause_init.unsqueeze(0).expand(num_clauses_total, -1)
                Hl = model.literal_init.unsqueeze(0).expand(num_literals_total, -1)

                compute_start = time.perf_counter()
                scores, Hl, Hc = model(
                    Hc, Hl, Ci, Lj, flip, layer_multiplier=test_layer_multiplier
                )

                loss = F.binary_cross_entropy_with_logits(scores, target)

                batch_size_actual = target.size(0) // (per_problem)
                scores_view = scores.view(batch_size_actual, per_problem)
                target_view = target.view(batch_size_actual, per_problem)

                num_vars = per_problem // 2
                scores_pair = scores_view.view(batch_size_actual, num_vars, 2)
                target_pair = target_view.view(batch_size_actual, num_vars, 2)

                pred_true = (scores_pair[..., 0] >= scores_pair[..., 1])
                target_true = target_pair[..., 0] > 0.5

                correct = (pred_true == target_true)
                literal_accuracy = correct.float().mean().item()
                exact_accuracy = correct.all(dim=1).float().mean().item()

                test_loss += loss.item()
                test_literal_acc += literal_accuracy
                test_exact_acc += exact_accuracy
                test_batches += 1

        test_loss /= max(test_batches, 1)
        test_literal_acc /= max(test_batches, 1)
        test_exact_acc /= max(test_batches, 1)

        print(
            f"epoch {epoch} | "
            f"train_loss {train_loss:.4f} | train_acc {train_literal_acc:.3f} | train_exact_acc {train_exact_acc:.3f} | "
            f"test_loss {test_loss:.4f} | test_acc {test_literal_acc:.3f} | test_exact_acc {test_exact_acc:.3f} | "
            f"train_compute_s {compute_time_train:.3f}"
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
        type=float,
        default=1.0,
        help="Multiplier for the number of message-passing layers during evaluation.",
    )
    args = parser.parse_args(argv)

    train_toy(
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
