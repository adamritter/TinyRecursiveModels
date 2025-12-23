# toy_neurosat.py
# These don't help: 2/3 layer MLP, skip connections, layer norm didn't really matter
# Harder problem didn't help
# python toy_neurosat.py --iterations 25000 --num-vars 7 --num-clauses 40 --lr 1e-3 --batch-size 64 --epochs 10 --dim 512 --muon --num-layers 16 --test-every-s 10 --test-layer-multiplier 12 --train-layer-multiplier 6
# python toy_neurosat.py --iterations 25000 --num-vars 20 --num-clauses 150 --lr 1e-3 --batch-size 64 --epochs 10 --dim 512 --muon --num-layers 16 --test-every-s 10 --test-layer-multiplier 12 --train-layer-multiplier 6 --save model20
# large model was trained for 7 hours on GH200 by: python toy_neurosat.py --iterations 250000 --num-vars 20 --num-clauses 150 --lr 1e-3 --batch-size 64 --epochs 80 --dim 512 --muon --num-layers 16 --test-every-s 10 --test-layer-multiplier 12 --train-layer-multiplier 6 --save large_model
# it overfits, test_exact_acc: 0 :)
# For larger problems need to increase test-every-s to avoid spending all time in eval
import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file

from sat_utils import cadical_solve, evaluate_solution, parse_cnf, random_3sat


def _sync_device(device: torch.device):
    """Synchronize the current device to get accurate timings."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except (AttributeError, RuntimeError):
            # Older PyTorch versions or CPU-only builds may not support this.
            pass


def _device_memory_gb(device: torch.device):
    """Return (allocated_gb, reserved_gb) for the active device if available."""
    allocated = reserved = None
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
    elif device.type == "mps":
        try:
            allocated = torch.mps.current_allocated_memory() / 1e9
            reserved = torch.mps.driver_allocated_memory() / 1e9
        except (AttributeError, RuntimeError):
            pass
    return allocated, reserved


def _rss_gb():
    """Return resident set size in GB if psutil is available."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return None


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


def build_graph_with_num_vars(clauses, num_vars: int):
    """
    Build a bipartite clause–literal graph for a CNF instance with a fixed var count.
    """
    if num_vars < 0:
        raise ValueError("num_vars must be non-negative.")

    m = len(clauses)
    if num_vars == 0:
        empty = torch.tensor([], dtype=torch.long)
        return 0, m, empty, empty, empty

    lit_index = {
        (v, s): 2 * (v - 1) + (0 if s == 1 else 1)
        for v in range(1, num_vars + 1)
        for s in (1, -1)
    }
    Ci, Lj = [], []
    for ci, clause in enumerate(clauses):
        for lit in clause:
            val = int(lit)
            v = abs(val)
            s = 1 if val > 0 else -1
            if v < 1 or v > num_vars:
                raise ValueError(f"Literal {lit} outside 1..{num_vars}.")
            Ci.append(ci)
            Lj.append(lit_index[(v, s)])
    Ci = torch.tensor(Ci, dtype=torch.long)
    Lj = torch.tensor(Lj, dtype=torch.long)

    flip = torch.empty(2 * num_vars, dtype=torch.long)
    for v in range(1, num_vars + 1):
        pos = lit_index[(v, 1)]
        neg = lit_index[(v, -1)]
        flip[pos] = neg
        flip[neg] = pos
    return num_vars, m, Ci, Lj, flip


def random_planted_3sat_torch(
    num_vars: int,
    num_clauses: int,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Fast torch-only generator for a planted 3-SAT instance.

    Returns:
        Ci (LongTensor): shape (num_clauses * 3,), clause index per literal occurrence.
        Lj (LongTensor): shape (num_clauses * 3,), literal indices in [0, 2*num_vars).
        flip (LongTensor): shape (2*num_vars,), index of the negation for each literal.
        target (FloatTensor): shape (2*num_vars,), literal truth values under the planted assignment.

    Notes:
        - Variables inside clauses are sampled uniformly with replacement; duplicates inside
          a clause are extremely rare when num_vars is large.
        - Signs are flipped randomly but each clause is forced to have at least one literal
          satisfied by the planted assignment.
        - Optionally pass a torch.Generator (on the same device) to make sampling reproducible.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # planted assignment in {-1, +1} for each variable
    assignment = torch.randint(
        0, 2, (num_vars,), device=device, dtype=torch.int8, generator=generator
    ) * 2 - 1  # -> {-1, +1}

    # sample 3 variable indices per clause (0-based)
    var_ids = torch.randint(
        0, num_vars, (num_clauses, 3), device=device, dtype=torch.int64, generator=generator
    )

    # base signs from planted assignment
    base_signs = assignment[var_ids].to(torch.int8)  # (num_clauses, 3)

    # random flips; start with potentially unsatisfied clauses
    flip_mask = torch.randint(
        0, 2, (num_clauses, 3), device=device, dtype=torch.int8, generator=generator
    )
    literal_signs = torch.where(flip_mask.bool(), -base_signs, base_signs)

    # ensure at least one satisfied literal per clause
    satisfied = literal_signs == base_signs
    has_sat = satisfied.any(dim=1)
    if not has_sat.all():
        need_fix = (~has_sat).nonzero(as_tuple=False).squeeze(1)
        if need_fix.numel() > 0:
            fix_positions = torch.randint(
                0, 3, (need_fix.numel(),), device=device, dtype=torch.int64, generator=generator
            )
            literal_signs[need_fix, fix_positions] = base_signs[need_fix, fix_positions]

    # Build Ci and Lj
    Ci = torch.repeat_interleave(torch.arange(num_clauses, device=device, dtype=torch.int64), 3)
    # literal index: 2*var for positive, 2*var+1 for negative
    is_neg = (literal_signs < 0).to(torch.int64)
    lit_indices = (var_ids * 2 + is_neg).reshape(-1)
    Lj = lit_indices

    # flip mapping for 2*num_vars literals
    flip = torch.empty(2 * num_vars, device=device, dtype=torch.long)
    pos = torch.arange(num_vars, device=device, dtype=torch.long) * 2
    neg = pos + 1
    flip[pos] = neg
    flip[neg] = pos

    # literal targets: even indices for positive literals, odd for negative
    pos_true = (assignment == 1).to(torch.float32)
    target = torch.empty(2 * num_vars, device=device, dtype=torch.float32)
    target[0::2] = pos_true
    target[1::2] = 1.0 - pos_true

    return Ci, Lj, flip, target


def random_3sat_graph(num_vars=10, num_clauses=40, planted=False):
    """
    Sample a random satisfiable 3-SAT instance and return its graph.

    Returns:
        clauses (list[tuple[int]]): CNF clauses for the instance.
        assignment (tuple[int]): one satisfying assignment (0/1 per variable).
        n, m, Ci, Lj, flip: outputs of `build_graph(clauses)`.
    """
    clauses, assignment = random_3sat(num_vars=num_clauses and num_vars, num_clauses=num_clauses, planted=planted)
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
    planted=False,
):
    """Pre-generate (and cache) a fixed dataset of satisfiable 3-SAT instances."""
    os.makedirs(cache_dir, exist_ok=True)
    planted_suffix = "_planted" if planted else ""
    filename = (
        f"toy_neurosat_dataset_size{dataset_size}_"
        f"vars{num_vars}_clauses{num_clauses}_v3{planted_suffix}.pt"
    )
    path = os.path.join(cache_dir, filename)

    if os.path.exists(path):
        t0 = time.perf_counter()
        data = torch.load(path)
        load_time = time.perf_counter() - t0
        print(f"Loaded dataset from {path} in {load_time:.3f}s")
        return data

    def _make_one(_):
        clauses, assignment_raw = random_3sat(num_vars, num_clauses, planted=planted)
        assignment = [1 if int(lit) > 0 else 0 for lit in assignment_raw]
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
        f"(vars={num_vars}, clauses={num_clauses}, planted={planted}) "
        f"in {create_time:.3f}s, saved to {path}"
    )

    torch.save(dataset, path)
    return dataset


def get_exact_accuracy(exact_per_example: torch.Tensor):
    """Compute exact accuracy from BoolTensor of per-example exactness."""
    return exact_per_example.float().mean().item()


class Problem:
    """
    Container for a single SAT problem in graph form (no batching).
    """

    def __init__(
        self,
        Ci: torch.Tensor,
        Lj: torch.Tensor,
        flip: torch.Tensor,
        num_vars: int,
        num_clauses: int,
        target: Optional[torch.Tensor] = None,
    ):
        self.Ci = Ci
        self.Lj = Lj
        self.flip = flip
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.target = target

    @classmethod
    def from_cnf(cls, filename: str):
        """
        Build a Problem from a DIMACS CNF file.
        """
        content = Path(filename).read_text(encoding="utf-8")
        num_vars, clauses = parse_cnf(content)
        num_vars, num_clauses, Ci, Lj, flip = build_graph_with_num_vars(clauses, num_vars)
        return cls(Ci=Ci, Lj=Lj, flip=flip, num_vars=num_vars, num_clauses=num_clauses, target=None)

    def num_literals(self) -> int:
        return 2 * self.num_vars

    def to(self, device):
        return Problem(
            Ci=self.Ci.to(device),
            Lj=self.Lj.to(device),
            flip=self.flip.to(device),
            num_vars=self.num_vars,
            num_clauses=self.num_clauses,
            target=None if self.target is None else self.target.to(device),
        )

    def check(self, scores: torch.Tensor, use_target: bool = False):
        """
        Return BoolTensor[1] indicating if the single problem is satisfied.
        """
        per_problem = self.num_literals()
        if scores.numel() != per_problem:
            raise ValueError("scores length must match literals in the problem")

        if use_target:
            if self.target is None:
                raise ValueError("Target is required when use_target=True")
            scores_view = scores.view(1, per_problem)
            target_view = self.target.view(1, per_problem)

            num_vars = per_problem // 2
            scores_pair = scores_view.view(1, num_vars, 2)
            target_pair = target_view.view(1, num_vars, 2)

            pred_true = scores_pair[..., 0] >= scores_pair[..., 1]
            target_true = target_pair[..., 0] > 0.5
            exact_per_example = (pred_true == target_true).all(dim=1)
            return exact_per_example

        literal_true = scores >= scores[self.flip]
        tie_mask = scores == scores[self.flip]
        lit_true_for_occurrence = literal_true[self.Lj]

        clause_true_counts = torch.zeros(
            self.num_clauses, dtype=torch.int64, device=scores.device
        )
        clause_true_counts.index_add_(0, self.Ci, lit_true_for_occurrence.to(torch.int64))
        clause_sat = clause_true_counts > 0

        has_tie = tie_mask.view(1, per_problem).any(dim=1)
        exact_per_example = clause_sat.view(1, self.num_clauses).all(dim=1) & (~has_tie)
        return exact_per_example


class ProblemSet:
    """
    Container for a batch of same-sized SAT problems in graph form.

    Attributes:
        Ci: (E,) clause indices for each literal occurrence.
        Lj: (E,) literal indices in [0, 2*num_vars) matching ``Ci``.
        flip: (B * 2*num_vars,) index of ¬ℓ for each literal ℓ.
        num_vars: number of variables per problem.
        num_clauses: number of clauses per problem.
        target: optional literal targets (can be None).
    """

    def __init__(
        self,
        Ci: torch.Tensor,
        Lj: torch.Tensor,
        flip: torch.Tensor,
        num_vars: int,
        num_clauses: int,
        target: Optional[torch.Tensor] = None,
    ):
        self.Ci = Ci
        self.Lj = Lj
        self.flip = flip
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.target = target

    def num_literals(self) -> int:
        return 2 * self.num_vars

    def batch_size(self) -> int:
        literals = self.num_literals()
        if literals == 0:
            return 0
        total = self.flip.numel()
        if total % literals != 0:
            raise ValueError("flip length is not divisible by literals per problem")
        return total // literals

    def to(self, device):
        """Return a new ProblemSet with tensors moved to ``device``."""
        return ProblemSet(
            Ci=self.Ci.to(device),
            Lj=self.Lj.to(device),
            flip=self.flip.to(device),
            num_vars=self.num_vars,
            num_clauses=self.num_clauses,
            target=None if self.target is None else self.target.to(device),
        )

    @classmethod
    def build_batch(cls, samples):
        """Collate a batch of pre-built graphs into one ProblemSet."""
        all_Ci = []
        all_Lj = []
        all_flip = []
        all_targets = []
        clause_offset = 0
        lit_offset = 0
        per_problem = None
        num_clauses_per_problem = None

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
            if num_clauses_per_problem is None:
                num_clauses_per_problem = num_clauses
            elif num_clauses_per_problem != num_clauses:
                raise ValueError("All problems in a batch must have the same clause count")

            clause_offset += num_clauses
            lit_offset += num_literals

        Ci = torch.cat(all_Ci, dim=0)
        Lj = torch.cat(all_Lj, dim=0)
        flip = torch.cat(all_flip, dim=0)
        target = torch.cat(all_targets, dim=0)

        if per_problem is None or num_clauses_per_problem is None:
            raise ValueError("Empty batch encountered when building ProblemSet")

        num_vars = per_problem // 2
        return cls(
            Ci=Ci,
            Lj=Lj,
            flip=flip,
            num_vars=num_vars,
            num_clauses=num_clauses_per_problem,
            target=target,
        )

    def check(self, scores: torch.Tensor, use_target=False):
        """
        Given logits and the clause–literal graph, check whether the model's argmax
        assignment satisfies each CNF in the batch.

        Args:
            scores: (B * per_problem,) logits for literals.
            self: selfet describing the CNF batch.

        Returns:
            BoolTensor[B]: True if every clause in the problem is satisfied.
        """
        if use_target:
            _, _, exact_per_example = compute_literal_metrics(scores, self)
            return exact_per_example
        with torch.no_grad():
            per_problem = self.num_literals()
            total_literals = scores.numel()
            if total_literals % per_problem != 0:
                raise ValueError("scores length is not divisible by per_problem")
            batch_size = total_literals // per_problem

            total_clauses = self.num_clauses * batch_size
            if batch_size == 0:
                return torch.zeros(0, dtype=torch.bool, device=scores.device)
            clauses_per_problem = total_clauses // batch_size

            # Literal truth: a literal is true if its score >= its negation's score.
            # Ties mean the variable did not converge, so the whole example is marked unsolved.
            literal_true = scores >= scores[self.flip]  # (total_literals,)
            tie_mask = scores == scores[self.flip]      # (total_literals,)
            lit_true_for_occurrence = literal_true[self.Lj]  # (E,)

            clause_true_counts = torch.zeros(
                total_clauses, dtype=torch.int64, device=scores.device
            )
            clause_true_counts.index_add_(0, self.Ci, lit_true_for_occurrence.to(torch.int64))
            clause_sat = clause_true_counts > 0  # (total_clauses,)

            clause_all = clause_sat.view(batch_size, clauses_per_problem).all(dim=1)
            has_tie = tie_mask.view(batch_size, per_problem).any(dim=1)
            exact_per_example = clause_all & (~has_tie)
            # set all to false for now:
            # exact_per_example = torch.zeros_like(exact_per_example, dtype=torch.bool)
        return exact_per_example


def compute_literal_metrics(scores: torch.Tensor, problems: ProblemSet):
    """
    Compute literal-wise and exact accuracy from logits and targets.

    Args:
        scores: (B * per_problem,) logits for literals.
        problems: ProblemSet with targets populated.

    Returns:
        literal_accuracy (float), exact_accuracy (float), exact_per_example (BoolTensor[B])
    """
    with torch.no_grad():
        if problems.target is None:
            raise ValueError("compute_literal_metrics requires targets in ProblemSet")

        per_problem = problems.num_literals()
        target = problems.target
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
        exact_accuracy = get_exact_accuracy(exact_per_example)
    return literal_accuracy, exact_accuracy, exact_per_example


def _default_solution_path(cnf_path: str) -> str:
    path = Path(cnf_path)
    if path.suffix.lower() == ".cnf":
        return str(path.with_suffix(".sol"))
    return f"{path}.sol"


def _write_solution_file(path: str, assignment, is_sat: bool) -> None:
    status = "SATISFIABLE" if is_sat else "UNKNOWN"
    lines = [f"s {status}"]
    if assignment:
        literals = " ".join(str(int(lit)) for lit in assignment)
        lines.append(f"v {literals} 0")
    else:
        lines.append("v 0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scores_to_assignment(scores: torch.Tensor, num_vars: int):
    assignment = []
    for var in range(1, num_vars + 1):
        pos_idx = 2 * (var - 1)
        neg_idx = pos_idx + 1
        if scores[pos_idx] >= scores[neg_idx]:
            assignment.append(var)
        else:
            assignment.append(-var)
    return assignment


def solve_cnf_file(
    model,
    device,
    cnf_path: str,
    output_path: str,
    num_layers: int,
    test_layer_multiplier: int,
) -> bool:
    problem = Problem.from_cnf(cnf_path).to(device)

    per_problem = problem.num_literals()
    if per_problem == 0:
        _write_solution_file(output_path, [], True)
        return True

    B = 1
    num_clauses_total = problem.num_clauses * B
    num_literals_total = per_problem * B

    Hc0 = model.clause_init.unsqueeze(0).expand(num_clauses_total, -1)
    Hl0 = model.literal_init.unsqueeze(0).expand(num_literals_total, -1)

    solved = torch.zeros(B, dtype=torch.bool, device=device)
    final_scores_view = torch.zeros(B, per_problem, device=device)

    Hl = Hl0
    Hc = Hc0
    scores_view = None

    model.eval()
    with torch.no_grad():
        attempts = max(1, int(test_layer_multiplier))
        for _attempt in range(attempts):
            for _ in range(num_layers):
                Hl, Hc = model(Hc, Hl, problem.Ci, problem.Lj, problem.flip)
            scores = model.readout(Hl).squeeze(-1)
            scores_view = scores.view(B, per_problem)

            exact_per_example = problem.check(scores)
            newly_solved = exact_per_example & ~solved
            if newly_solved.any():
                final_scores_view[newly_solved] = scores_view[newly_solved]
                solved = solved | newly_solved
            if solved.all():
                break

    if scores_view is None:
        raise RuntimeError("No scores produced during evaluation.")

    if not solved.all():
        final_scores_view[~solved] = scores_view[~solved]

    final_scores = final_scores_view.view(-1).detach().cpu()
    assignment = _scores_to_assignment(final_scores, problem.num_vars)
    is_sat = evaluate_solution(parse_cnf(Path(cnf_path).read_text(encoding="utf-8"))[1], assignment)
    if is_sat:
        _write_solution_file(output_path, assignment, is_sat)
    return is_sat


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
            problems = ProblemSet.build_batch(batch).to(device)

            per_problem = problems.num_literals()
            B = problems.batch_size()
            num_clauses_total = problems.num_clauses * B
            num_literals_total = per_problem * B
            Ci, Lj, flip = problems.Ci, problems.Lj, problems.flip

            Hc0 = model.clause_init.unsqueeze(0).expand(num_clauses_total, -1)
            Hl0 = model.literal_init.unsqueeze(0).expand(num_literals_total, -1)

            # "Cheating" test-time unrolls: per-example early stopping when exact.
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
                exact_per_example = problems.check(scores)

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

            loss = F.binary_cross_entropy_with_logits(final_scores, problems.target)
            literal_accuracy, exact_accuracy, _ = compute_literal_metrics(
                final_scores, problems
            )
            exact_accuracy = get_exact_accuracy(problems.check(final_scores))

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
    planted=False,
    debug_log_interval=0,
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
        state = load_file(load_path)
        model.load_state_dict(state)
        print(f"Loaded model state_dict from {load_path}")

    model.to(device)

    graphs = generate_dataset(
        num_vars=num_vars,
        num_clauses=num_clauses,
        dataset_size=dataset_size,
        planted=planted,
    )
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
    test_size = min(500, num_samples // 10) if num_samples > 1 else 0
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
    problems_current: Optional[ProblemSet] = None
    phalt_total = torch.zeros(batch_size, dtype=torch.float32, device=device)
    # set starting multiplier depending on flag:
    # - if increase_multiplier_slowly: start at 1 and grow
    # - otherwise: start directly at the requested train_layer_multiplier
    current_train_layer_multiplier = 1 if increase_multiplier_slowly else train_layer_multiplier

    debug_enabled = debug_log_interval > 0

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()

        total_loss = 0.0
        total_literal_acc = 0.0
        total_exact_acc = 0.0
        num_batches = 0
        compute_time_train = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):

            # Timings per batch (all zero when debug is disabled).
            t_load = t_fwd = t_bwd = t_step = t_reset = 0.0

            if debug_enabled:
                _sync_device(device)
                t0 = time.perf_counter()

            problems_new = ProblemSet.build_batch(batch).to(device)
            # Dynamic problem creation:
            if planted:
                # Create new problems using random planted 3-SAT generator torch:
                batch2 = []
                for _ in range(batch_size):
                    Ci, Lj, flip, target = random_planted_3sat_torch(
                        num_vars=num_vars,
                        num_clauses=num_clauses,
                        device=device,
                    )
                    num_clauses_total = num_clauses
                    num_literals = num_vars * 2
                    per_problem = num_literals
                    batch2.append((Ci, Lj, flip, target, num_clauses_total, num_literals, per_problem))
                problems_new = ProblemSet.build_batch(batch2).to(device)

            if debug_enabled:
                _sync_device(device)
                t_load = time.perf_counter() - t0

            if problems_current is None:
                problems_current = problems_new
            else:
                if (
                    problems_current.num_literals() != problems_new.num_literals()
                    or problems_current.num_clauses != problems_new.num_clauses
                ):
                    raise ValueError("All problems must share size for persistent training")

                reset_mask = (step == 0)  # Shape: [batch_size] (Boolean)

                if reset_mask.any():
                    # 2. Expand masks for FLATTENED inputs
                    # For Ci/Lj (clauses): Expand mask to [Batch * num_clauses * 3]
                    edges_per_problem = problems_current.num_clauses * 3
                    reset_mask_clauses = reset_mask.repeat_interleave(edges_per_problem)

                    # For Flip/target (variables): Expand mask to [Batch * num_vars * 2]
                    reset_mask_vars = reset_mask.repeat_interleave(problems_current.num_literals())

                    # 3. Apply Updates using the Expanded Masks
                    problems_current.Ci[reset_mask_clauses] = problems_new.Ci[reset_mask_clauses]
                    problems_current.Lj[reset_mask_clauses] = problems_new.Lj[reset_mask_clauses]
                    problems_current.flip[reset_mask_vars] = problems_new.flip[reset_mask_vars]
                    if problems_current.target is not None and problems_new.target is not None:
                        problems_current.target[reset_mask_vars] = problems_new.target[reset_mask_vars]
                    phalt_total[reset_mask] = 0.0

            num_vars_local = problems_current.num_vars

            if debug_enabled:
                _sync_device(device)
                t1 = time.perf_counter()

            compute_start = time.perf_counter()
            # apply a stack of num_layers one-step updates
            for _ in range(num_layers):
                Hl, Hc = model(Hc, Hl, problems_current.Ci, problems_current.Lj, problems_current.flip)
            scores = model.readout(Hl).squeeze(-1)

            if debug_enabled:
                _sync_device(device)
                t_fwd = time.perf_counter() - t1

            literal_accuracy, exact_accuracy, exact_per_example = compute_literal_metrics(
                scores, problems_current
            )
            exact_per_example = problems_current.check(scores)
            exact_accuracy = get_exact_accuracy(exact_per_example)
            if use_act:
                phalt_current = torch.sigmoid(
                    model.halt(
                        Hl.reshape(batch_size, num_vars_local * 2, model.d)
                        .mean(dim=1)
                        .squeeze(-1)
                    ).squeeze(-1)
                )
                phalt_current_detach = phalt_current.detach()
                overflow_mask = (phalt_total + phalt_current_detach) > 0.99
                phalt_total = (phalt_total + phalt_current).detach()
                losses = (
                    F.binary_cross_entropy_with_logits(scores, problems_current.target, reduction='none')
                    .reshape(batch_size, num_vars_local * 2)
                    .mean(dim=1)
                )
                phalt_current_loss = torch.where(overflow_mask, 1.0 - phalt_total, phalt_current)
                loss = ((losses * phalt_current_loss + (1.0 - phalt_current_loss) * const_train_loss ).mean()) * current_train_layer_multiplier
            else:
                losses = (
                    F.binary_cross_entropy_with_logits(scores, problems_current.target, reduction='none')
                    .reshape(batch_size, num_vars_local * 2)
                    .mean(dim=1)
                )
                #losses = torch.where(exact_per_example, losses * 10.0, losses)
                #loss = F.binary_cross_entropy_with_logits(scores, target)
                loss = losses.mean()
                if const_train_loss != 0.0:
                    loss = loss + const_train_loss

            if debug_enabled:
                _sync_device(device)
                t_bwd_start = time.perf_counter()
            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            if debug_enabled:
                _sync_device(device)
                t_bwd = time.perf_counter() - t_bwd_start
                t_step_start = time.perf_counter()
            for opt in optimizers:
                opt.step()
            if debug_enabled:
                _sync_device(device)
                t_step = time.perf_counter() - t_step_start


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
            halt_clauses = halt.repeat_interleave(problems_current.num_clauses) # Shape: [B * clauses]
            halt_literals = halt.repeat_interleave(problems_current.num_literals()) # Shape: [B * vars * 2]

            # Detach history to stop backprop into past steps
            Hc = Hc.detach()
            Hl = Hl.detach()

            if debug_enabled:
                _sync_device(device)
                t_reset_start = time.perf_counter()

            # Reset only the halted parts to Init state
            Hc[halt_clauses] = HcInit[halt_clauses]
            Hl[halt_literals] = HlInit[halt_literals]

            step = torch.where(halt, 0, step)

            if debug_enabled:
                _sync_device(device)
                t_reset = time.perf_counter() - t_reset_start

            if debug_enabled and (batch_idx % debug_log_interval == 0):
                alloc_gb, reserved_gb = _device_memory_gb(device)
                rss_gb = _rss_gb()
                msg = (
                    f"[debug] batch {batch_idx} | "
                    f"load {t_load:.4f}s fwd {t_fwd:.4f}s "
                    f"bwd {t_bwd:.4f}s step {t_step:.4f}s reset {t_reset:.4f}s"
                )
                if alloc_gb is not None:
                    msg += f" | mem {alloc_gb:.2f}G"
                    if reserved_gb is not None:
                        msg += f" (reserved {reserved_gb:.2f}G)"
                if rss_gb is not None:
                    msg += f" | rss {rss_gb:.2f}G"
                print(msg)

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
        "--planted",
        action="store_true",
        help="If set, generate planted-solution 3-SAT instances instead of unique ones.",
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
        "--debug-log-interval",
        type=int,
        default=0,
        help="If >0, print per-batch timing/memory stats every this many batches.",
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
        "--eval",
        type=str,
        default=None,
        help="If set, solve the provided DIMACS CNF file and write a solution file.",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Optional output path for the DIMACS solution file (default: <cnf>.sol).",
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

    if args.eval:
        if args.load is None:
            raise ValueError("--eval requires --load to provide model weights.")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        model = OneLayerNeuroSAT(args.dim, use_act=args.act)
        state = load_file(args.load)
        model.load_state_dict(state)
        model.to(device)
        print(f"Loaded model state_dict from {args.load}")

        output_path = args.eval_output or _default_solution_path(args.eval)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        is_sat = solve_cnf_file(
            model=model,
            device=device,
            cnf_path=args.eval,
            output_path=output_path,
            num_layers=args.num_layers,
            test_layer_multiplier=args.test_layer_multiplier,
        )
        if is_sat:
            print(f"Wrote SAT solution to {output_path}")
        else:
            print("UNKNOWN")
        return 0

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
        planted=args.planted,
        debug_log_interval=args.debug_log_interval,
    )
    if args.save:
        save_dir = os.path.dirname(args.save)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_file(model.state_dict(), args.save)
        print(f"Saved model state_dict to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
