#!/usr/bin/env python3
"""
Run a trained Tiny Recursive Model on a DIMACS CNF instance read from stdin.

The model continuously recurses until its adaptive computation halts, then a
SAT assignment is decoded from the logits and emitted as a DIMACS `v ...` line.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.functions import load_model_class

NUM_VARS = 500
VOCAB_SIZE = 2 * NUM_VARS + 1  # includes padding token at index 0
SEQ_LEN = 8250
IGNORE_LABEL_ID = -100


def parse_dimacs(lines: Iterable[str]) -> Tuple[int, List[List[int]]]:
    """Parse DIMACS CNF from the given lines."""
    num_vars = None
    clauses: List[List[int]] = []
    current_clause: List[int] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) < 4 or parts[1].lower() != "cnf":
                raise ValueError("Expected header of form 'p cnf <vars> <clauses>'.")
            num_vars = int(parts[2])
            continue

        for token in line.split():
            lit = int(token)
            if lit == 0:
                if not current_clause:
                    raise ValueError("Encountered clause terminator without literals.")
                if len(current_clause) != 3:
                    raise ValueError("All clauses must contain exactly 3 literals.")
                clauses.append(current_clause)
                current_clause = []
            else:
                current_clause.append(lit)

    if current_clause:
        raise ValueError("unterminated clause at end of file.")

    if num_vars is None:
        raise ValueError("CNF header not found.")
    if not clauses:
        raise ValueError("No clauses provided.")

    max_var = max(abs(lit) for clause in clauses for lit in clause)
    if max_var > NUM_VARS:
        raise ValueError(f"Variable {max_var} exceeds supported limit ({NUM_VARS}).")

    if num_vars < max_var:
        num_vars = max_var

    return num_vars, clauses


def literal_to_token(lit: int) -> int:
    """Convert a literal to the token ID used during training."""
    return lit if lit > 0 else NUM_VARS - lit


def token_to_literal(token: int) -> int:
    """Convert a token ID back to a literal."""
    if token == 0:
        return 0
    if 1 <= token <= NUM_VARS:
        return token
    return NUM_VARS - token


def encode_problem(
    clauses: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, Dict[int, List[int]], List[List[int]], List[List[int]]]:
    """Encode a CNF instance into the fixed-length token sequence used by the TRM."""
    tokens = np.zeros(SEQ_LEN, dtype=np.int32)
    literal_positions: Dict[int, List[int]] = {}
    clause_literals: List[List[int]] = []
    clause_positions: List[List[int]] = []

    cursor = 0
    for clause in clauses:
        clause_literals.append(list(clause))
        indices: List[int] = []
        for lit in clause:
            if cursor >= SEQ_LEN:
                raise ValueError("Encoded clause sequence exceeds model sequence length.")
            token = literal_to_token(lit)
            tokens[cursor] = token
            literal_positions.setdefault(lit, []).append(cursor)
            indices.append(cursor)
            cursor += 1
        clause_positions.append(indices)

    return tokens, literal_positions, clause_literals, clause_positions


def decode_assignment_from_logits(
    logits: np.ndarray,
    literal_positions: Dict[int, List[int]],
    clause_positions: Sequence[Sequence[int]],
    clause_literals: Sequence[Sequence[int]],
    num_vars: int,
    pred_tokens: np.ndarray,
) -> List[int]:
    """Produce a variable assignment by aggregating literal scores clause-wise."""
    pos_scores = np.zeros(num_vars + 1, dtype=np.float64)
    neg_scores = np.zeros(num_vars + 1, dtype=np.float64)

    for lit, positions in literal_positions.items():
        if not positions:
            continue
        token = literal_to_token(lit)
        values = logits[positions, token]
        if lit > 0:
            pos_scores[abs(lit)] += float(np.sum(values))
        else:
            neg_scores[abs(lit)] += float(np.sum(values))

    assignment = []
    for var in range(1, num_vars + 1):
        pos = pos_scores[var]
        neg = neg_scores[var]
        if pos == 0.0 and neg == 0.0:
            assignment.append(var)
        elif pos >= neg:
            assignment.append(var)
        else:
            assignment.append(-var)

    def satisfied(clause: Sequence[int], values: Sequence[int]) -> bool:
        truth_map = {abs(var): (var > 0) for var in values}
        for lit in clause:
            value = truth_map.get(abs(lit), lit > 0)
            if (lit > 0 and value) or (lit < 0 and not value):
                return True
        return False

    def apply_literal(lit: int) -> None:
        var = abs(lit)
        assignment[var - 1] = lit

    # Ensure every clause is satisfied by nudging conflicted clauses using logits.
    changed = True
    for _ in range(2):  # iterate a couple of times in case adjustments cascade
        if not changed:
            break
        changed = False
        for clause_idx, clause in enumerate(clause_literals):
            if satisfied(clause, assignment):
                continue
            best_lit = None
            best_score = -float("inf")
            for lit, pos in zip(clause, clause_positions[clause_idx]):
                token = literal_to_token(lit)
                score = float(logits[pos, token])
                if score > best_score:
                    best_score = score
                    best_lit = lit
            if best_lit is not None:
                apply_literal(best_lit)
                changed = True

    # Fallback: align with model argmax at remaining unsatisfied clauses.
    for clause_idx, clause in enumerate(clause_literals):
        if satisfied(clause, assignment):
            continue
        for token in pred_tokens[clause_positions[clause_idx]]:
            lit = token_to_literal(int(token))
            if lit == 0:
                continue
            apply_literal(lit)
            break

    return assignment


def build_model(config_path: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Instantiate the TRM + ACT loss head and load its weights."""
    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)  # type: ignore[arg-type]

    if not isinstance(config_dict, dict):
        raise TypeError("Model config must resolve to a dictionary.")
    config_dict = dict(config_dict)

    model_identifier = config_dict.pop("name")
    loss_cfg = config_dict.pop("loss")
    if not isinstance(loss_cfg, dict):
        raise TypeError("Loss configuration must be a mapping.")
    loss_cfg = dict(loss_cfg)

    model_kwargs = dict(config_dict)
    model_kwargs.update(
        {
            "batch_size": 1,
            "seq_len": SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "num_puzzle_identifiers": 1,
        }
    )

    model_cls = load_model_class(model_identifier)
    loss_cls = load_model_class(loss_cfg.pop("name"))

    model = model_cls(model_kwargs)
    model = loss_cls(model, **loss_cfg)  # type: ignore[arg-type]
    model.to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        if isinstance(state_dict, dict) and all(key.startswith("_orig_mod.") for key in state_dict.keys()):
            remapped = {key[len("_orig_mod.") :]: value for key, value in state_dict.items()}
            model.load_state_dict(remapped, strict=True)
        else:
            raise

    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Recursively apply the model until all sequences halt."""
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        carry = model.initial_carry(batch=batch)  # type: ignore[attr-defined]
        outputs: Dict[str, torch.Tensor] = {}

        max_steps = getattr(getattr(model, "model", None), "config", None)
        max_steps = getattr(max_steps, "halt_max_steps", 64)

        steps = 0
        while True:
            carry, _loss, _metrics, preds, all_finish = model(  # type: ignore[operator]
                carry=carry,
                batch=batch,
                return_keys=("logits", "preds", "q_halt_logits"),
            )
            outputs = preds
            steps += 1
            if bool(all_finish):
                break
            if steps >= max_steps:
                break

    return {k: v.detach().cpu() for k, v in outputs.items()}


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Solve a 3-SAT instance using a trained TRM.")
    parser.add_argument("--config", default="config/arch/trm.yaml", help="Model architecture config.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint.")
    parser.add_argument(
        "--device",
        default=None,
        help="Preferred device (e.g. 'cuda', 'cuda:0', or 'cpu'). Defaults to CUDA if available.",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    num_vars, clauses = parse_dimacs(sys.stdin)
    tokens, literal_positions, clause_literals, clause_positions = encode_problem(clauses)

    inputs = torch.from_numpy(tokens).unsqueeze(0).to(torch.int32)
    labels = torch.full((1, SEQ_LEN), IGNORE_LABEL_ID, dtype=torch.int32)
    puzzle_ids = torch.zeros((1,), dtype=torch.int32)

    model = build_model(args.config, args.checkpoint, device)
    outputs = run_inference(
        model,
        batch={"inputs": inputs, "labels": labels, "puzzle_identifiers": puzzle_ids},
        device=device,
    )

    logits = outputs["logits"][0].to(torch.float32).numpy()
    pred_tokens = outputs["preds"][0].numpy().astype(np.int32)

    assignment = decode_assignment_from_logits(
        logits,
        literal_positions,
        clause_positions,
        clause_literals,
        num_vars,
        pred_tokens,
    )

    assignment.extend(var for var in range(len(assignment) + 1, num_vars + 1))

    print("v", *assignment, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

