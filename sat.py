import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pysat.solvers import Solver

from dataset.common import PuzzleDatasetMetadata


NUM_VARS = 100
MCLAUSES = int(4.26 * NUM_VARS)
TOKENS_PER_FORMULA = MCLAUSES * 3
SEQ_LEN = TOKENS_PER_FORMULA
VOCAB_SIZE = 2 * NUM_VARS + 1  # includes PAD token at index 0


@dataclass(frozen=True)
class DatasetSplitConfig:
    name: str
    num_examples: int
    seed: int


def make_rand_3sat(nvars: int, max_clauses: int, batch_size: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Generate batches of random 3-SAT clause arrays with varied clause counts."""
    clause_counts = rng.integers(1, max_clauses + 1, size=batch_size, dtype=np.int32)
    max_batch_clauses = int(clause_counts.max())

    literals = rng.integers(0, 2 * nvars, size=(batch_size, max_batch_clauses, 3), dtype=np.int32)
    literals -= nvars
    literals += 1 + (literals >> 31)

    return [literals[idx, :count].copy() for idx, count in enumerate(clause_counts)]


def encode_clauses(clauses: np.ndarray) -> np.ndarray:
    """Flatten clauses into token sequence."""
    flat = clauses.reshape(-1).astype(np.int32)
    tokens = np.where(flat > 0, flat, NUM_VARS - flat).astype(np.int32)
    return tokens


def encode_satisfied_literals(clauses: np.ndarray, tokens: np.ndarray, model: List[int]) -> np.ndarray:
    """Return tokens for literals satisfied by the solver assignment; unsatisfied entries are zero."""
    assignment = np.zeros(NUM_VARS + 1, dtype=np.bool_)
    for lit in model:
        if lit == 0:
            continue
        var = abs(lit)
        if var <= NUM_VARS:
            assignment[var] = lit > 0

    flat_clauses = clauses.reshape(-1)
    satisfied_mask = np.zeros(flat_clauses.size, dtype=np.bool_)
    for idx, lit in enumerate(flat_clauses):
        var = abs(lit)
        if var == 0:
            continue
        value = assignment[var]
        satisfied_mask[idx] = value if lit > 0 else (not value)

    labels = np.zeros_like(tokens, dtype=np.int32)
    labels[satisfied_mask] = tokens[satisfied_mask]
    return labels


def _generate_chunk(seed: np.random.SeedSequence, target_examples: int, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """Generate a fixed number of SAT examples with a dedicated RNG."""
    rng = np.random.default_rng(seed)
    inputs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sat_count = 0
    unsat_count = 0

    while len(inputs) < target_examples:
        batch_target = max(batch_size, target_examples - len(inputs))
        batch = make_rand_3sat(NUM_VARS, MCLAUSES, batch_target, rng)
        for clauses in batch:
            if len(inputs) >= target_examples:
                break

            tokens = encode_clauses(clauses)

            with Solver(bootstrap_with=clauses.tolist()) as solver:
                if not solver.solve():
                    unsat_count += 1
                    continue

                model = solver.get_model()
                if model is None:
                    unsat_count += 1
                    continue

                solver.add_clause([-lit for lit in model if lit != 0])
                if solver.solve():
                    # More than one satisfying assignment; skip to enforce uniqueness.
                    unsat_count += 1
                    continue

                sat_count += 1
                labels.append(encode_satisfied_literals(clauses, tokens, model))
                inputs.append(tokens)

    return inputs, labels, sat_count, unsat_count


def _print_progress(previous: int, current: int) -> None:
    """Emit progress updates when reaching 1k SAT milestones."""
    start = previous // 1000
    end = current // 1000
    for milestone in range(start + 1, end + 1):
        print(f"Generated {milestone * 1000} SAT examples...")


def generate_examples(
    num_examples: int,
    seed: int,
    num_workers: Optional[int] = None,
    chunk_size: int = 512,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Generate SAT dataset examples using multi-process workers."""
    if num_examples <= 0:
        empty = np.zeros((0, SEQ_LEN), dtype=np.int32)
        data = {
            "inputs": empty,
            "labels": empty,
            "puzzle_identifiers": empty,
            "puzzle_indices": np.arange(1, dtype=np.int32),
            "group_indices": np.arange(1, dtype=np.int32),
        }
        return data, 0, 0

    worker_count = num_workers or (os.cpu_count() or 1)
    worker_count = max(1, worker_count)

    chunk_size = max(1, min(chunk_size, num_examples))
    num_chunks = math.ceil(num_examples / chunk_size)
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(num_chunks + 1)
    worker_seeds = child_seeds[:-1]
    permutation_seed = child_seeds[-1]

    chunk_sizes = [chunk_size] * num_chunks
    remainder = num_examples - chunk_size * (num_chunks - 1)
    if num_chunks > 0:
        chunk_sizes[-1] = remainder

    inputs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sat_count = 0
    unsat_count = 0
    batch_size = 256

    def handle_chunk(result: Tuple[List[np.ndarray], List[np.ndarray], int, int]) -> None:
        nonlocal inputs, labels, sat_count, unsat_count
        chunk_inputs, chunk_labels, chunk_sat, chunk_unsat = result
        previous_sat = sat_count
        sat_count += chunk_sat
        unsat_count += chunk_unsat
        _print_progress(previous_sat, sat_count)
        inputs.extend(chunk_inputs)
        labels.extend(chunk_labels)

    max_workers = min(worker_count, num_chunks)

    if max_workers <= 1:
        for seed_item, target in zip(worker_seeds, chunk_sizes):
            handle_chunk(_generate_chunk(seed_item, target, batch_size))
    else:
        chunk_iter = iter(zip(worker_seeds, chunk_sizes))

        def submit_next(pending: Dict) -> None:
            try:
                seed_item, target = next(chunk_iter)
            except StopIteration:
                return
            future = executor.submit(_generate_chunk, seed_item, target, batch_size)
            pending[future] = None

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending: Dict = {}
            for _ in range(max_workers):
                submit_next(pending)

            while pending:
                for future in as_completed(pending):
                    pending.pop(future)
                    handle_chunk(future.result())
                    submit_next(pending)
                    break

    if len(inputs) > num_examples:
        inputs = inputs[:num_examples]
        labels = labels[:num_examples]

    pad_length = SEQ_LEN
    if any(example.shape[0] > pad_length for example in inputs):
        raise ValueError("Encountered sequence longer than configured SEQ_LEN.")

    inputs_arr = np.zeros((len(inputs), pad_length), dtype=np.int32)
    labels_arr = np.zeros((len(labels), pad_length), dtype=np.int32)

    for idx, example in enumerate(inputs):
        inputs_arr[idx, : example.shape[0]] = example

    for idx, label in enumerate(labels):
        labels_arr[idx, : label.shape[0]] = label

    puzzle_identifiers_arr = np.zeros((len(inputs),), dtype=np.int32)

    num_examples = inputs_arr.shape[0]
    permutation = np.random.default_rng(permutation_seed).permutation(num_examples)

    inputs_arr = inputs_arr[permutation]
    labels_arr = labels_arr[permutation]
    puzzle_identifiers_arr = puzzle_identifiers_arr[permutation]

    data = {
        "inputs": inputs_arr,
        "labels": labels_arr,
        "puzzle_identifiers": puzzle_identifiers_arr,
        "puzzle_indices": np.arange(num_examples + 1, dtype=np.int32),
        "group_indices": np.arange(num_examples + 1, dtype=np.int32),
    }
    return data, sat_count, unsat_count


def save_dataset(root: Path, split: DatasetSplitConfig, data: Dict[str, np.ndarray]) -> None:
    split_dir = root / split.name
    split_dir.mkdir(parents=True, exist_ok=True)

    for field, array in data.items():
        np.save(split_dir / f"all__{field}.npy", array)

    num_examples = int(data["inputs"].shape[0])
    metadata = PuzzleDatasetMetadata(
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=num_examples,
        mean_puzzle_examples=1.0,
        total_puzzles=num_examples,
        sets=["all"],
    )

    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata.model_dump(), f)

    identifiers_path = root / "identifiers.json"
    if not identifiers_path.exists():
        with open(identifiers_path, "w") as f:
            json.dump(["3sat"], f)


def main() -> None:
    output_root = Path("sat_examples")
    splits = [
        DatasetSplitConfig(name="train", num_examples=50000, seed=17),
        DatasetSplitConfig(name="test", num_examples=20000, seed=23),
    ]

    for split in splits:
        data, sat_count, unsat_count = generate_examples(split.num_examples, split.seed)
        save_dataset(output_root, split, data)
        print(
            f"Wrote {split.num_examples} {split.name} examples to {output_root / split.name} "
            f"(sat={sat_count}, unsat={unsat_count})"
        )


if __name__ == "__main__":
    main()
