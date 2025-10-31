import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from pysat.solvers import Solver

from dataset.common import PuzzleDatasetMetadata


NUM_VARS = 5
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


def generate_examples(num_examples: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    inputs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    puzzle_identifiers = []

    batch_size = 256

    sat_count = 0
    unsat_count = 0

    while len(inputs) < num_examples:
        batch = make_rand_3sat(NUM_VARS, MCLAUSES, max(batch_size, num_examples - len(inputs)), rng)
        for clauses in batch:
            if len(inputs) >= num_examples:
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
                puzzle_identifiers.append(0)
    pad_length = SEQ_LEN
    if any(example.shape[0] > pad_length for example in inputs):
        raise ValueError("Encountered sequence longer than configured SEQ_LEN.")

    inputs_arr = np.zeros((len(inputs), pad_length), dtype=np.int32)
    labels_arr = np.zeros((len(labels), pad_length), dtype=np.int32)

    for idx, example in enumerate(inputs):
        inputs_arr[idx, : example.shape[0]] = example

    for idx, label in enumerate(labels):
        labels_arr[idx, : label.shape[0]] = label
    puzzle_identifiers_arr = np.array(puzzle_identifiers, dtype=np.int32)

    num_examples = inputs_arr.shape[0]
    permutation = rng.permutation(num_examples)

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
        DatasetSplitConfig(name="train", num_examples=2000000, seed=17),
        DatasetSplitConfig(name="test", num_examples=50000, seed=23),
    ]

    for split in splits:
        rng = np.random.default_rng(split.seed)
        data, sat_count, unsat_count = generate_examples(split.num_examples, rng)
        save_dataset(output_root, split, data)
        print(
            f"Wrote {split.num_examples} {split.name} examples to {output_root / split.name} "
            f"(sat={sat_count}, unsat={unsat_count})"
        )


if __name__ == "__main__":
    main()
