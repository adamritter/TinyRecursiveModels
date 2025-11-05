import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import tempfile

import numpy as np
from pysat.solvers import Solver

from dataset.common import PuzzleDatasetMetadata

NUM_VARS = 1000
TRAIN_NUM_EXAMPLES = 25000
TEST_NUM_EXAMPLES = 100
TRAIN_NUM_VARS_USED = None
MIN_CLAUSES = int(4.26 * NUM_VARS)
MCLAUSES = int(math.ceil(5.5 * NUM_VARS))
TOKENS_PER_FORMULA = MCLAUSES * 3
SEQ_LEN = TOKENS_PER_FORMULA
VOCAB_SIZE = 2 * NUM_VARS + 1  # includes PAD token at index 0
CHUNK_SIZE = 25


@dataclass(frozen=True)
class DatasetSplitConfig:
    name: str
    num_examples: int
    seed: int
    nvars_used: Optional[int] = None
    unique: bool = True
    planted: bool = False


# Alias maintained for external callers that expect DataSplitConfig naming.
DataSplitConfig = DatasetSplitConfig


def _remap_instance_literals_and_model(
    clauses: np.ndarray,
    model: List[int],
    rng: np.random.Generator,
    nvars: int,
    base_nvars: int,
) -> Tuple[np.ndarray, List[int]]:
    """Remap a single clause array and its model onto randomly chosen global variables."""
    if clauses.size == 0:
        return clauses, model

    mapping = np.empty(base_nvars + 1, dtype=np.int32)
    mapping[0] = 0
    mapping[1:] = rng.choice(nvars, size=base_nvars, replace=False) + 1

    abs_clauses = np.abs(clauses)
    mapped_abs = mapping[abs_clauses]
    clause_signs = np.where(clauses > 0, 1, -1)
    remapped_clauses = (mapped_abs * clause_signs).astype(np.int32, copy=False)

    if model:
        model_array = np.asarray(model, dtype=np.int32)
        zero_mask = model_array == 0
        model_abs = np.abs(model_array)
        mapped_model_abs = mapping[model_abs]
        model_signs = np.where(model_array > 0, 1, -1)
        remapped_model_array = mapped_model_abs * model_signs
        remapped_model_array[zero_mask] = 0
        remapped_model = remapped_model_array.tolist()
    else:
        remapped_model = model

    return remapped_clauses, remapped_model


def make_rand_3sat(
    nvars: int,
    num_clauses: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Generate a single random 3-SAT instance expressed as a clause list."""
    if num_clauses <= 0:
        raise ValueError("num_clauses must be positive")

    literals = rng.integers(0, 2 * nvars, size=(num_clauses, 3), dtype=np.int32)
    literals -= nvars
    literals += 1 + (literals >> 31)

    return literals.tolist()


def make_planted_rand_3sat(
    nvars: int,
    rng: np.random.Generator,
    alpha: float = 5.4,
) -> Tuple[List[List[int]], List[int]]:
    """Generate a random 3-SAT instance with a planted solution, returning clauses and the solution."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    planted_assignment = rng.choice([-1, 1], size=nvars).astype(np.int32)
    total_clauses = max(1, int(alpha * nvars))

    clauses: List[List[int]] = []
    for _ in range(total_clauses):
        vars_sample = rng.choice(nvars, size=3, replace=False)
        clause: List[int] = []
        satisfied = False
        for var_idx in vars_sample:
            var = var_idx + 1
            sign = planted_assignment[var_idx]
            # Flip sign with probability 0.5 to allow both satisfied and unsatisfied literals.
            if rng.random() < 0.5:
                sign = -sign
            literal = var if sign > 0 else -var
            clause.append(literal)
            if planted_assignment[var_idx] == (1 if literal > 0 else -1):
                satisfied = True

        if not satisfied:
            flip_idx = rng.integers(3)
            var_idx = vars_sample[flip_idx]
            clause[flip_idx] = vars_sample[flip_idx] + 1 if planted_assignment[var_idx] > 0 else -(vars_sample[flip_idx] + 1)

        clauses.append(clause)

    planted_model = [idx if sign > 0 else -idx for idx, sign in enumerate(planted_assignment, start=1)]

    return clauses, planted_model


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


def token_to_literal(token: int) -> int:
    """Inverse of encode_clauses for a single token."""
    if token == 0:
        raise ValueError("Token 0 does not map to a literal.")
    if 1 <= token <= NUM_VARS:
        return token
    if NUM_VARS < token <= 2 * NUM_VARS:
        return -(token - NUM_VARS)
    raise ValueError(f"Token {token} out of expected range for negative literal.")


def _model_to_assignment(model: List[int]) -> Dict[int, int]:
    assignment: Dict[int, int] = {}
    for lit in model:
        if lit == 0:
            continue
        assignment[abs(lit)] = lit
    return assignment


def find_any_model(clauses: List[List[int]]) -> Optional[List[int]]:
    """Return any satisfying assignment for the clause set, or None if UNSAT."""
    solver = Solver(bootstrap_with=clauses)
    try:
        if not solver.solve():
            return None
        model = solver.get_model()
        return model if model is not None else None
    finally:
        solver.delete()


def cadical_solve(clauses: List[List[int]]) -> Optional[List[int]]:
    """Solve CNF via external cadical; return a model or None if UNSAT."""
    if not clauses:
        return []

    max_var = 0
    for clause in clauses:
        for lit in clause:
            max_var = max(max_var, abs(lit))

    if max_var == 0:
        return []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as cnf_file:
        cnf_path = cnf_file.name
        cnf_file.write(f"p cnf {max_var} {len(clauses)}\n")
        for clause in clauses:
            cnf_file.write(" ".join(str(lit) for lit in clause) + " 0\n")
        cnf_file.flush()

    try:
        try:
            result = subprocess.run(
                ["cadical", "-q", cnf_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except FileNotFoundError as err:
            raise RuntimeError("cadical binary not found in PATH") from err

        if result.returncode not in (10, 20):
            raise RuntimeError(
                f"cadical returned unexpected code {result.returncode}: {result.stderr.strip()}"
            )

        if result.returncode == 20 or "UNSAT" in result.stdout:
            return None

        model: List[int] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line[0] != "v":
                continue
            for token in line.split()[1:]:
                if token == "0":
                    continue
                model.append(int(token))

        return model if model or result.returncode == 10 else None
    finally:
        try:
            os.remove(cnf_path)
        except OSError:
            pass


def find_unique_model(clauses: List[List[int]]) -> Optional[List[int]]:
    """Mutate clauses to enforce a unique satisfying assignment; return the model or None."""
    while True:
        model = cadical_solve(clauses)
        if model is None:
            return None

        block_clause = [-lit for lit in model if lit != 0]
        alt_model = cadical_solve(clauses + [block_clause])
        if alt_model is None:
            return model

        base_assignment = _model_to_assignment(model)
        alt_assignment = _model_to_assignment(alt_model)
        differing_vars = [var for var, lit in base_assignment.items() if alt_assignment.get(var) != lit]
        if not differing_vars:
            return None

        chosen_lit = base_assignment[differing_vars[0]]
        # Repeat literal to enforce the unit constraint while keeping 3-SAT clause shape.
        clauses.append([chosen_lit, chosen_lit, chosen_lit])


def _generate_chunk(
    seed: np.random.SeedSequence,
    target_examples: int,
    num_clauses: int,
    split_config: DataSplitConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """Generate a fixed number of SAT examples with a dedicated RNG."""
    rng = np.random.default_rng(seed)
    inputs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sat_count = 0
    unsat_count = 0

    nvars_used = split_config.nvars_used
    unique = split_config.unique
    planted = split_config.planted

    nvars = NUM_VARS if nvars_used is None else nvars_used
    num_clauses = int(4.26 * nvars)

    while len(inputs) < target_examples:
        if planted:
            clauses_list, model = make_planted_rand_3sat(nvars, rng)
        else:
            clauses_list = make_rand_3sat(nvars, num_clauses, rng)

            if unique:
                model = find_unique_model(clauses_list)
            else:
                model = cadical_solve(clauses_list)

        if model is None or len(clauses_list) > MCLAUSES:
            unsat_count += 1
            continue

        sat_count += 1
        clauses_arr = np.array(clauses_list, dtype=np.int32)

        if nvars_used is not None and nvars_used < NUM_VARS:
            clauses_arr, model = _remap_instance_literals_and_model(
                clauses_arr,
                model,
                rng,
                NUM_VARS,
                nvars_used,
            )

        tokens = encode_clauses(clauses_arr)
        labels.append(encode_satisfied_literals(clauses_arr, tokens, model))
        inputs.append(tokens)
        if len(inputs) >= target_examples:
            break

    return inputs, labels, sat_count, unsat_count


def _print_progress(previous: int, current: int) -> None:
    """Emit progress updates when reaching 1k SAT milestones."""
    start = previous // 1000
    end = current // 1000
    for milestone in range(start + 1, end + 1):
        print(f"Generated {milestone * 1000} SAT examples...")


def generate_examples(
    split_config: DataSplitConfig,
    num_workers: Optional[int] = None,
    chunk_size: int = CHUNK_SIZE,
    num_clauses: int = MIN_CLAUSES,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Generate SAT dataset examples using multi-process workers."""
    num_examples = split_config.num_examples
    seed = split_config.seed

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
            handle_chunk(_generate_chunk(seed_item, target, num_clauses, split_config))
    else:
        chunk_iter = iter(zip(worker_seeds, chunk_sizes))

        def submit_next(pending: Dict) -> None:
            try:
                seed_item, target = next(chunk_iter)
            except StopIteration:
                return
            future = executor.submit(
                _generate_chunk,
                seed_item,
                target,
                num_clauses,
                split_config,
            )
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

    triplet_rng = np.random.default_rng(permutation_seed)
    total_triplet_slots = SEQ_LEN // 3
    for idx, (example, label) in enumerate(zip(inputs, labels)):
        example_len = example.shape[0]
        label_len = label.shape[0]
        if label_len != example_len:
            raise ValueError("Input and label lengths must match.")

        if example_len and example_len % 3 != 0:
            raise ValueError("Example length must be divisible by 3.")

        if example_len:
            clause_count = example_len // 3
            if clause_count > total_triplet_slots:
                raise ValueError("Clause count exceeds available triplet slots.")

            clause_perm = triplet_rng.permutation(total_triplet_slots)[:clause_count]
            example_triplets = example.reshape(clause_count, 3)
            label_triplets = label.reshape(clause_count, 3)

            inputs_arr[idx].reshape(total_triplet_slots, 3)[clause_perm] = example_triplets
            labels_arr[idx].reshape(total_triplet_slots, 3)[clause_perm] = label_triplets

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
        pad_id=-1,
        ignore_label_id=-1,
        blank_identifier_id=-1,
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


def _decode_tokens_to_clauses(tokens: np.ndarray) -> List[List[int]]:
    """Convert flat token sequence into clause lists."""
    nonzero = tokens[tokens != 0]
    if nonzero.size % 3 != 0:
        raise ValueError("Token sequence length not divisible by 3.")
    triplets = nonzero.reshape(-1, 3)
    clauses: List[List[int]] = []
    for triplet in triplets:
        clause: List[int] = []
        for token in triplet:
            if 1 <= token <= NUM_VARS:
                clause.append(int(token))
            elif NUM_VARS < token <= 2 * NUM_VARS:
                clause.append(-int(token - NUM_VARS))
            else:
                raise ValueError(f"Token {token} out of expected range.")
        clauses.append(clause)
    return clauses


def _print_problem(split: str, index: int) -> None:
    root = Path("sat_examples")
    inputs_path = root / split / "all__inputs.npy"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Inputs file not found at {inputs_path}")

    inputs = np.load(inputs_path)
    if index < 0 or index >= inputs.shape[0]:
        raise IndexError(f"Problem index {index} out of range (0..{inputs.shape[0]-1}).")

    tokens = inputs[index]
    clauses = _decode_tokens_to_clauses(tokens)

    if clauses:
        num_vars = max(abs(lit) for clause in clauses for lit in clause)
    else:
        num_vars = NUM_VARS

    print(f"p cnf {num_vars} {len(clauses)}")
    for clause in clauses:
        print(" ".join(str(lit) for lit in clause), "0")


def _print_solution(split: str, index: int) -> None:
    root = Path("sat_examples")
    inputs_path = root / split / "all__inputs.npy"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Inputs file not found at {inputs_path}")

    labels_path = root / split / "all__labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

    inputs = np.load(inputs_path)
    labels = np.load(labels_path)
    if index < 0 or index >= inputs.shape[0]:
        raise IndexError(f"Problem index {index} out of range (0..{inputs.shape[0]-1}).")

    tokens = inputs[index]
    label_tokens = labels[index]
    clauses = _decode_tokens_to_clauses(tokens)

    assignment: Dict[int, int] = {}
    for token in label_tokens:
        if token == 0:
            continue
        lit = token_to_literal(token)
        assignment[abs(lit)] = lit

    if clauses:
        max_var = max(max(abs(lit) for lit in clause) for clause in clauses)
    else:
        max_var = max(assignment.keys(), default=0)

    print("s SATISFIABLE")
    if max_var == 0:
        print("v 0")
        return

    literals: List[int] = []
    for var in range(1, max_var + 1):
        value = assignment.get(var)
        if value is None:
            # Default to positive assignment if solver omitted the variable.
            value = var
        literals.append(value)
    print("v", *literals, 0)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate SAT datasets or inspect stored problems.")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--problem",
        nargs=2,
        metavar=("SPLIT", "INDEX"),
        help="Output the specified problem from the saved dataset in DIMACS CNF format.",
    )
    action_group.add_argument(
        "--solution",
        nargs=2,
        metavar=("SPLIT", "INDEX"),
        help="Output a satisfying assignment for the specified problem in DIMACS solution format.",
    )
    args = parser.parse_args(argv)

    if args.problem:
        split, index_str = args.problem
        _print_problem(split, int(index_str))
        return
    if args.solution:
        split, index_str = args.solution
        _print_solution(split, int(index_str))
        return

    output_root = Path("sat_examples")
    splits = [
        DatasetSplitConfig(
            name="train",
            num_examples=TRAIN_NUM_EXAMPLES,
            seed=17,
            nvars_used=TRAIN_NUM_VARS_USED,
            planted=True,
        ),
        DatasetSplitConfig(name="test", num_examples=TEST_NUM_EXAMPLES, seed=23, unique=False, planted=True),
    ]

    for split in splits:
        data, sat_count, unsat_count = generate_examples(split)
        save_dataset(output_root, split, data)
        print(
            f"Wrote {split.num_examples} {split.name} examples to {output_root / split.name} "
            f"(sat={sat_count}, unsat={unsat_count})"
        )


if __name__ == "__main__":
    main()
