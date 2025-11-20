"""Utility functions for working with DIMACS CNF and solution files."""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def parse_cnf(content: str) -> Tuple[int, List[List[int]]]:
    """Parse a DIMACS CNF file content into ``(num_vars, clauses)``."""
    num_vars: Optional[int] = None
    clauses: List[List[int]] = []
    current_clause: List[int] = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue

        if line.startswith("p"):
            parts = line.split()
            if len(parts) < 4 or parts[1].lower() != "cnf":
                raise ValueError("Invalid CNF header (expected 'p cnf <vars> <clauses>').")
            num_vars = int(parts[2])
            continue

        for token in line.split():
            lit = int(token)
            if lit == 0:
                if not current_clause:
                    raise ValueError("Clause terminator encountered without literals.")
                clauses.append(current_clause)
                current_clause = []
            else:
                current_clause.append(lit)

    if current_clause:
        raise ValueError("Missing clause terminator at end of CNF file.")
    if num_vars is None:
        raise ValueError("CNF header not found.")

    return num_vars, clauses


def parse_solution(content: str) -> Optional[List[int]]:
    """Parse a DIMACS solution file and return a sequence of literals.

    Returns a list of non-zero literals (e.g. ``[1, -2, 3]``). If the
    solution declares the instance UNSAT or no assignment is given, returns
    ``None``.
    """
    literals: List[int] = []
    declared_unsat = False

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("c"):
            continue

        if stripped.startswith("s"):
            parts = stripped.split()
            if len(parts) >= 2 and parts[1].lower() == "unsatisfiable":
                declared_unsat = True
            continue

        tokens: Iterable[str]
        if stripped.startswith("v"):
            tokens = stripped.split()[1:]
        else:
            tokens = stripped.split()

        for token in tokens:
            lit = int(token)
            if lit == 0:
                continue
            literals.append(lit)

    if declared_unsat:
        return None
    if not literals:
        return None
    return literals


def serialize_cnf(num_vars: int, clauses: Sequence[Sequence[int]]) -> str:
    """Serialize ``(num_vars, clauses)`` into DIMACS CNF format."""
    if num_vars < 0:
        raise ValueError("Number of variables must be non-negative.")

    header = f"p cnf {num_vars} {len(clauses)}"
    lines: List[str] = [header]

    for clause in clauses:
        if not clause:
            raise ValueError("Clauses must contain at least one literal.")

        serialized_literals: List[str] = []
        for lit in clause:
            if lit == 0:
                raise ValueError("Literals must be non-zero in DIMACS CNF.")
            serialized_literals.append(str(int(lit)))

        lines.append(" ".join(serialized_literals + ["0"]))

    return "\n".join(lines) + "\n"


def make_rand_3sat_simple(nvars: int, num_clauses: int) -> List[List[int]]:
    """Generate a random 3-SAT instance using Python's ``random`` module."""
    if num_clauses <= 0:
        raise ValueError("num_clauses must be positive")
    if nvars <= 0:
        raise ValueError("nvars must be positive")

    clauses: List[List[int]] = []
    for _ in range(num_clauses):
        clause: List[int] = []
        for _ in range(3):
            var = random.randint(1, nvars)
            lit = var if random.choice((True, False)) else -var
            clause.append(lit)
        clauses.append(clause)
    return clauses


def make_rand_3sat_planted(nvars: int, num_clauses: int) -> Tuple[List[List[int]], List[int]]:
    """Generate a random 3-SAT instance with a planted solution.

    Returns:
        clauses: list of 3-literal clauses (each a list of ints).
        solution: list of literals encoding the planted model (length nvars).
    """
    if num_clauses <= 0:
        raise ValueError("num_clauses must be positive")
    if nvars <= 0:
        raise ValueError("nvars must be positive")

    # Random planted assignment in literal form: +i means True, -i means False.
    solution: List[int] = []
    for v in range(1, nvars + 1):
        if random.choice((True, False)):
            solution.append(v)
        else:
            solution.append(-v)

    clauses: List[List[int]] = []
    for _ in range(num_clauses):
        # Choose three distinct variables.
        vars_chosen = random.sample(range(1, nvars + 1), k=3)

        # Start by picking literals consistent with the planted assignment,
        # then randomly flip some signs so that at least one literal remains
        # satisfied by the planted assignment.
        clause: List[int] = []
        satisfied = False
        for v in vars_chosen:
            planted_lit = solution[v - 1]
            lit = planted_lit
            # With 50% chance, flip the literal.
            if random.choice((True, False)):
                lit = -lit
            clause.append(lit)
            if lit == planted_lit:
                satisfied = True

        if not satisfied:
            # Force at least one satisfied literal by resetting one position
            # to its planted sign.
            idx = random.randrange(3)
            v = vars_chosen[idx]
            clause[idx] = solution[v - 1]

        clauses.append(clause)

    return clauses, solution


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
        cnf_file.write(serialize_cnf(max_var, clauses))
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


def cadical_is_unique(clauses: List[List[int]], solution: List[int]) -> bool:
    """Return True if ``solution`` is the unique model of ``clauses``.

    This constructs a blocking clause (the negation of every literal in
    ``solution``) and checks satisfiability of the augmented formula using
    :func:`cadical_solve`. If the augmented formula is UNSAT, then
    ``solution`` is unique.
    """
    if not solution:
        return False

    block_clause = [-lit for lit in solution if lit != 0]
    if not block_clause:
        raise ValueError("Solution must contain at least one non-zero literal.")

    alt_model = cadical_solve(clauses + [block_clause])
    return alt_model is None


def cadical_generate_unique(
    nvars: int,
    num_clauses: int,
    max_tries: int = 1000,
) -> Tuple[List[List[int]], List[int]]:
    """Generate a random 3-SAT instance with a unique model.

    Uses :func:`make_rand_3sat_planted` and :func:`cadical_is_unique`
    (which in turn uses :func:`cadical_solve`) to repeatedly sample
    formulas until one is found that is SAT with a unique solution.

    Raises:
        RuntimeError: If a unique instance is not found within ``max_tries``.
    """
    if nvars <= 0:
        raise ValueError("nvars must be positive")
    if num_clauses <= 0:
        raise ValueError("num_clauses must be positive")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive")

    for _ in range(max_tries):
        clauses, solution = make_rand_3sat_planted(nvars, num_clauses)
        if cadical_is_unique(clauses, solution):
            return clauses, solution

    raise RuntimeError("Failed to generate a unique SAT instance within max_tries.")


def clause_satisfied(clause: Sequence[int], assignment: Mapping[int, bool]) -> bool:
    """Return ``True`` if ``clause`` is satisfied under ``assignment``."""

    for lit in clause:
        var = abs(lit)
        value = assignment.get(var)
        if value is None:
            continue
        literal_truth = (lit > 0 and value) or (lit < 0 and not value)
        if literal_truth:
            return True
    return False


def evaluate_solution(
    clauses: Sequence[Sequence[int]],
    assignment: Optional[Sequence[int]],
) -> bool:
    """Return ``True`` if ``assignment`` satisfies ``clauses``.

    ``assignment`` is a sequence of literals (e.g. ``[1, -2, 3]``). If
    ``assignment`` is ``None``, this indicates no solution and returns ``False``.
    Conflicting literals for the same variable invalidate the assignment.
    """

    if assignment is None:
        return False
    if not clauses:
        return True
    if len(assignment) == 0:
        return False

    # Build a truth map from the literal sequence, rejecting conflicts.
    truth_map: Dict[int, bool] = {}
    for lit in assignment:
        if lit == 0:
            continue
        var = abs(lit)
        val = lit > 0
        prev = truth_map.get(var)
        if prev is not None and prev != val:
            return False  # inconsistent assignment
        truth_map[var] = val

    return all(clause_satisfied(clause, truth_map) for clause in clauses)


__all__ = [
    "cadical_generate_unique",
    "cadical_is_unique",
    "cadical_solve",
    "clause_satisfied",
    "evaluate_solution",
    "make_rand_3sat_planted",
    "make_rand_3sat_simple",
    "parse_cnf",
    "parse_solution",
    "serialize_cnf",
]
