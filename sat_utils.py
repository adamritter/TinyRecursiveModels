"""Utility functions for working with DIMACS CNF and solution files."""

from __future__ import annotations

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


def parse_solution(content: str) -> Tuple[Dict[int, bool], Optional[str], bool]:
    """Parse a DIMACS solution file content into ``(assignment, status, consistent)``."""
    assignment: Dict[int, bool] = {}
    status: Optional[str] = None
    consistent = True

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("c"):
            continue

        if stripped.startswith("s"):
            parts = stripped.split()
            if len(parts) >= 2:
                status = parts[1].lower()
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

            var = abs(lit)
            value = lit > 0

            if var in assignment and assignment[var] != value:
                consistent = False
            assignment[var] = value

    return assignment, status, consistent


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
    assignment: Mapping[int, bool],
    status: Optional[str],
    consistent: bool,
) -> bool:
    """Return ``True`` if the parsed solution satisfies ``clauses``."""

    if not consistent:
        return False
    if status == "unsatisfiable":
        return False
    if not clauses:
        return True
    if not assignment:
        return False

    return all(clause_satisfied(clause, assignment) for clause in clauses)


__all__ = [
    "clause_satisfied",
    "evaluate_solution",
    "parse_cnf",
    "parse_solution",
    "serialize_cnf",
]

