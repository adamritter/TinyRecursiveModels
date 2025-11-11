"""Utility functions for working with DIMACS CNF and solution files."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple


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
    """Parse a DIMACS solution file content into a list of assignments.

    The returned list contains the literals that form the assignment (e.g. ``1``
    means variable 1 is ``True`` while ``-2`` means variable 2 is ``False``).
    If the solver reports the instance as unsatisfiable, ``None`` is returned.
    """

    status: Optional[str] = None
    assignments: List[int] = []

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
            assignments.append(lit)

    if status == "unsatisfiable":
        return None

    return assignments


__all__ = ["parse_cnf", "parse_solution"]

