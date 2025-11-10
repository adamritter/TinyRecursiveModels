#!/usr/bin/env python3
"""
Verify that a DIMACS CNF file is satisfied by a DIMACS solution file.

Outputs "correct" if every clause is satisfied by the assignment provided in
the solution file; otherwise outputs "incorrect".
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_cnf(content: str) -> Tuple[int, List[List[int]]]:
    """Parse a DIMACS CNF file content into (num_vars, clauses)."""
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
    """Parse a DIMACS solution file content into (assignment, status, consistent)."""
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


def evaluate(
    clauses: List[List[int]],
    assignment: Dict[int, bool],
    status: Optional[str],
    consistent: bool,
) -> bool:
    """Return True if the assignment satisfies the clauses."""
    if not consistent:
        return False
    if status == "unsatisfiable":
        # Claimed UNSAT but an assignment was provided.
        return False
    if not clauses:
        return True
    if not assignment:
        return False

    for clause in clauses:
        clause_satisfied = False
        for lit in clause:
            var = abs(lit)
            if var not in assignment:
                continue
            value = assignment[var]
            literal_truth = (lit > 0 and value) or (lit < 0 and not value)
            if literal_truth:
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a DIMACS CNF solution.")
    parser.add_argument("cnf", help="Path to the DIMACS CNF file.")
    parser.add_argument("solution", help="Path to the DIMACS solution file.")
    args = parser.parse_args()

    try:
        cnf_content = Path(args.cnf).read_text(encoding="utf-8")
        solution_content = Path(args.solution).read_text(encoding="utf-8")
        _num_vars, clauses = parse_cnf(cnf_content)
        assignment, status, consistent = parse_solution(solution_content)
        ok = evaluate(clauses, assignment, status, consistent)
    except Exception:
        ok = False

    print("correct" if ok else "incorrect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

