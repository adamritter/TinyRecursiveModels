#!/usr/bin/env python3
"""
Verify that a DIMACS CNF file is satisfied by a DIMACS solution file.

Outputs "correct" if every clause is satisfied by the assignment provided in
the solution file; otherwise outputs "incorrect".
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from sat_utils import parse_cnf, parse_solution


def evaluate(clauses: List[List[int]], literals: Optional[List[int]]) -> bool:
    """Return True if the assignment satisfies the clauses."""

    if literals is None:
        return False

    assignment: Dict[int, bool] = {}
    for lit in literals:
        var = abs(lit)
        value = lit > 0
        if var in assignment and assignment[var] != value:
            return False
        assignment[var] = value

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
        literals = parse_solution(solution_content)
        ok = evaluate(clauses, literals)
    except Exception:
        ok = False

    print("correct" if ok else "incorrect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

