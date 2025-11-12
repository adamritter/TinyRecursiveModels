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

from sat_utils import evaluate_solution, parse_cnf, parse_solution


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
        ok = evaluate_solution(clauses, assignment, status, consistent)
    except Exception:
        ok = False

    print("correct" if ok else "incorrect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

