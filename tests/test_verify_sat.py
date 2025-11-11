"""Unit tests for :mod:`verify_sat`."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from sat_utils import parse_cnf, parse_solution
from verify_sat import evaluate


def write_tmp_file(tmp_path: Path, name: str, content: str) -> Path:
    """Helper to create a temporary file with the provided content."""

    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_verify_sat_reports_correct_for_satisfied_instance(tmp_path: Path) -> None:
    """A consistent assignment that satisfies every clause is accepted."""

    cnf_path = write_tmp_file(
        tmp_path,
        "instance.cnf",
        """c Example CNF file
p cnf 2 2
1 -2 0
-2 0
""",
    )
    solution_path = write_tmp_file(
        tmp_path,
        "instance.sol",
        """c Example SAT solver output
s SATISFIABLE
v 1 -2 0
""",
    )

    _num_vars, clauses = parse_cnf(cnf_path.read_text(encoding="utf-8"))
    assignment, status, consistent = parse_solution(
        solution_path.read_text(encoding="utf-8")
    )

    assert clauses == [[1, -2], [-2]]
    assert assignment == {1: True, 2: False}
    assert status == "satisfiable"
    assert consistent is True
    assert evaluate(clauses, assignment, status, consistent) is True


def test_verify_sat_rejects_inconsistent_assignment(tmp_path: Path) -> None:
    """Conflicting literals in the solution render it inconsistent."""

    cnf_path = write_tmp_file(
        tmp_path,
        "conflict.cnf",
        """p cnf 1 1
1 0
""",
    )
    solution_path = write_tmp_file(
        tmp_path,
        "conflict.sol",
        """v 1 -1 0
""",
    )

    _num_vars, clauses = parse_cnf(cnf_path.read_text(encoding="utf-8"))
    assignment, status, consistent = parse_solution(
        solution_path.read_text(encoding="utf-8")
    )

    assert consistent is False
    assert evaluate(clauses, assignment, status, consistent) is False


def test_verify_sat_rejects_unsatisfiable_claim(tmp_path: Path) -> None:
    """If the solver claims UNSAT while giving an assignment, verification fails."""

    cnf_path = write_tmp_file(
        tmp_path,
        "unsat.cnf",
        """p cnf 1 1
1 0
""",
    )
    solution_path = write_tmp_file(
        tmp_path,
        "unsat.sol",
        """s UNSATISFIABLE
1 0
""",
    )

    _num_vars, clauses = parse_cnf(cnf_path.read_text(encoding="utf-8"))
    assignment, status, consistent = parse_solution(
        solution_path.read_text(encoding="utf-8")
    )

    assert status == "unsatisfiable"
    assert evaluate(clauses, assignment, status, consistent) is False


def test_parse_cnf_requires_header(tmp_path: Path) -> None:
    """The CNF parser requires the "p cnf" header to be present."""

    cnf_path = write_tmp_file(
        tmp_path,
        "missing_header.cnf",
        """1 -2 0
2 0
""",
    )

    with pytest.raises(ValueError, match="CNF header not found"):
        parse_cnf(cnf_path.read_text(encoding="utf-8"))
