from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from sat_utils import parse_cnf, serialize_cnf


def test_serialize_cnf_roundtrip() -> None:
    num_vars = 3
    clauses = [[1, -2, 3], [-1]]

    serialized = serialize_cnf(num_vars, clauses)
    parsed_num_vars, parsed_clauses = parse_cnf(serialized)

    assert parsed_num_vars == num_vars
    assert parsed_clauses == [clause[:] for clause in clauses]


def test_serialize_cnf_rejects_invalid_literals() -> None:
    with pytest.raises(ValueError, match="Literals must be non-zero"):
        serialize_cnf(1, [[0]])
