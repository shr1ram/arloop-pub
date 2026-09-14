"""Code normalisation and error signatures."""
from __future__ import annotations

from arloop.taxonomy import error_signature, normalize_code


def test_renaming_a_variable_leaves_the_fingerprint_unchanged():
    a = normalize_code("def f(x):\n    total = x + 1\n    return total\n")
    b = normalize_code("def g(y):\n    acc = y + 1\n    return acc\n")
    assert a.fingerprint == b.fingerprint
    assert a.mode == "ast"


def test_comments_and_formatting_do_not_change_the_fingerprint():
    a = normalize_code("x = 1  # a comment\ny = x + 2\n")
    b = normalize_code("x   =  1\n\n\ny = x+2\n")
    assert a.fingerprint == b.fingerprint


def test_a_real_edit_changes_the_fingerprint():
    a = normalize_code("x = 1\ny = x + 2\n")
    b = normalize_code("x = 1\ny = x * 2\n")
    assert a.fingerprint != b.fingerprint


def test_block_structure_survives_normalisation():
    a = normalize_code("if x:\n    a = 1\n")
    b = normalize_code("if x:\n    a = 1\n    b = 2\n")
    assert a.fingerprint != b.fingerprint


def test_keywords_builtins_and_attributes_are_not_renamed():
    a = normalize_code("xs = []\nxs.append(1)\n")
    b = normalize_code("ys = []\nys.extend(1)\n")
    assert a.fingerprint != b.fingerprint
    assert "append" in a.tokens and "len" not in a.tokens


def test_unparseable_source_falls_back_to_lines():
    form = normalize_code("def f(:\n  ???not python???\n")
    assert form.mode == "lines"
    assert form.tokens == ("def f(:", "???not python???")


def test_embedded_solver_code_is_expanded_not_one_token():
    inner_a = "a = 1\nb = a + 1\nprint(b)\n" + "# pad\n" * 40
    inner_b = "a = 1\nb = a * 9\nprint(b)\n" + "# pad\n" * 40
    a = normalize_code(f"SOLVER_CODE = r'''{inner_a}'''\n")
    b = normalize_code(f"SOLVER_CODE = r'''{inner_b}'''\n")
    assert len(inner_a) >= 200
    assert a.fingerprint != b.fingerprint
    assert "<str>" in a.tokens


def test_error_signature_is_stable_across_paths_lines_and_numbers():
    one = ('Traceback (most recent call last):\n'
           '  File "/tmp/run-1/solution.py", line 12, in <module>\n'
           'ValueError: bad value 7\n')
    two = ('Traceback (most recent call last):\n'
           '  File "/scratch/run-2/solution.py", line 98, in <module>\n'
           'ValueError: bad value 4210\n')
    assert error_signature(one, 1)[0] == error_signature(two, 1)[0]


def test_different_failure_modes_get_different_signatures():
    a = error_signature("ValueError: bad value", 1)
    b = error_signature("KeyError: missing", 1)
    assert a[0] != b[0]


def test_empty_error_text_records_the_exit_code():
    h, norm = error_signature("", 137)
    assert norm == "<no-error-text:exit=137>"
    assert h == error_signature(None, 137)[0]


def test_signature_uses_only_the_last_lines():
    tail = "".join(f"tail {i}\n" for i in range(6))
    a = error_signature("noise\n" * 20 + tail, 1)
    b = error_signature("other noise\n" * 3 + tail, 1)
    assert a[0] == b[0]
    assert a[1] == "tail <NUM> | tail <NUM> | tail <NUM> | tail <NUM> | " \
                   "tail <NUM> | tail <NUM>"
