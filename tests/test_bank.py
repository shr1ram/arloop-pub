"""Bank round trip and bank-id determinism."""
from arloop.memory.bank import bank_id_for, build_bank, load_bank
from arloop.memory.cases import make_case

PROV = [{"run_id": "r1", "seq_span": [0, 9]}]


def test_build_load_round_trip(tmp_path):
    cases = [make_case("first lesson", "ahc001", PROV, {"retrieval_context": "ctx"}),
             make_case("second lesson", "ahc002", PROV)]
    derivation = {"corpus_hash": "abc", "source_cells": ["r0-w100-x0"]}
    built = build_bank(tmp_path / "bank", cases, derivation)
    loaded = load_bank(tmp_path / "bank")

    assert loaded.bank_id == built.bank_id
    assert [c.to_dict() for c in loaded.cases] == [c.to_dict() for c in cases]
    assert loaded.cases[0].task_id == "ahc001"
    assert loaded.cases[0].meta["retrieval_context"] == "ctx"
    assert loaded.cases[0].meta["chars"] == len("first lesson")
    assert loaded.manifest["n_cases"] == 2
    assert loaded.manifest["total_chars"] == sum(c.chars for c in cases)
    assert loaded.dir == tmp_path / "bank"
    assert set(loaded.by_id) == {c.id for c in cases}


def test_bank_id_is_derivation_addressed():
    a = bank_id_for({"corpus_hash": "x", "n": 1})
    assert a == bank_id_for({"n": 1, "corpus_hash": "x"})   # key order is not identity
    assert a != bank_id_for({"corpus_hash": "y", "n": 1})
    assert a.startswith("bank-") and len(a) == len("bank-") + 12


def test_case_id_is_content_addressed():
    a = make_case("same", "t", PROV)
    b = make_case("same", "t", PROV)
    c = make_case("other", "t", PROV)
    assert a.id == b.id and a.id != c.id
    # the anchor task is not part of the id; identical derivation, identical id
    assert make_case("same", "other-task", PROV).id == a.id
