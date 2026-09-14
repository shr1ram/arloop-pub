"""Retrievers and MemoryView packing. Embedding and cross-encoder models are
always injected as plain callables — no model ever loads here."""
import json

import pytest

from arloop.memory.bank import Bank, MemoryView, build_bank
from arloop.memory.cases import make_case
from arloop.memory.retrievers import (BM25Retriever, EmbeddingRetriever,
                                      HybridRerankRetriever, HybridRetriever,
                                      indexed_text)
from arloop.tokens import approx_tokens

PROV = [{"run_id": "r1", "seq_span": [0, 1]}]


def bank_of(*pairs) -> Bank:
    """Bank from (content, task_id) pairs, with no dir (no on-disk index)."""
    cases = [make_case(content, task_id, PROV) for content, task_id in pairs]
    return Bank(bank_id="bank-test", cases=cases, manifest={}, dir=None)


def fake_embed(texts, is_query):
    """Two-dim vectors: 'alpha' count and 'beta' count, so a query built from
    one word is nearest the case dominated by it."""
    return [[float(t.lower().count("alpha")), float(t.lower().count("beta"))]
            for t in texts]


def test_bm25_ranks_overlap_and_drops_zero_overlap():
    bank = bank_of(("segment tree beats range query", "t1"),
                   ("segment tree basics", "t2"),
                   ("simulated annealing schedule", "t3"))
    ranked = BM25Retriever().query("segment tree beats", bank, 10)
    assert len(ranked) == 2                       # the annealing case shares no term
    assert ranked[0][0].content.startswith("segment tree beats")
    assert all(score > 0 for _, score in ranked)


def test_bm25_returns_nothing_without_overlap():
    assert BM25Retriever().query("zzz", bank_of(("alpha beta", "t1")), 5) == []


def test_indexed_text_prepends_retrieval_context():
    case = make_case("body text", "t1", PROV, {"retrieval_context": "context line"})
    assert indexed_text(case) == "context line\nbody text"
    plain = make_case("body text", "t1", PROV)
    assert indexed_text(plain) == "body text"


def test_bm25_matches_the_retrieval_context():
    """The context is retrieval surface: matching it surfaces the case even
    when the content shares no term with the query."""
    ctx = make_case("body one", "t1", PROV, {"retrieval_context": "annealing run"})
    bank = Bank(bank_id="b", cases=[ctx, make_case("body two", "t2", PROV)],
                manifest={}, dir=None)
    ranked = BM25Retriever().query("annealing", bank, 5)
    assert [c.id for c, _ in ranked] == [ctx.id]


def test_embedding_ranks_by_cosine():
    bank = bank_of(("alpha alpha alpha", "t1"), ("beta beta beta", "t2"))
    ranked = EmbeddingRetriever("fake-model", embed_fn=fake_embed).query(
        "alpha", bank, 10)
    assert ranked[0][0].content == "alpha alpha alpha"
    assert ranked[0][1] == pytest.approx(1.0)
    assert ranked[1][1] == pytest.approx(0.0)


def test_embedding_index_written_then_reused(tmp_path):
    calls = []

    def counting_embed(texts, is_query):
        calls.append((len(texts), is_query))
        return fake_embed(texts, is_query)

    bank = build_bank(tmp_path / "bank",
                      [make_case("alpha alpha", "t1", PROV),
                       make_case("beta beta", "t2", PROV)],
                      {"corpus_hash": "h"})
    r = EmbeddingRetriever("fake/model", embed_fn=counting_embed)
    r.embed_bank(bank)

    path = tmp_path / "bank" / "embeddings-fake--model.json"
    written = json.loads(path.read_text())
    assert written["model"] == "fake/model"
    assert len(written["ids"]) == 2 and len(written["vectors"]) == 2
    index_calls = len(calls)

    # a FRESH retriever (empty in-memory cache) must read the index, not rebuild
    fresh = EmbeddingRetriever("fake/model", embed_fn=counting_embed)
    ranked = fresh.query("alpha", bank, 10)
    assert ranked[0][0].content == "alpha alpha"
    assert calls[index_calls:] == [(1, True)]      # the query embed only


def test_embedding_index_rebuilt_when_model_differs(tmp_path):
    bank = build_bank(tmp_path / "bank", [make_case("alpha", "t1", PROV)],
                      {"corpus_hash": "h"})
    EmbeddingRetriever("fake/model", embed_fn=fake_embed).embed_bank(bank)
    other = tmp_path / "bank" / "embeddings-other--model.json"
    assert not other.exists()
    EmbeddingRetriever("other/model", embed_fn=fake_embed).embed_bank(bank)
    assert json.loads(other.read_text())["model"] == "other/model"


def test_hybrid_fuses_by_rrf():
    """A case surfaced by BOTH retrievers outranks one surfaced strongly by a
    single retriever."""
    both = make_case("alpha segment tree", "t1", PROV)
    lexical_only = make_case("segment tree segment tree", "t2", PROV)
    dense_only = make_case("alpha alpha alpha", "t3", PROV)
    bank = Bank(bank_id="b", cases=[both, lexical_only, dense_only], manifest={},
                dir=None)

    hybrid = HybridRetriever("fake-model", embed_fn=fake_embed, rrf_k=60)
    ranked = hybrid.query("alpha segment tree", bank, 10)
    assert ranked[0][0].id == both.id
    # rank-1 in both lists: 2/(60+1); each single-list case is at most 1/61+1/62
    assert ranked[0][1] == pytest.approx(2.0 / 61.0, abs=1e-6)
    assert ranked[0][1] > ranked[1][1]


def test_hybrid_rerank_orders_by_sigmoid_cross_encoder_score():
    bank = bank_of(("alpha one", "t1"), ("alpha two", "t2"), ("alpha three", "t3"))
    seen = {}

    def fake_ce(query, texts):
        seen["query"] = query
        seen["texts"] = list(texts)
        # deliberately invert the base order: the cross-encoder owns the result
        return [-4.0, 0.0, 4.0]

    r = HybridRerankRetriever("fake-model", reranker_model="fake-reranker",
                              embed_fn=fake_embed, ce_fn=fake_ce, rerank_pool=64)
    ranked = r.query("alpha", bank, 10)
    assert [c.content for c, _ in ranked] == [seen["texts"][2], seen["texts"][1],
                                              seen["texts"][0]]
    assert [s for _, s in ranked] == [pytest.approx(0.982014, abs=1e-5),
                                      pytest.approx(0.5), pytest.approx(0.017986,
                                                                        abs=1e-5)]
    assert seen["query"] == "alpha"


def test_hybrid_rerank_pool_bounds_the_cross_encoder():
    bank = bank_of(*[(f"alpha case {i}", f"t{i}") for i in range(10)])
    sizes = []

    def fake_ce(query, texts):
        sizes.append(len(texts))
        return [float(i) for i in range(len(texts))]

    r = HybridRerankRetriever("fake-model", embed_fn=fake_embed, ce_fn=fake_ce,
                              rerank_pool=3)
    assert len(r.query("alpha", bank, 10)) == 3
    assert sizes == [3]


def test_empty_bank_is_empty_everywhere():
    empty = Bank(bank_id="b", cases=[], manifest={}, dir=None)
    assert BM25Retriever().query("q", empty, 5) == []
    assert EmbeddingRetriever(embed_fn=fake_embed).query("q", empty, 5) == []
    assert HybridRetriever(embed_fn=fake_embed).query("q", empty, 5) == []
    assert HybridRerankRetriever(embed_fn=fake_embed,
                                 ce_fn=lambda q, t: []).query("q", empty, 5) == []


# ------------------------------------------------------------- MemoryView

class RankedRetriever:
    """Returns the bank's cases in bank order with descending scores."""
    name = "fixed"

    def query(self, text, bank, k):
        return [(c, float(len(bank.cases) - i))
                for i, c in enumerate(bank.cases)][:k]


def test_view_packs_greedily_and_skips_rather_than_stops():
    small_a, big, small_b = "a" * 40, "b" * 4000, "c" * 40
    bank = bank_of((small_a, "t1"), (big, "t2"), (small_b, "t3"))
    budget = approx_tokens(small_a) + approx_tokens(small_b)
    view = MemoryView(bank, RankedRetriever(), budget_tokens=budget)
    assert [c.content for c, _ in view.query("q")] == [small_a, small_b]


def test_view_drops_identical_content_once():
    bank = bank_of(("same text", "t1"), ("same text", "t2"), ("other text", "t3"))
    view = MemoryView(bank, RankedRetriever(), budget_tokens=10_000)
    packed = [c.content for c, _ in view.query("q")]
    assert packed == ["same text", "other text"]


def test_view_excludes_the_running_task_loto():
    bank = bank_of(("own case", "ahc001"), ("other case", "ahc002"))
    view = MemoryView(bank, RankedRetriever(), budget_tokens=10_000,
                      exclude_task_id="ahc001")
    assert [c.task_id for c, _ in view.query("q")] == ["ahc002"]


def test_view_injects_the_top_case_over_budget_when_it_alone_does_not_fit():
    big, small = "b" * 4000, "s" * 8
    bank = bank_of((big, "t1"), (small, "t2"))
    view = MemoryView(bank, RankedRetriever(), budget_tokens=approx_tokens(small))
    packed = view.query("q")
    assert [c.content for c, _ in packed] == [big]   # not the short substitute


def test_view_packs_normally_when_the_top_case_fits():
    bank = bank_of(("alpha", "t1"), ("b" * 4000, "t2"), ("gamma", "t3"))
    view = MemoryView(bank, RankedRetriever(),
                      budget_tokens=approx_tokens("alpha") + approx_tokens("gamma"))
    assert [c.content for c, _ in view.query("q")] == ["alpha", "gamma"]


def test_view_on_empty_bank():
    empty = Bank(bank_id="b", cases=[], manifest={}, dir=None)
    assert MemoryView(empty, RankedRetriever(), budget_tokens=100).query("q") == []
