"""The model server end to end over a real Unix socket, with stub models
injected at the load seam so sentence-transformers never loads."""
import threading
import time

import pytest

from arloop.memory import model_server as ms


class StubModel:
    """Stands in for SentenceTransformer and CrossEncoder: deterministic, no torch."""

    def __init__(self):
        self.batch_sizes = []

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        self.batch_sizes.append(batch_size)
        return [[float(len(t)), float(ord(t[0]) if t else 0)] for t in texts]

    def predict(self, pairs, batch_size=32):
        self.batch_sizes.append(batch_size)
        return [float(len(q) + len(t)) for q, t in pairs]


def start(tmp_path, embed_model="embed-x", reranker_model="rerank-x"):
    """A server on a tmp socket with stub models, running in a thread."""
    sock = str(tmp_path / f"{embed_model}.sock")
    srv = ms.ModelServer(embed_model, reranker_model, device="cpu",
                         socket_path=sock, idle_timeout_s=0)
    srv._embed = StubModel()
    srv._cross = StubModel()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for _ in range(500):
        if ms.ModelClient(sock).available():
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("test server did not come up")
    return srv, sock


@pytest.fixture
def server(tmp_path):
    srv, sock = start(tmp_path)
    yield srv, sock
    if srv._srv is not None:
        srv._srv.close()        # exit the accept loop so the thread does not leak


def test_ping_info_embed_and_cross(server):
    srv, sock = server
    c = ms.ModelClient(sock)
    assert c.available()
    assert c.info() == {"embed_model": "embed-x", "reranker_model": "rerank-x",
                        "device": "cpu"}
    assert c.embed_fn(["ab", "cde"], is_query=True) == [[2.0, float(ord("a"))],
                                                        [3.0, float(ord("c"))]]
    assert c.ce_fn("query", ["a", "bb"]) == [6.0, 7.0]
    assert srv._cross.batch_sizes == [ms.CROSS_MAX_BATCH]


def test_client_fns_drive_the_retrievers(server, tmp_path):
    """embed_fn / ce_fn must satisfy exactly what the retrievers inject."""
    from arloop.memory.bank import Bank
    from arloop.memory.cases import make_case
    from arloop.memory.retrievers import HybridRerankRetriever

    _, sock = server
    c = ms.ModelClient(sock)
    prov = [{"run_id": "r", "seq_span": [0, 1]}]
    bank = Bank(bank_id="b", cases=[make_case("lesson one", "t1", prov),
                                    make_case("lesson two two", "t2", prov)],
                manifest={}, dir=None)
    r = HybridRerankRetriever("embed-x", reranker_model="rerank-x",
                              embed_fn=c.embed_fn, ce_fn=c.ce_fn)
    assert len(r.query("lesson", bank, 2)) == 2


def test_requests_are_chunked(server):
    _, sock = server
    c = ms.ModelClient(sock)
    texts = [f"text-{i}" for i in range(ms.RPC_MAX_TEXTS * 2 + 3)]
    assert len(c.embed_fn(texts, is_query=False)) == len(texts)
    assert len(c.ce_fn("q", texts)) == len(texts)


def test_chunk_texts_bounds_count_and_chars():
    assert list(ms.chunk_texts(["a", "b", "c"], 2, 1000)) == [["a", "b"], ["c"]]
    assert list(ms.chunk_texts(["aaa", "bbb"], 10, 4)) == [["aaa"], ["bbb"]]
    # a single over-budget text still goes, alone
    assert list(ms.chunk_texts(["x" * 50], 10, 4)) == [["x" * 50]]


def test_bad_op_errors_without_killing_the_server(server):
    _, sock = server
    c = ms.ModelClient(sock)
    with pytest.raises(RuntimeError, match="unknown op"):
        c._rpc({"op": "nope"})
    assert c.available()


def test_available_is_false_without_a_server(tmp_path):
    assert ms.ModelClient(str(tmp_path / "nope.sock")).available() is False


def test_connect_raises_when_no_socket(tmp_path):
    missing = str(tmp_path / "absent.sock")
    with pytest.raises(RuntimeError, match=missing):
        ms.connect("embed-x", "rerank-x", socket_path=missing)


def test_connect_raises_on_model_mismatch(server):
    _, sock = server
    with pytest.raises(RuntimeError, match="embed"):
        ms.connect("other-embed", "rerank-x", socket_path=sock)
    with pytest.raises(RuntimeError, match="rerank"):
        ms.connect("embed-x", "other-rerank", socket_path=sock)


def test_connect_returns_a_client_on_a_match(server):
    _, sock = server
    client = ms.connect("embed-x", "rerank-x", socket_path=sock)
    assert isinstance(client, ms.ModelClient)
    assert client.embed_fn(["ab"], is_query=False) == [[2.0, float(ord("a"))]]


def test_serve_forever_refuses_to_bind_over_a_live_socket(server):
    _, sock = server
    other = ms.ModelServer("embed-y", "rerank-y", device="cpu", socket_path=sock,
                           idle_timeout_s=0)
    other._embed = StubModel()
    other._cross = StubModel()
    other.serve_forever()       # returns immediately rather than stealing
    assert ms.connect("embed-x", "rerank-x", socket_path=sock) is not None


def test_oom_safe_halves_the_batch_then_succeeds():
    srv = ms.ModelServer("e", "r", device="cpu", socket_path="/tmp/unused.sock")
    tried = []

    def run(bs):
        tried.append(bs)
        if bs > 2:
            raise RuntimeError("CUDA out of memory")
        return "done"

    assert srv._oom_safe(run, 16, "embed", ms.EMBED_MAX_BATCH) == "done"
    assert tried == [8, 4, 2]


def test_oom_safe_reraises_non_oom_errors():
    srv = ms.ModelServer("e", "r", device="cpu", socket_path="/tmp/unused.sock")

    def run(bs):
        raise RuntimeError("device-side assert triggered")

    with pytest.raises(RuntimeError, match="device-side"):
        srv._oom_safe(run, 8, "cross", ms.EMBED_MAX_BATCH)


def test_default_socket_path_honours_the_env(monkeypatch):
    monkeypatch.setenv("ARLOOP_MODEL_SOCKET", "/tmp/explicit.sock")
    assert ms.default_socket_path() == "/tmp/explicit.sock"
    monkeypatch.delenv("ARLOOP_MODEL_SOCKET")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp/run")
    assert ms.default_socket_path().startswith("/tmp/run/arloop-models-")
