"""Per-box retrieval model server: load the embedding + reranker models ONCE
per box and serve every worker over a Unix socket.

Each run_cell is its own process, so without this each loads its own model
copies and oversubscribes the box's cores. The operator starts the server; a
cell either connects to it or raises.

Protocol (newline-delimited JSON over AF_UNIX):
  request   {"op": "ping"} | {"op": "info"}
            {"op": "embed", "texts": [...], "is_query": bool}
            {"op": "cross", "query": "...", "texts": [...]}
  response  {"ok": true, "result": ...} | {"ok": false, "error": "..."}

The client's embed_fn / ce_fn are exactly the hooks the retrievers inject.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path

from arloop.memory.retrievers import DEFAULT_EMBED_MODEL, DEFAULT_RERANKER_MODEL

log = logging.getLogger(__name__)

# sentence-transformers sorts a request's texts by length and encodes in
# batches of 32 by default, so one whole-bank request puts the 32 LONGEST
# texts into a single forward pass — a multi-GB activation allocation for
# episode-scale cases. Bound the batch and halve on OOM instead.
EMBED_MAX_BATCH = 8
# Cross-encoder batching is a loss here: over heterogeneous case lengths the
# padding compute exceeds the batching win, and starting higher makes every
# call climb an OOM ladder first. Scores are identical across batch sizes.
CROSS_MAX_BATCH = 1

RPC_MAX_TEXTS = 16
RPC_MAX_CHARS = 200_000


def default_socket_path() -> str:
    """One socket per (user, box); AF_UNIX sockets are host-local anyway."""
    run = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    host = socket.gethostname().split(".")[0]
    return os.environ.get("ARLOOP_MODEL_SOCKET",
                          str(Path(run) / f"arloop-models-{host}.sock"))


def cross_encoder_kwargs(device: str) -> dict:
    """Extra CrossEncoder kwargs: fp32 on CPU.

    sentence-transformers resolves the dtype from the checkpoint config, so a
    reranker declaring bfloat16 loads as bf16 — which cores without native
    bf16 emulate at roughly twice the cost of a plain fp32 cast, while bf16's
    8-bit mantissa collapses nearby scores into ties fp32 resolves. On CUDA no
    override: bf16 runs natively and an fp32 cast would double VRAM."""
    if device != "cpu":
        return {}
    import torch
    return {"model_kwargs": {"torch_dtype": torch.float32}}


class ModelServer:
    """Serves one resident embedding model and one cross-encoder over a Unix
    socket. One lock per model, one thread per connection."""

    def __init__(self, embed_model: str, reranker_model: str,
                 device: str = "cuda", socket_path: str | None = None,
                 idle_timeout_s: float = 1800.0):
        self.embed_model = embed_model
        self.reranker_model = reranker_model
        self.device = device
        self.socket_path = socket_path or default_socket_path()
        # A detached server outliving its grid holds the card's allocator cache
        # on a shared box; no requests for idle_timeout_s means exit.
        self.idle_timeout_s = idle_timeout_s
        self._last_activity = time.time()
        self._embed = None
        self._cross = None
        # Separate locks: embed and rerank are different models, so they must
        # not serialise behind one another.
        self._embed_lock = threading.Lock()
        self._cross_lock = threading.Lock()
        self._srv: socket.socket | None = None

    # The load seam: tests set _embed / _cross to stubs so no model is loaded.
    def _ensure_embed(self):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer
            self._embed = SentenceTransformer(self.embed_model, device=self.device)
        return self._embed

    def _ensure_cross(self):
        if self._cross is None:
            from sentence_transformers import CrossEncoder
            self._cross = CrossEncoder(self.reranker_model, device=self.device,
                                       **cross_encoder_kwargs(self.device))
        return self._cross

    @staticmethod
    def _is_cuda_oom(e: Exception) -> bool:
        # Matching the message avoids importing torch on paths that never load
        # it. Device-side asserts and shape errors must NOT match: retrying
        # those at a smaller batch cannot help.
        return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()

    def _oom_safe(self, run, n_items: int, what: str, max_batch: int):
        """run(batch_size), halving on CUDA OOM down to 1 (releasing the
        allocator's fragmented reserve between tries). At batch 1 an OOM is
        real and propagates to the client."""
        bs = max(1, min(max_batch, n_items or 1))
        while True:
            try:
                return run(bs)
            except RuntimeError as e:
                if not self._is_cuda_oom(e) or bs <= 1:
                    raise
                bs = max(1, bs // 2)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 — cache release is best-effort
                    pass
                log.warning("%s OOM — empty_cache, retrying at batch_size=%d",
                            what, bs)

    def _handle(self, req: dict) -> dict:
        self._last_activity = time.time()
        op = req.get("op")
        try:
            if op == "ping":
                return {"ok": True, "result": "pong"}
            if op == "info":
                return {"ok": True, "result": {"embed_model": self.embed_model,
                                               "reranker_model": self.reranker_model,
                                               "device": self.device}}
            if op == "embed":
                texts = req["texts"]
                with self._embed_lock:
                    vecs = self._oom_safe(
                        lambda bs: self._ensure_embed().encode(
                            texts, normalize_embeddings=True, batch_size=bs),
                        len(texts), "embed", EMBED_MAX_BATCH)
                return {"ok": True, "result": [[float(x) for x in v] for v in vecs]}
            if op == "cross":
                pairs = [(req["query"], t) for t in req["texts"]]
                with self._cross_lock:
                    scores = self._oom_safe(
                        lambda bs: self._ensure_cross().predict(pairs, batch_size=bs),
                        len(pairs), "cross", CROSS_MAX_BATCH)
                return {"ok": True, "result": [float(x) for x in scores]}
            return {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001 — a bad request must not kill the server
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _serve_conn(self, conn: socket.socket) -> None:
        # A dropped peer is normal, not an error: swallow it rather than spew a
        # traceback or kill the thread.
        try:
            with conn:
                for line in conn.makefile("rb"):
                    if not line.strip():
                        continue
                    try:
                        req = json.loads(line)
                    except json.JSONDecodeError:
                        resp = {"ok": False, "error": "bad json"}
                    else:
                        resp = self._handle(req)
                    conn.sendall((json.dumps(resp) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def serve_forever(self) -> None:
        """Preload both models, bind, and serve until idle or closed."""
        self._ensure_embed()
        self._ensure_cross()
        p = Path(self.socket_path)
        if p.exists():
            # A LIVE server must not be evicted: unlink-then-bind would steal
            # its socket and leave two model-loaded servers on the box. Only a
            # dead socket file is cleaned up.
            if ModelClient(self.socket_path, timeout=5.0).available():
                log.error("another model server is live on %s — refusing to "
                          "replace it", self.socket_path)
                return
            p.unlink()
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self._srv.listen(64)
        if self.idle_timeout_s > 0:
            # Poll accept() so the idle check runs even with zero traffic.
            self._srv.settimeout(min(60.0, max(0.05, self.idle_timeout_s / 4)))
        try:
            while True:
                try:
                    conn, _ = self._srv.accept()
                except socket.timeout:
                    # Subclass of OSError — must be caught FIRST, or an idle
                    # poll would read as socket-closed shutdown.
                    idle = time.time() - self._last_activity
                    if 0 < self.idle_timeout_s <= idle:
                        log.info("model server idle %.0fs — shutting down", idle)
                        break
                    continue
                except OSError:
                    break
                self._last_activity = time.time()
                threading.Thread(target=self._serve_conn, args=(conn,),
                                 daemon=True).start()
        finally:
            self._srv.close()
            Path(self.socket_path).unlink(missing_ok=True)


def chunk_texts(texts: list[str], max_texts: int, max_chars: int):
    """Greedy count/char-bounded chunking; a single over-budget text still goes,
    alone, never dropped."""
    chunk: list[str] = []
    chars = 0
    for t in texts:
        if chunk and (len(chunk) >= max_texts or chars + len(t) > max_chars):
            yield chunk
            chunk, chars = [], 0
        chunk.append(t)
        chars += len(t)
    if chunk:
        yield chunk


class ModelClient:
    """Thin client exposing embed_fn(texts, is_query) and ce_fn(query, texts) —
    the exact signatures the retrievers inject. Connects per call (cheap over
    AF_UNIX) and raises if no server answers."""

    def __init__(self, socket_path: str | None = None, timeout: float = 120.0,
                 cross_timeout: float = 600.0):
        self.socket_path = socket_path or default_socket_path()
        self.timeout = timeout
        # A rerank scores the query against every candidate at full sequence
        # length, so its cost tracks case LENGTH, not just count; embed keeps
        # the tighter bound because it has no comparable tail.
        self.cross_timeout = cross_timeout

    def _rpc(self, req: dict):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.cross_timeout if req.get("op") == "cross" else self.timeout)
        s.connect(self.socket_path)
        try:
            s.sendall((json.dumps(req) + "\n").encode())
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        finally:
            s.close()
        resp = json.loads(data)
        if not resp.get("ok"):
            raise RuntimeError(f"model server: {resp.get('error')}")
        return resp["result"]

    def available(self) -> bool:
        """True when a server answers ping on this socket."""
        try:
            return self._rpc({"op": "ping"}) == "pong"
        except Exception:  # noqa: BLE001 — any failure means 'not available'
            return False

    def info(self) -> dict:
        """The models and device the server is serving."""
        return self._rpc({"op": "info"})

    def embed_fn(self, texts, is_query) -> list[list[float]]:
        """Embedding hook for EmbeddingRetriever, one RPC per bounded chunk."""
        out: list[list[float]] = []
        for chunk in chunk_texts(list(texts), RPC_MAX_TEXTS, RPC_MAX_CHARS):
            out.extend(self._rpc({"op": "embed", "texts": chunk,
                                  "is_query": bool(is_query)}))
        return out

    def ce_fn(self, query, texts) -> list[float]:
        """Cross-encoder hook for CrossScorer, one RPC per bounded chunk."""
        out: list[float] = []
        for chunk in chunk_texts(list(texts), RPC_MAX_TEXTS, RPC_MAX_CHARS):
            out.extend(self._rpc({"op": "cross", "query": query, "texts": chunk}))
        return out


def connect(embed_model: str, reranker_model: str,
            socket_path: str | None = None) -> ModelClient:
    """Return a client for the box's model server, or raise naming the socket.

    A mismatch is fatal rather than tolerated: a server holding different
    models would return embeddings from the wrong model while the bank's index
    cache keys on the configured name."""
    client = ModelClient(socket_path)
    if not client.available():
        raise RuntimeError(f"no model server is answering on {client.socket_path} — "
                           f"start one with `python -m arloop.memory.model_server`")
    info = client.info()
    for want, got, what in ((embed_model, info.get("embed_model"), "embed"),
                            (reranker_model, info.get("reranker_model"), "rerank")):
        if want != got:
            raise RuntimeError(
                f"model server on {client.socket_path} serves {what}={got!r} but "
                f"this arm wants {want!r}")
    return client


def main() -> None:
    """CLI entry point: run the per-box model server in the foreground."""
    ap = argparse.ArgumentParser(description="per-box retrieval model server")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--socket", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    srv = ModelServer(a.embed_model, a.reranker_model, device=a.device,
                      socket_path=a.socket)
    print(f"[model-server] listening on {srv.socket_path} "
          f"(embed={a.embed_model} rerank={a.reranker_model} device={a.device})",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
