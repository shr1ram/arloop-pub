"""Which provider errors ride the retry ladder, and which fail at once."""
import types

from arloop.llm import is_transient


class _Status(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _from_module(module: str) -> Exception:
    cls = types.new_class("ReadTimeout", (Exception,))
    cls.__module__ = module
    return cls("timed out")


def test_status_codes_split_on_retryability():
    assert is_transient(_Status(429))
    assert is_transient(_Status(503))
    assert not is_transient(_Status(400))
    assert not is_transient(_Status(413))


def test_transport_errors_are_transient_whatever_the_client_module_is_called():
    assert is_transient(_from_module("httpx._exceptions"))
    assert is_transient(_from_module("httpx2._exceptions"))
    assert is_transient(_from_module("openai._exceptions"))
    assert is_transient(ConnectionResetError())
    assert is_transient(TimeoutError())


def test_local_errors_are_deterministic():
    assert not is_transient(RuntimeError("ScriptedLLM ran out of responses"))
    assert not is_transient(ValueError("bad reply"))
