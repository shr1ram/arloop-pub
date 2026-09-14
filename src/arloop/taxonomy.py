"""Per-attempt primitives the loop logs: code normalisation and error
signatures.

Code novelty is threshold-free normalised equality (AST round-trip -> token
lex with alpha-renaming -> line fallback); an error signature answers "the
same failure MODE?", not "the same bytes?".
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import keyword
import re
import tokenize
from dataclasses import dataclass

_BUILTIN_NAMES = frozenset(dir(builtins))

#: pseudo-tokens that keep block structure in the token stream (dropping
#: NEWLINE/INDENT/DEDENT outright would merge `if x: a` with `if x: a; b`)
_T_NL, _T_IN, _T_OUT = "<nl>", "<in>", "<out>"

#: a string literal at least this long is checked for embedded code
_MIN_EMBED_CHARS = 200
_MAX_EMBED_DEPTH = 2


@dataclass(frozen=True)
class CodeForm:
    """A solution source in canonical form, ready for comparison."""
    tokens: tuple            # alpha-normalised token strings
    mode: str                # ast | tokens | lines (how far normalisation got)
    fingerprint: str         # md5 of the token stream - the equality proxy


def _embedded_code(tok_string: str) -> str | None:
    """Canonicalised source hiding inside a string literal, else None.

    Reactive solutions carry the actual solver as `SOLVER_CODE = r'''...'''`.
    Treating that literal as ONE token would make a full rewrite of the solver
    look like a one-token edit.
    """
    try:
        value = ast.literal_eval(tok_string)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(value, str) or "\n" not in value:
        return None
    try:
        tree = ast.parse(value)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    if len(tree.body) < 2:
        return None
    return ast.unparse(tree)


def _lex(src: str, depth: int = 0) -> list | None:
    """(is_name, token) pairs without comments or pure formatting; None if the
    source does not tokenise (the caller falls back to line mode)."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING,
                            tokenize.ENDMARKER):
                continue
            if tok.type == tokenize.NEWLINE:
                out.append((False, _T_NL))
            elif tok.type == tokenize.INDENT:
                out.append((False, _T_IN))
            elif tok.type == tokenize.DEDENT:
                out.append((False, _T_OUT))
            elif (tok.type == tokenize.STRING
                    and len(tok.string) >= _MIN_EMBED_CHARS
                    and depth < _MAX_EMBED_DEPTH):
                inner = _embedded_code(tok.string)
                sub = _lex(inner, depth + 1) if inner is not None else None
                if sub:
                    out.append((False, "<str>"))
                    out.extend(sub)
                    out.append((False, "</str>"))
                else:
                    out.append((False, tok.string))
            else:
                out.append((tok.type == tokenize.NAME, tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return None
    return out


def _alpha(toks) -> tuple:
    """Alpha-rename identifiers to first-occurrence indices.

    Keywords, builtins and attribute names (any NAME right after a ".") stay
    verbatim: renaming attributes would collapse `xs.append` and `xs.extend`
    into the same stream, which is a semantic change, not a rename.
    """
    mapping: dict[str, str] = {}
    out = []
    prev = ""
    for is_name, s in toks:
        if (is_name and not keyword.iskeyword(s)
                and not keyword.issoftkeyword(s)
                and s not in _BUILTIN_NAMES and prev != "."):
            if s not in mapping:
                mapping[s] = f"N{len(mapping)}"
            out.append(mapping[s])
        else:
            out.append(s)
        prev = s
    return tuple(out)


def normalize_code(src: str) -> CodeForm:
    """Canonicalise a solution source (see the module docstring for the tiers)."""
    canon, mode = None, "ast"
    try:
        canon = ast.unparse(ast.parse(src))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        mode = "tokens"
    toks = _lex(canon if canon is not None else src)
    if toks is None:
        mode = "lines"
        alpha = tuple(ln.strip() for ln in src.splitlines() if ln.strip())
    else:
        alpha = _alpha(toks)
    fp = hashlib.md5("\x00".join(alpha).encode()).hexdigest()[:12]
    return CodeForm(tokens=alpha, mode=mode, fingerprint=fp)


_SIG_TAIL_LINES = 6

_SIG_SUBS = (
    (re.compile(r'File "[^"]*"'), "File <PATH>"),
    (re.compile(r"(?:/[\w.\-+]+){2,}"), "<PATH>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<ADDR>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]?[\d:.]*\b"), "<TS>"),
    (re.compile(r"line \d+"), "line <N>"),
    (re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"), "<NUM>"),
    (re.compile(r"\s+"), " "),
)


def error_signature(error_text: str, exit_code=None) -> tuple[str, str]:
    """(hash, normalised tail) for a failure, robust to paths, line numbers,
    addresses, timestamps and numeric values."""
    lines = [ln for ln in (error_text or "").splitlines() if ln.strip()]
    norm = " | ".join(lines[-_SIG_TAIL_LINES:])
    for rx, rep in _SIG_SUBS:
        norm = rx.sub(rep, norm)
    norm = norm.strip()
    if not norm:
        norm = f"<no-error-text:exit={exit_code}>"
    return hashlib.md5(norm.encode()).hexdigest()[:10], norm
