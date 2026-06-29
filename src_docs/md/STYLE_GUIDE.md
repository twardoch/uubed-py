---
title: Style Guide
---

# uubed Style Guide

Guidelines for contributors to the `uubed` Python package.

---

## Code style

- **Formatter**: `ruff format` (Black-compatible, line length 88).
- **Linter**: `ruff check --fix --unsafe-fixes`.
- **Type checker**: `mypy` in strict mode; all public functions must have full type annotations.
- **Python target**: 3.12+. Use PEP 695 type alias syntax (`type Foo = ...`) and `X | Y` unions (not `Optional[X]`).

Run all checks in one step:

```bash
fd -e py -x uvx autoflake -i {}
fd -e py -x uvx pyupgrade --py312-plus {}
fd -e py -x uvx ruff check --output-format=github --fix --unsafe-fixes {}
fd -e py -x uvx ruff format --respect-gitignore --target-version py312 {}
uvx hatch test
```

---

## File headers

Every source file must begin with:

```python
#!/usr/bin/env python3
# this_file: src/uubed/module_name.py
"""One-line module summary.

Extended description (optional).
"""
```

---

## Docstrings

Use Google-style docstrings. All public functions, classes, and methods require:

- Summary line (imperative mood, ≤ 79 chars).
- `Args:` section with type and description for every parameter.
- `Returns:` section with type and description.
- `Raises:` section listing every exception the function may raise, with conditions.
- At least one `Example:` showing typical usage.

```python
def encode(embedding: EmbeddingInput, method: str = "auto", **kwargs: Any) -> str:
    """Encode an embedding vector to a position-safe QuadB64 string.

    Args:
        embedding: Input vector as list[int], np.ndarray, or bytes.
            Integer values must be in [0, 255]; float values in [0, 1] or [0, 255].
        method: Encoding method. One of ``eq64``, ``shq64``, ``t8q64``,
            ``zoq64``, ``mq64``, or ``auto``.
        **kwargs: Method-specific parameters (e.g. ``planes`` for shq64).

    Returns:
        Encoded string using a position-dependent alphabet.

    Raises:
        UubedValidationError: If the embedding or parameters are invalid.
        UubedEncodingError: If encoding fails for any other reason.

    Example:
        >>> import numpy as np
        >>> from uubed import encode
        >>> emb = np.zeros(768, dtype=np.uint8)
        >>> s = encode(emb, method="eq64")
        >>> len(s) == 1024
        True
    """
```

---

## Error handling

- Raise from the `uubed.exceptions` hierarchy, never bare `ValueError`/`TypeError` in public APIs.
- Always pass `suggestion=` with a human-readable fix hint.
- Use the convenience factory functions (`validation_error()`, `encoding_error()`, etc.) rather than constructing exceptions directly.

```python
# Good
raise validation_error(
    "Embedding cannot be None",
    parameter="embedding",
    expected="array-like of integers 0-255",
    received="None",
)

# Avoid
raise ValueError("embedding is None")
```

---

## Testing

- Every public function must have at least one unit test.
- Tests live in `tests/` and are named `test_<module>.py`.
- Test functions follow the pattern `test_<function>_when_<condition>_then_<result>`.
- Use `np.zeros(N, dtype=np.uint8)` (not `np.array(range(N), ...)`) for large arrays — `range(N)` with N > 255 overflows `uint8` in NumPy 2.x.
- Never rely on global state; reset `uubed.config._global_config = None` in fixtures.

Run tests:

```bash
python -m pytest tests/ -q
# or
uvx hatch test
```

---

## Encoding format contract

All encoding methods must satisfy:

1. **Position-safety** — two distinct byte sequences never produce strings with a common non-trivial prefix or suffix.
2. **Printable ASCII output** — only characters in the range `0x21`–`0x7E` (no spaces, no control characters).
3. **Determinism** — same input always produces same output.
4. **Stable alphabet** — alphabet assignments are frozen after the first release; changing them is a breaking change.

---

## Versioning

The package uses `hatch-vcs` / `setuptools-scm` to derive the version from git tags. Do not edit `__version__` manually. Tag releases with `vX.Y.Z`.

---

## Commit messages

Follow Conventional Commits:

```
<type>(<scope>): <summary>

[optional body]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`.

Examples:

```
fix(validation): wrap path.exists() in PermissionError guard for Python 3.13
test(validation): replace np.array(range(N)) with np.zeros(N) for N > 255
feat(native_wrapper): add detailed fallback comments
```
