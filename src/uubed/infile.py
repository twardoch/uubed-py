# this_file: src/uubed/infile.py
"""Atomic embedding trailers for UTF-8 text, Markdown, XML and TMX.

The trailer is an XML processing instruction, so URL-safe base64 double
hyphens cannot break XML comments. Removal restores the original bytes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from uubed.compact import PackedEmbedding

MARKER = b"\n<?uubed "
SUFFIXES = {".md", ".markdown", ".txt", ".xml", ".tmx"}


def _split(data: bytes) -> tuple[bytes, PackedEmbedding | None, str | None]:
    if b"<?uubed " not in data:
        return data, None, None
    if data.count(b"<?uubed ") != 1 or MARKER not in data or not data.endswith(b"?>\n"):
        msg = "malformed or duplicate embedding trailer"
        raise ValueError(msg)
    source, trailer = data.rsplit(MARKER, 1)
    try:
        digest, record = trailer[:-3].decode("ascii").split(" ", 1)
        if len(digest) != hashlib.sha256().digest_size * 2 or any(c not in "0123456789abcdef" for c in digest):
            msg = "invalid source digest"
            raise ValueError(msg)
        return source, PackedEmbedding.loads(record), digest
    except (UnicodeError, ValueError) as error:
        msg = "invalid embedding trailer"
        raise ValueError(msg) from error


def source_bytes(path: Path | str) -> bytes:
    """Return source bytes without the generated trailer, even if now stale."""
    return _split(Path(path).read_bytes())[0]


def read_embedding(path: Path | str) -> PackedEmbedding | None:
    """Read a vector only when the accompanying source hash still matches."""
    source, record, digest = _split(Path(path).read_bytes())
    if record is not None and hashlib.sha256(source).hexdigest() != digest:
        msg = "embedding is stale: source content changed"
        raise ValueError(msg)
    return record


def _validate(path: Path, source: bytes) -> None:
    if path.is_symlink() or not path.is_file():
        msg = "embedding writes require a regular file, not a symlink"
        raise ValueError(msg)
    if path.suffix.lower() not in SUFFIXES:
        msg = "supported files: UTF-8 .txt, .md, .markdown, .xml and .tmx"
        raise ValueError(msg)
    try:
        text = source.decode("utf-8")
        if "\x00" in text:
            msg = "binary source is not supported"
            raise ValueError(msg)
        if path.suffix.lower() in {".xml", ".tmx"}:
            if "<!ENTITY" in text:
                msg = "XML entity declarations are not supported"
                raise ValueError(msg)
            ET.fromstring(source)  # noqa: S314 - local XML; entity declarations rejected; ElementTree does not fetch external DTDs
    except (UnicodeError, ET.ParseError) as error:
        msg = "source must be valid UTF-8 text or XML"
        raise ValueError(msg) from error


def _replace(path: Path, before: bytes, after: bytes) -> None:
    if before == after:
        return
    permissions = path.stat().st_mode & 0o777
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(after)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(permissions)
        if path.is_symlink() or path.read_bytes() != before:
            msg = "source changed during embedding write"
            raise ValueError(msg)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_embedding(path: Path | str, record: PackedEmbedding, *, expected_sha256: str | None = None) -> None:
    """Insert/replace one document embedding; optionally enforce the pre-inference hash.

    Pass ``expected_sha256`` when inference and writing are separate steps to
    reject a file edited while inference ran. Source bytes and mode are retained.
    """
    path = Path(path)
    before = path.read_bytes()
    source, _, _ = _split(before)
    _validate(path, source)
    digest = hashlib.sha256(source).hexdigest()
    if expected_sha256 is not None and expected_sha256 != digest:
        msg = "source changed during inference"
        raise ValueError(msg)
    record.vector()
    trailer = MARKER + digest.encode() + b" " + record.dumps().encode("ascii") + b"?>\n"
    _replace(path, before, source + trailer)


def remove_embedding(path: Path | str) -> None:
    """Remove the generated trailer and restore the source byte-for-byte."""
    path = Path(path)
    before = path.read_bytes()
    source, _, _ = _split(before)
    _validate(path, source)
    _replace(path, before, source)
