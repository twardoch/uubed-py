# this_file: tests/test_infile.py
"""Original-file preservation and stale-vector detection."""

import os

import numpy as np
import pytest

from uubed.compact import pack
from uubed.infile import read_embedding, remove_embedding, source_bytes, write_embedding


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("document.md", b"# Title\r\n\r\nPolski tekst: \xc5\x82\xc4\x85ka.\r\n"),
        ("document.txt", b"No trailing newline"),
        ("document.tmx", b'<?xml version="1.0" encoding="UTF-8"?><tmx version="1.4"><body/></tmx>'),
        ("empty.txt", b""),
    ],
)
def test_write_when_supported_file_then_preserve_replace_and_remove(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    path.chmod(0o640)
    first = pack(np.arange(1, 257), space="m")
    write_embedding(path, first)
    assert source_bytes(path) == content
    assert read_embedding(path) == first
    result = path.read_bytes()
    write_embedding(path, first)
    assert path.read_bytes() == result
    second = pack(-np.arange(1, 257), space="m")
    write_embedding(path, second)
    assert read_embedding(path) == second
    assert path.read_bytes().count(b"<?uubed ") == 1
    assert path.stat().st_mode & 0o777 == 0o640
    remove_embedding(path)
    assert path.read_bytes() == content
    assert read_embedding(path) is None
    remove_embedding(path)
    assert path.read_bytes() == content


def test_read_when_source_modified_then_reject_stale_record(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("Original")
    write_embedding(path, pack([1, 2], space="m"))
    path.write_bytes(path.read_bytes().replace(b"Original", b"Changed"))
    with pytest.raises(ValueError, match="stale"):
        read_embedding(path)


def test_write_when_replace_fails_then_original_intact(tmp_path, monkeypatch):
    path = tmp_path / "doc.txt"
    path.write_bytes(b"original")

    def fail(*_args):
        msg = "replace failed"
        raise OSError(msg)

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        write_embedding(path, pack([1], space="m"))
    assert path.read_bytes() == b"original"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doc.txt"]


@pytest.mark.parametrize(
    ("name", "content"), [("doc.pdf", b"%PDF"), ("doc.xml", b"<unclosed>"), ("doc.txt", b"\xff\x00")]
)
def test_write_when_unsupported_or_malformed_then_rejected(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(ValueError, match=r".+"):
        write_embedding(path, pack([1], space="m"))
    assert path.read_bytes() == content


def test_write_when_symlink_then_rejected(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("source")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(ValueError, match=r".+"):
        write_embedding(link, pack([1], space="m"))
    assert target.read_text() == "source"


def test_read_when_record_truncated_then_rejected(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("source")
    write_embedding(path, pack([1], space="m"))
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(ValueError, match=r".+"):
        read_embedding(path)


def test_write_when_tmx_external_dtd_then_preserved_without_fetch(tmp_path):
    path = tmp_path / "dictionary.tmx"
    content = b'<?xml version="1.0"?><!DOCTYPE tmx SYSTEM "https://invalid.example/tmx.dtd"><tmx><body/></tmx>'
    path.write_bytes(content)
    write_embedding(path, pack([1, -1], space="m"))
    assert source_bytes(path) == content
    remove_embedding(path)
    assert path.read_bytes() == content


def test_write_when_internal_entity_then_rejected(tmp_path):
    path = tmp_path / "dictionary.xml"
    content = b'<!DOCTYPE x [<!ENTITY e "text">]><x>&e;</x>'
    path.write_bytes(content)
    with pytest.raises(ValueError, match="entity"):
        write_embedding(path, pack([1, -1], space="m"))
    assert path.read_bytes() == content


def test_write_when_source_changed_during_inference_then_rejected(tmp_path):
    path = tmp_path / "doc.md"
    path.write_bytes(b"new content")
    with pytest.raises(ValueError, match="changed during inference"):
        write_embedding(path, pack([1, -1], space="m"), expected_sha256="0" * 64)
    assert path.read_bytes() == b"new content"
