"""Tests for the corpus chunking pipeline in the_vault/ingest.py."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from missions.the_vault import ingest

MARKDOWN = """# Vault Handbook

Some intro prose about the vault.

## Access

You need a keycard and a PIN.

### Emergency

Break the glass.

## Contents

Gold bars and paperwork.
"""


def test_section_path_joins_present_headers():
    assert ingest._section_path({"h1": "A", "h2": "B", "h3": "C"}) == "A > B > C"


def test_section_path_skips_missing_levels():
    assert ingest._section_path({"h1": "A", "h3": "C"}) == "A > C"


def test_section_path_is_empty_when_no_headers():
    assert ingest._section_path({}) == ""


def test_split_documents_produces_header_aware_chunks():
    chunks = ingest.split_documents(
        [Document(page_content=MARKDOWN, metadata={"source": "handbook.md"})]
    )

    assert chunks, "expected at least one chunk"
    # Every chunk carries the trimmed metadata contract the vector store relies on.
    for chunk in chunks:
        assert set(chunk.metadata) == {"source", "section", "chunk"}
        assert chunk.metadata["source"] == "handbook.md"
        assert isinstance(chunk.metadata["chunk"], int)

    sections = {chunk.metadata["section"] for chunk in chunks}
    assert "Vault Handbook" in sections
    assert "Vault Handbook > Access" in sections
    assert "Vault Handbook > Access > Emergency" in sections
    assert "Vault Handbook > Contents" in sections


def test_split_documents_reduces_source_to_the_bare_filename():
    chunks = ingest.split_documents(
        [Document(page_content=MARKDOWN, metadata={"source": "/abs/path/to/corpus/handbook.md"})]
    )

    assert {chunk.metadata["source"] for chunk in chunks} == {"handbook.md"}


def test_split_documents_numbers_chunks_within_each_section(
    monkeypatch: pytest.MonkeyPatch,
):
    # Force the body splitter to cut the "Access" section into several pieces.
    monkeypatch.setattr(ingest, "CHUNK_SIZE", 20)
    monkeypatch.setattr(ingest, "CHUNK_OVERLAP", 0)

    chunks = ingest.split_documents(
        [Document(page_content=MARKDOWN, metadata={"source": "handbook.md"})]
    )

    by_section: dict[str, list[int]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.metadata["section"], []).append(chunk.metadata["chunk"])

    multi = next(indexes for indexes in by_section.values() if len(indexes) > 1)
    assert multi == list(range(len(multi)))  # 0, 1, 2, ... per section


def test_load_chunks_reads_corpus_markdown_and_excludes_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "alpha.md").write_text("# Alpha\n\nalpha body\n", encoding="utf-8")
    (tmp_path / "beta.md").write_text("# Beta\n\nbeta body\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Readme\n\nignore me\n", encoding="utf-8")
    monkeypatch.setattr(ingest, "CORPUS_DIR", tmp_path)

    chunks = ingest.load_chunks()

    sources = {chunk.metadata["source"] for chunk in chunks}
    assert sources == {"alpha.md", "beta.md"}
