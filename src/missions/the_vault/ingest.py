"""Corpus chunking: Markdown files -> header-aware, size-bounded ``Document`` chunks.

This module turns the corpus into chunks
"""

from pathlib import Path

import structlog
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

log = structlog.get_logger()

CORPUS_DIR = Path(__file__).parent / "corpus"

# Split Markdown on headers first, then split each section to a bounded size.
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _section_path(metadata: dict[str, str]) -> str:
    """Breadcrumb like ``h1 > h2 > h3`` from a section's header metadata."""
    return " > ".join(
        value for value in (metadata.get("h1"), metadata.get("h2"), metadata.get("h3")) if value
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split each Markdown document into header-aware, size-bounded chunks."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    body_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks: list[Document] = []
    for document in documents:
        source = Path(document.metadata["source"]).name
        for section in header_splitter.split_text(document.page_content):
            section_path = _section_path(section.metadata)
            for index, piece in enumerate(body_splitter.split_documents([section])):
                piece.metadata = {
                    "source": source,
                    "section": section_path,
                    "chunk": index,
                }
                chunks.append(piece)
    log.info("split corpus into chunks", documents=len(documents), chunks=len(chunks))
    return chunks


def load_chunks() -> list[Document]:
    """Load every corpus Markdown file (except the README) and chunk it."""
    loader = DirectoryLoader(
        str(CORPUS_DIR),
        glob="*.md",
        exclude="README.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    log.info("loaded corpus", corpus_dir=str(CORPUS_DIR), files=len(documents))
    return split_documents(documents)
