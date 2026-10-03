"""Load markdown docs, split them into heading-aware chunks and index them in Chroma.

Run:  python -m rag_app.ingest [--rebuild]
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

from rag_app.config import Settings, settings as default_settings
from rag_app.llm import embed

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def split_markdown(doc_id: str, raw: str, max_chars: int) -> list[Chunk]:
    """One chunk per '## ' section, prefixed with the doc title so each chunk stands alone."""
    raw = HTML_COMMENT.sub("", raw)
    title_match = re.search(r"^# (.+)$", raw, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else doc_id

    parts = re.split(r"^## (.+)$", raw, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []
    intro = re.sub(r"^# .+$", "", parts[0], flags=re.MULTILINE).strip()
    if intro:
        sections.append(("Overview", intro))
    for heading, body in zip(parts[1::2], parts[2::2]):
        sections.append((heading.strip(), body.strip()))

    chunks = []
    for heading, body in sections:
        # Long sections are split on paragraph boundaries.
        pieces, buf = [], ""
        for para in body.split("\n\n"):
            if buf and len(buf) + len(para) > max_chars:
                pieces.append(buf)
                buf = ""
            buf = f"{buf}\n\n{para}".strip()
        if buf:
            pieces.append(buf)
        for i, piece in enumerate(pieces):
            suffix = f"-{i}" if len(pieces) > 1 else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{_slug(heading)}{suffix}",
                    doc_id=doc_id,
                    title=title,
                    section=heading,
                    text=f"{title} > {heading}\n{piece}",
                )
            )
    return chunks


def load_chunks(cfg: Settings) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(cfg.docs_dir).rglob("*.md")):
        chunks += split_markdown(path.stem, path.read_text(encoding="utf-8"), cfg.chunk_max_chars)
    return chunks


def get_collection(cfg: Settings):
    client = chromadb.PersistentClient(path=str(cfg.index_dir))
    return client.get_or_create_collection(cfg.collection_name, metadata={"hnsw:space": "cosine"})


def build_index(cfg: Settings = default_settings, rebuild: bool = False) -> int:
    client = chromadb.PersistentClient(path=str(cfg.index_dir))
    if rebuild:
        try:
            client.delete_collection(cfg.collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(cfg.collection_name, metadata={"hnsw:space": "cosine"})
    if collection.count() and not rebuild:
        return collection.count()

    chunks = load_chunks(cfg)
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed([c.text for c in chunks], cfg.embedding_model),
        metadatas=[{"doc_id": c.doc_id, "title": c.title, "section": c.section} for c in chunks],
    )
    return collection.count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="drop and re-create the index")
    args = parser.parse_args()
    n = build_index(rebuild=args.rebuild)
    print(f"Indexed {n} chunks into '{default_settings.collection_name}' at {default_settings.index_dir}")
