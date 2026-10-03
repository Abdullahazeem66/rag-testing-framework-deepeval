from dataclasses import dataclass

from rag_app.config import Settings, settings as default_settings
from rag_app.ingest import get_collection
from rag_app.llm import embed


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # cosine similarity, higher is better


class Retriever:
    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg
        self.collection = get_collection(cfg)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        res = self.collection.query(
            query_embeddings=embed([query], self.cfg.embedding_model),
            n_results=top_k or self.cfg.top_k,
        )
        return [
            RetrievedChunk(chunk_id=cid, doc_id=meta["doc_id"], text=doc, score=1 - dist)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
