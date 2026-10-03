from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI()


def embed(texts: list[str], model: str) -> list[list[float]]:
    resp = get_client().embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
