from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, List, Optional

from django.conf import settings


@functools.lru_cache(maxsize=1)
def get_milvus_client():
    # Ленивая загрузка, чтобы не падать при импорте проекта.
    from milvus_client import MilvusClient

    return MilvusClient(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT, alias=settings.MILVUS_ALIAS)


def milvus_list_collections() -> list[str]:
    """
    Получить список коллекций в Milvus.

    Почему отдельная функция:
    - в разных версиях pymilvus API клиента отличается (у объекта может не быть list_collections)
    - utility.list_collections() работает стабильно, но сигнатуры различаются (using= может отсутствовать)
    """
    client = get_milvus_client()

    # Сначала пробуем через метод клиента (если он есть).
    maybe = getattr(client, "list_collections", None)
    if callable(maybe):
        return list(maybe())

    # Fallback на pymilvus.utility
    from pymilvus import utility

    alias = getattr(client, "alias", None) or settings.MILVUS_ALIAS
    try:
        return list(utility.list_collections(using=alias))
    except TypeError:
        # На старых версиях может не быть параметра using=
        return list(utility.list_collections())


@dataclass(frozen=True)
class EmbedderHandle:
    model_name: str
    batch_size: int

    def encode_passages(self, texts: List[str]) -> List[List[float]]:
        return get_embedder(self.model_name, self.batch_size).encode(texts)

    def encode_query(self, query: str) -> List[float]:
        return get_embedder(self.model_name, self.batch_size).encode_query(query)

    def dimension(self) -> int:
        return get_embedder(self.model_name, self.batch_size).get_dimension()


@functools.lru_cache(maxsize=4)
def get_embedder(model_name: str, batch_size: int):
    # Тяжёлая зависимость (torch/sentence-transformers) — импортируем только по необходимости.
    from embedder import Embedder

    return Embedder(model_name=model_name, batch_size=batch_size)


def get_embedder_handle(model_name: Optional[str] = None, batch_size: Optional[int] = None) -> EmbedderHandle:
    mn = (model_name or settings.DEFAULT_MODEL_NAME).strip() or settings.DEFAULT_MODEL_NAME
    bs = int(batch_size or settings.DEFAULT_EMBED_BATCH_SIZE)
    return EmbedderHandle(model_name=mn, batch_size=bs)


def make_document_processor(
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    from document_processor import DocumentProcessor

    handle = get_embedder_handle(model_name=model_name, batch_size=batch_size)

    # DocumentProcessor ожидает функцию: List[str] -> List[List[float]]
    embedding_function: Callable[[List[str]], List[List[float]]] = handle.encode_passages

    return DocumentProcessor(
        milvus_client=get_milvus_client(),
        chunk_size=int(chunk_size or settings.DEFAULT_CHUNK_SIZE),
        chunk_overlap=int(chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP),
        embedding_function=embedding_function,
    )

