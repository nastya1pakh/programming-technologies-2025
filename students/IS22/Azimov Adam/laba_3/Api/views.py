from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter
from milvus_client import MilvusClient
from document_processor import DocumentProcessor
from embedder import create_embedding_function
from rest_framework.decorators import action, api_view
from rest_framework import status
from .serializers import CollectionCreateSerializer, SearchQuerySerializer, TextAddSerializer
from drf_spectacular.types import OpenApiTypes

from embedder import Embedder

model_name = r"C:\stady\prog-tech\lab3\multilingual-e5-base"

milvus = MilvusClient(host="localhost", port=19530)
embedding_func = create_embedding_function(model_name=model_name, batch_size=32)
processor = DocumentProcessor(milvus_client=milvus, chunk_size=256, chunk_overlap=64, embedding_function=embedding_func)
embedder = Embedder(model_name=model_name)


@api_view(["GET"])
def get_list_collections(request):
    list_collections = milvus.list_collections(with_info=True)
    return Response({"message": list_collections})
# jango-ninja

@extend_schema(request=CollectionCreateSerializer)
@api_view(["POST"])
def add_collections(request):
    serializer = CollectionCreateSerializer(data=request.data)

    if not serializer.is_valid():
        return Response({"error": "Validation failed", "details": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    collection_name = serializer.validated_data["collection_name"]
    dimension = serializer.validated_data.get("dimension", 768)
    description = serializer.validated_data.get("description", "Коллекция документов для семантического поиска")
    metric_type = serializer.validated_data.get("metric_type", "COSINE")
    list_collections = milvus.list_collections()
    if collection_name in list_collections:
        return Response({"message": "Коллекция уже существует", "collection_name": collection_name}, status=status.HTTP_200_OK)

    milvus.create_collection(collection_name=collection_name, dimension=dimension, description=description, metric_type=metric_type)
    return Response(
        {
            "message": "Collection created successfully",
            "collection_name": collection_name,
            "dimension": dimension,
            "description": description,
            "metric_type": metric_type,
        },
        status=status.HTTP_201_CREATED,
    )


@extend_schema(
    parameters=[
        OpenApiParameter(
            name="collection_name",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="Название коллекции для удаления",
            required=True,
        )
    ],
    description="Удаление коллекции из Milvus",
)
@api_view(["DELETE"])
def delete_collections(request, collection_name):
    """Удаление коллекции по имени"""
    list_collections = milvus.list_collections()
    if collection_name not in list_collections:
        return Response(
            {"message": "Коллекция не существует", "collection_name": collection_name, "error": "Not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    milvus.delete_collection(collection_name)
    return Response(
        {
            "message": "Коллекция успешно удалена",
            "collection_name": collection_name,
            "status": "deleted",
        },
        status=status.HTTP_200_OK,
    )


@extend_schema(request=TextAddSerializer)
@api_view(["POST"])
def add_text(request):
    serializer = TextAddSerializer(data=request.data)
    if not serializer.is_valid():
        return Response({"error": "Validation failed", "details": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    collection_name = serializer.validated_data["collection_name"]
    text = serializer.validated_data["text"]

    list_collections = milvus.list_collections()
    if collection_name not in list_collections:
        return Response(
            {"message": "Коллекция не существует", "collection_name": collection_name, "error": "Not found"},
            status=status.HTTP_404_NOT_FOUND,
        )
    result = processor.process_text(text, collection_name)
    if result["success"]:
        return Response(
            {
                "message": "Текст успешно обработан",
                "statistics": {
                    "chunks_count": result["chunks_count"],
                    "inserted_records": len(result.get("inserted_ids", [])),
                    "collection_name": collection_name,
                },
            },
            status=status.HTTP_200_OK,
        )
    else:
        return Response(
            {"message": "Ошибка при обработке текста", "error": result.get("error", "Неизвестная ошибка"), "collection_name": collection_name},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,  # Исправлено: было 200
        )


@extend_schema(request=SearchQuerySerializer)
@api_view(["POST"])
def search_query(request):
    serializer = SearchQuerySerializer(data=request.data)
    if not serializer.is_valid():
        return Response({"error": "Validation failed", "details": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    collection_name = serializer.validated_data["collection_name"]
    query = serializer.validated_data["query"]

    list_collections = milvus.list_collections()
    if collection_name not in list_collections:
        return Response(
            {"message": "Коллекция не существует", "collection_name": collection_name, "error": "Not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    query_embedding = embedder.encode_query(query)

    search_results = milvus.search(collection_name=collection_name, query_vectors=[query_embedding], top_k=3)

    answer = {"Найдено результатов": len(search_results[0])}
    results = []
    for i, hit in enumerate(search_results[0], 1):
        results.append(
            {
                "Distance": round(hit["distance"], 4),
                "Файл": hit.get("file_name", "N/A"),
                "Чанк": hit.get("chunk_index", -1),
                "Текст": hit["text"][:150],
            }
        )
    answer["result"] = results
    return Response(answer)


@api_view(["GET"])
def get_collection_info(request, collection_name):
    list_collections = milvus.list_collections()
    if collection_name not in list_collections:
        return Response(
            {"message": "Коллекция не существует", "collection_name": collection_name, "error": "Not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    info = milvus.get_collection_info(collection_name)
    print(info)
    parsed = {
        "exists": info["exists"],
        "num_entities": info["num_entities"],
        "schema": {
            "auto_id": info["schema"].auto_id,
            "description": info["schema"].description,
            "enable_dynamic_field": info["schema"].enable_dynamic_field,
            "fields": [
                {
                    "name": field.name,
                    "description": field.description,
                    "type": str(field.dtype),
                    "is_primary": field.is_primary,
                    "auto_id": getattr(field, "auto_id", False),
                    **({"params": field.params} if hasattr(field, "params") and field.params else {}),
                }
                for field in info["schema"].fields
            ],
        },
    }
    return Response(parsed)
