from __future__ import annotations

import os
import tempfile

from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import (
    CreateCollectionSerializer,
    DocumentChunksQuerySerializer,
    IngestFileSerializer,
    IngestTextSerializer,
    ReconstructQuerySerializer,
    SearchSerializer,
)
from .services import get_embedder_handle, get_milvus_client, make_document_processor, milvus_list_collections


class HealthView(APIView):
    @extend_schema(operation_id="health_get", responses={200: {"type": "object"}})
    def get(self, request):
        from django.conf import settings

        try:
            # Простая проверка, что соединение живое
            _ = milvus_list_collections()
            return Response(
                {
                    "status": "ok",
                    "milvus_host": settings.MILVUS_HOST,
                    "milvus_port": settings.MILVUS_PORT,
                    "milvus_connected": True,
                }
            )
        except Exception as e:
            return Response(
                {
                    "status": "degraded",
                    "milvus_host": settings.MILVUS_HOST,
                    "milvus_port": settings.MILVUS_PORT,
                    "milvus_connected": False,
                    "error": str(e),
                },
                status=status.HTTP_200_OK,
            )


class CollectionsView(APIView):
    @extend_schema(operation_id="collections_list", responses={200: {"type": "object"}})
    def get(self, request):
        """Список коллекций."""
        try:
            cols = milvus_list_collections()
            return Response({"collections": cols})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @extend_schema(operation_id="collections_create", request=CreateCollectionSerializer, responses={200: {"type": "object"}})
    def post(self, request):
        """Создать коллекцию."""
        ser = CreateCollectionSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        collection_name: str = data["collection_name"]
        description: str = data.get("description", "")
        metric_type: str = data.get("metric_type", "COSINE")
        model_name = data.get("model_name")

        dimension = data.get("dimension")
        if dimension is None:
            try:
                dimension = get_embedder_handle(model_name=model_name).dimension()
            except Exception as e:
                return Response(
                    {"error": f"Не удалось определить размерность модели. Укажите dimension явно. Ошибка: {e}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        try:
            client = get_milvus_client()
            client.create_collection(
                collection_name=collection_name,
                dimension=int(dimension),
                description=description,
                metric_type=metric_type,
            )
            info = client.get_collection_info(collection_name)
            return Response(info)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CollectionDetailView(APIView):
    @extend_schema(operation_id="collection_get_info", responses={200: {"type": "object"}})
    def get(self, request, collection_name: str):
        try:
            info = get_milvus_client().get_collection_info(collection_name)
            return Response(info)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @extend_schema(operation_id="collection_delete", responses={200: {"type": "object"}})
    def delete(self, request, collection_name: str):
        try:
            get_milvus_client().delete_collection(collection_name)
            return Response({"deleted": True, "collection_name": collection_name})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class IngestTextView(APIView):
    @extend_schema(operation_id="ingest_text", request=IngestTextSerializer, responses={200: {"type": "object"}})
    def post(self, request):
        ser = IngestTextSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        processor = make_document_processor(
            chunk_size=data.get("chunk_size"),
            chunk_overlap=data.get("chunk_overlap"),
            model_name=data.get("model_name"),
            batch_size=data.get("batch_size"),
        )
        result = processor.process_text(text=data["text"], collection_name=data["collection_name"])
        return Response(result, status=status.HTTP_200_OK if result.get("success") else status.HTTP_400_BAD_REQUEST)


class IngestFileView(APIView):
    @extend_schema(operation_id="ingest_file", request=IngestFileSerializer, responses={200: {"type": "object"}})
    def post(self, request):
        ser = IngestFileSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        uploaded = data["file"]
        suffix = os.path.splitext(getattr(uploaded, "name", "") or "")[1] or ".txt"
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                for chunk in uploaded.chunks():
                    tmp.write(chunk)

            processor = make_document_processor(
                chunk_size=data.get("chunk_size"),
                chunk_overlap=data.get("chunk_overlap"),
                model_name=data.get("model_name"),
                batch_size=data.get("batch_size"),
            )
            result = processor.process_file(
                file_path=tmp_path,
                collection_name=data["collection_name"],
                encoding=data.get("encoding", "utf-8"),
            )
            return Response(
                result, status=status.HTTP_200_OK if result.get("success") else status.HTTP_400_BAD_REQUEST
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


class SearchView(APIView):
    @extend_schema(operation_id="semantic_search", request=SearchSerializer, responses={200: {"type": "object"}})
    def post(self, request):
        ser = SearchSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        handle = get_embedder_handle(model_name=data.get("model_name"))

        try:
            query_vec = handle.encode_query(data["query"])
            results = get_milvus_client().search(
                collection_name=data["collection_name"],
                query_vectors=[query_vec],
                top_k=data.get("top_k", 5),
                expr=(data.get("expr") or None),
            )
            return Response({"results": results})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DocumentChunksView(APIView):
    @extend_schema(
        operation_id="document_chunks_list",
        parameters=[
            OpenApiParameter(name="collection_name", required=True, type=str, location=OpenApiParameter.QUERY),
            OpenApiParameter(name="file_path", required=True, type=str, location=OpenApiParameter.QUERY),
            OpenApiParameter(name="order_by_index", required=False, type=bool, location=OpenApiParameter.QUERY),
        ],
        responses={200: {"type": "object"}},
    )
    def get(self, request):
        ser = DocumentChunksQuerySerializer(data=request.query_params)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        try:
            chunks = get_milvus_client().get_document_chunks(
                collection_name=data["collection_name"],
                file_path=data["file_path"],
                order_by_index=data.get("order_by_index", True),
            )
            return Response({"chunks": chunks})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ReconstructView(APIView):
    @extend_schema(
        operation_id="document_reconstruct",
        parameters=[
            OpenApiParameter(name="collection_name", required=True, type=str, location=OpenApiParameter.QUERY),
            OpenApiParameter(name="file_path", required=True, type=str, location=OpenApiParameter.QUERY),
        ],
        responses={200: {"type": "object"}},
    )
    def get(self, request):
        ser = ReconstructQuerySerializer(data=request.query_params)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        try:
            text = get_milvus_client().reconstruct_document(data["collection_name"], data["file_path"])
            return Response({"text": text})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

