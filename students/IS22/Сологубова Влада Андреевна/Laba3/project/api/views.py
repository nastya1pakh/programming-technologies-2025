from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import CreateCollectionSerializer, UploadDocumentsSerializer, SearchSerializer
from .services import milvus_client, document_processor, embedder


class CreateCollectionView(APIView):
    """Создание коллекции в Milvus"""

    def post(self, request):
        serializer = CreateCollectionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        if data.get("force_delete"):
            milvus_client.delete_collection(data["name"])

        collection = milvus_client.create_collection(
            collection_name=data["name"],
            dimension=data["dimension"],
            metric_type=data["metric_type"]
        )

        return Response(
            {"message": f"Коллекция '{collection.name}' создана"},
            status=status.HTTP_201_CREATED
        )


class UploadDocumentsView(APIView):
    """Загрузка файлов или текста в Milvus"""

    def post(self, request):
        serializer = UploadDocumentsSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        collection_name = data["collection_name"]
        directory = data.get("directory")

        import os

        if directory and os.path.exists(directory):
            txt_files = [
                f for f in os.listdir(directory)
                if f.endswith(".txt")
            ]

            results = []

            for txt_file in txt_files:
                file_path = os.path.join(directory, txt_file)

                result = document_processor.process_file(
                    file_path=file_path,
                    collection_name=collection_name
                )

                results.append({
                    "file": txt_file,
                    "success": result["success"],
                    "chunks_count": result.get("chunks_count", 0),
                    "error": result.get("error")
                })

            return Response({
                "directory": directory,
                "files_found": len(txt_files),
                "results": results
            })

        sample_text = data.get("text")
        if sample_text:
            result = document_processor.process_text(
                text=sample_text,
                collection_name=collection_name
            )
            return Response(result)

        return Response(
            {
                "error": "Директория не найдена и текст не передан"
            },
            status=status.HTTP_400_BAD_REQUEST
        )


class SemanticSearchView(APIView):
    """Семантический поиск в Milvus"""

    def post(self, request):
        serializer = SearchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        query_vector = embedder.encode_query(data["query"])

        results = milvus_client.search(
            collection_name=data["collection_name"],
            query_vectors=[query_vector],
            top_k=data["top_k"]
        )

        return Response(results[0])


class DocumentChunksView(APIView):
    """Получение всех чанков документа по пути"""

    def get(self, request, collection_name, file_path):
        try:
            chunks = milvus_client.get_document_chunks(
                collection_name, file_path)
            return Response(chunks)
        except Exception as e:
            return Response({"error": str(e)}, status=400)


class CollectionInfoView(APIView):
    """Информация о коллекции Milvus"""

    def get(self, request, name):
        try:
            info = milvus_client.get_collection_info(name)

            if not info['exists']:
                return Response({"error": f"Коллекция '{name}' не найдена"}, status=404)

            schema = info.get('schema')
            if schema:
                fields = []
                for field in getattr(schema, "fields", []):
                    fields.append({
                        "name": field.name,
                        "dtype": str(field.dtype),
                        "is_primary": getattr(field, "is_primary", False),
                        "auto_id": getattr(field, "auto_id", False),
                        "max_length": getattr(field, "max_length", None),
                        "description": getattr(field, "description", "")
                    })
                info['schema'] = {
                    "description": getattr(schema, "description", ""),
                    "fields": fields
                }

            indexes = info.get('indexes', [])
            serializable_indexes = []
            for index in indexes:
                serializable_indexes.append({
                    "field_name": getattr(index, "field_name", ""),
                    "index_type": getattr(index, "index_type", ""),
                    "params": getattr(index, "params", {})
                })
            info['indexes'] = serializable_indexes

            return Response(info)

        except Exception as e:
            return Response({"error": str(e)}, status=400)
