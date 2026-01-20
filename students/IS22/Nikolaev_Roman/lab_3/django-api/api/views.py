from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import DocumentReconstructRequestSerializer, SearchRequestSerializer
from .services import COLLECTION_NAME, semantic_search, milvus_client


class SearchAPIView(APIView):
    @swagger_auto_schema(request_body=SearchRequestSerializer)
    def post(self, request):
        serializer = SearchRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query = serializer.validated_data["query"]
        top_k = serializer.validated_data.get("top_k", 3)

        results = semantic_search(query=query, top_k=top_k)

        return Response(results, status=status.HTTP_200_OK)

class CollectionInfoAPIView(APIView):
    def get(self, request, name):
        info = milvus_client.get_collection_info(name)
        
        if not info.get("exists"):
            return Response(
                {"error": f"Collection '{name}' not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if "error" in info:
            return Response(
                {"error": info["error"]}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        formatted_info = {
            "exists": info["exists"],
            "num_entities": info["num_entities"],
            "schema": {
                "description": info["schema"].description,
                "fields": [
                    {
                        "name": f.name,
                        "type": str(f.dtype),
                        "description": f.description,
                        "params": f.params
                    } for f in info["schema"].fields
                ]
            },
            "indexes": [
                {
                    "field_name": idx.field_name,
                    "index_name": idx.index_name,
                    "params": idx.params
                } for idx in info["indexes"]
            ]
        }
        
        return Response(formatted_info, status=status.HTTP_200_OK)

class DocumentReconstructAPIView(APIView):
    @swagger_auto_schema(query_serializer=DocumentReconstructRequestSerializer)
    def get(self, request):
        serializer = DocumentReconstructRequestSerializer(data=request.query_params)
        
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        file_path = '/workspaces/files/' + serializer.validated_data['file_name']
        collection = serializer.validated_data.get('collection_name') or COLLECTION_NAME

        try:
            full_text = milvus_client.reconstruct_document(
                collection_name=collection, 
                file_path=file_path
            )

            if not full_text:
                return Response(
                    {"error": "Документ не найден"}, 
                    status=status.HTTP_404_NOT_FOUND
                )

            return Response({
                "file_path": file_path,
                "full_text": full_text
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )