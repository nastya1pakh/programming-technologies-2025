from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .embedder import Embedder
from .milvus_client import MilvusClient


class SearchView(APIView):
    def post(self, request):
        query = request.data.get("query", "")
        if not query:
            return Response({"error": "Query parameter is required."}, status=400)

        try:
            embedder = Embedder()
            query_embedding = embedder.encode_query(query)
        except Exception as e:
            return Response({"error": str(e)}, status=500)

        try:
            collection_name = 'documents'
            milvus = MilvusClient(host="standalone", port=19530)
            search_results = milvus.search(
                collection_name=collection_name,
                query_vectors=[query_embedding],
                top_k=5
            )

            formatted_results = []
            # search_results — список списков (по одному на query vector)
            for i, hit in enumerate(search_results[0], 1):
                formatted_results.append({
                    "id": hit["id"],
                    "distance": hit['distance'],
                    "file": hit.get('file_name', 'N/A'),
                    "chunk": hit.get('chunk_index', -1),
                    "text": hit['text'][:150]
                })

            return Response({"results": formatted_results}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
