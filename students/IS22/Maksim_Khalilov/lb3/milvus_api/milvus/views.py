from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .embedder import Embedder
from .milvus_client import MilvusClient

class SearchView(APIView):
    def post(self, request, *args, **kwargs):
        query = request.data.get("query", "")
        
        if not query:
            return Response({"error": "Запрос не может быть пустым"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            embedder = Embedder(model_name="intfloat/multilingual-e5-base")
            query_embedding = embedder.encode_query(query)
        except Exception as e:
            return Response({"error": f"Ошибка при генерации embedding для запроса: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        try:
            milvus = MilvusClient(host="standalone", port=19530) 
        except Exception as e:
            return Response({"error": f"Ошибка при подключении к Milvus: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        collection_name = "documents" 
        try:
            search_results = milvus.search(
                collection_name=collection_name,
                query_vectors=[query_embedding],
                top_k=3 
            )
            
            print(search_results)

            formatted_results = []
            for hit in search_results[0]: 
                formatted_results.append({
                    "distance": hit.get("distance", 0), 
                    "text": hit.get("text", "Нет текста"), 
                    "file_name": hit.get("file_name", "N/A"), 
                    "chunk_index": hit.get("chunk_index", -1)  
                })

            return Response({"results": formatted_results}, status=status.HTTP_200_OK)

        
        except Exception as e:
            return Response({"error": f"Ошибка при выполнении поиска в Milvus: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)