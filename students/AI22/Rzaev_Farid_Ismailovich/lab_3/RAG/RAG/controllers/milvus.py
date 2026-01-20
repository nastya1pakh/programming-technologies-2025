from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import action
from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from io import BytesIO

from RAG.clients.embedder import create_embedding_function
from RAG.clients.milvus_client import MilvusClient
from RAG.clients.minio import MinioService
from RAG.clients.document_processor import DocumentProcessor

milvus = MilvusClient()


minio_service = MinioService(
    host="minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
processor = DocumentProcessor(
        milvus_client=milvus,
        chunk_size=256,
        chunk_overlap=64,
        embedding_function=create_embedding_function()
    )
    

class MilvusController(ViewSet):
    parser_classes = [MultiPartParser, FormParser]  
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                name="collection_name",
                default="default",
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_STRING,
                required=False,
                description="Имя коллекции"
            ),
            openapi.Parameter(
                name="alias",
                default="default",
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_STRING,
                required=False,
                description="Имя коллекции"
            )
        ],
        responses={200: openapi.Response("Success")},
        tags=['milvus']
    )
    @action(detail=False, methods=["get"], url_path="collection/data")
    def get_data(self, request):
        collection_name = request.query_params.get("collection_name")
        alias = request.query_params.get("alias")
        try:
            milvus._connect(alias)
            if (alias == None):
                return Response(status=403)
            data = milvus.get_data(collection_name)
        finally:
            milvus.disconnect()
        return Response(data)

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                name="alias",
                default="default",
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_STRING,
                required=False,
                description="Имя коллекции"
            ),
            openapi.Parameter(
                name="collection_name",
                default="default",
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_STRING,
                required=False,
                description="Имя коллекции"
            ),
            openapi.Parameter(
                name="id",
                default="default",
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_NUMBER,
                required=False,
                description="Имя коллекции"
            ),
        ],
        responses={200: openapi.Response("Success")},
        tags=['milvus']
    )
    @action(detail=False, methods=["get"], url_path="collection/id")
    def get_data_by_id(self, request):
        collection_name = request.query_params.get("collection_name")
        alias = request.query_params.get("alias")
        id = request.query_params.get("id")
        try:
            milvus._connect(alias)
            if (alias == None):
                return Response(status=403)
            data = milvus.get_data_id(collection_name, id)
        finally:
            milvus.disconnect()
        return Response(data)

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                name="alias",
                default="default",
                in_=openapi.IN_QUERY,
                type=openapi.TYPE_STRING,
                required=False,
                description="Имя коллекции"
            )
        ],
        responses={200: openapi.Response("Success")},
        tags=['milvus']
    )
    @action(detail=False, methods=["get"], url_path="collections")
    def get_collections(self, request):
        try:

            alias = request.query_params.get("alias")
            milvus._connect(alias)
            if (alias == None):
                return Response(status=403)
            data = milvus.get_collections()
            print(data)
        finally:
            milvus.disconnect()
        return Response(data)

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(name="alias", in_=openapi.IN_QUERY,
                              type=openapi.TYPE_STRING, required=True, default="default"),
            openapi.Parameter(name="collection", in_=openapi.IN_QUERY,
                              type=openapi.TYPE_STRING, required=True),
            openapi.Parameter(name="query", in_=openapi.IN_QUERY,
                              type=openapi.TYPE_STRING, required=True),
            openapi.Parameter(name="top_k", in_=openapi.IN_QUERY,
                              type=openapi.TYPE_INTEGER, required=False, default=5)
        ],
        responses={200: openapi.Response("Search results")},
        tags=['milvus']
    )
    @action(detail=False, methods=["get"], url_path="search")
    def search(self, request):
        alias = request.query_params.get("alias")
        collection_name = request.query_params.get("collection")
        query_text = request.query_params.get("query")
        top_k = int(request.query_params.get("top_k", 5))
        if not collection_name or not query_text:
            return Response({"error": "collection и query обязательны"})

        try:
            milvus._connect(alias)
            embedder = create_embedding_function()
            query_embedding = embedder([query_text])[0]
            results = milvus.search(
                collection_name, [query_embedding], top_k=top_k)
        finally:
            milvus.disconnect()
        return Response(results[0])

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                name="alias",
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=True,
                description="Имя коллекции"
            ),
            openapi.Parameter(
                name="collection",
                in_=openapi.IN_FORM,
                type=openapi.TYPE_STRING,
                required=True,
                description="Имя коллекции"
            ),
            openapi.Parameter(
                name="file",
                in_=openapi.IN_FORM,
                type=openapi.TYPE_FILE,
                required=True,
                description="Файл коллекции"
            )
        ],
        responses={200: openapi.Response("Success")},
        tags=["milvus"]
    )
    @action(detail=False, methods=["post"], url_path="collections/add")
    def add_collection(self, request):
        data = request.data
        alias = data.get("alias")
        collection_name=data.get("collection")
        file = request.FILES.get("file")

        if not alias:
            return Response(
                {"error": "alias is required"},
                status=400
            )

        if not file:
            return Response(
                {"error": "file is required"},
                status=400
            )

        file.seek(0)
        file_bytes = file.read()

        object_name = minio_service.upload_file_RAM(
            file_name=file.name,
            mime_type=file.content_type,
            file_stream=BytesIO(file_bytes))

        processor.process_text(file_bytes.decode("utf-8"),collection_name)

        return Response(
            {
                "alias": alias,
                "file": object_name,
            },)
    
    @swagger_auto_schema(
    manual_parameters=[
        openapi.Parameter(
            name="alias",
            in_=openapi.IN_FORM,
            type=openapi.TYPE_STRING,
            required=True
        ),
        openapi.Parameter(
            name="collection",
            in_=openapi.IN_FORM,
            type=openapi.TYPE_STRING,
            required=True
        )
    ],
    responses={200: openapi.Response("Success")},
    tags=["milvus"]
    )
    @action(detail=False, methods=["delete"], url_path="collections/delete", parser_classes=[FormParser])
    def delete_collection(self, request):
        alias = request.data.get("alias")
        collection_name = request.data.get("collection")
        try:
            milvus._connect(alias)
            milvus.delete_data(collection_name)
        finally:
            milvus.disconnect()
        return Response({"status": "deleted"})