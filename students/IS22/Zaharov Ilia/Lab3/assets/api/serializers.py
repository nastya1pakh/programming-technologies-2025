from __future__ import annotations

from rest_framework import serializers


class HealthResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    milvus_host = serializers.CharField()
    milvus_port = serializers.IntegerField()
    milvus_connected = serializers.BooleanField()
    error = serializers.CharField(required=False, allow_blank=True)


class CreateCollectionSerializer(serializers.Serializer):
    collection_name = serializers.CharField()
    dimension = serializers.IntegerField(required=False, min_value=1)
    description = serializers.CharField(required=False, allow_blank=True, default="")
    metric_type = serializers.ChoiceField(required=False, choices=["COSINE", "L2", "IP"], default="COSINE")
    model_name = serializers.CharField(required=False, allow_blank=True)


class CollectionInfoSerializer(serializers.Serializer):
    exists = serializers.BooleanField()
    num_entities = serializers.IntegerField(required=False)
    error = serializers.CharField(required=False, allow_blank=True)


class IngestTextSerializer(serializers.Serializer):
    collection_name = serializers.CharField()
    text = serializers.CharField()
    chunk_size = serializers.IntegerField(required=False, min_value=1)
    chunk_overlap = serializers.IntegerField(required=False, min_value=0)
    model_name = serializers.CharField(required=False, allow_blank=True)
    batch_size = serializers.IntegerField(required=False, min_value=1)


class IngestFileSerializer(serializers.Serializer):
    collection_name = serializers.CharField()
    file = serializers.FileField()
    encoding = serializers.CharField(required=False, default="utf-8")
    chunk_size = serializers.IntegerField(required=False, min_value=1)
    chunk_overlap = serializers.IntegerField(required=False, min_value=0)
    model_name = serializers.CharField(required=False, allow_blank=True)
    batch_size = serializers.IntegerField(required=False, min_value=1)


class SearchSerializer(serializers.Serializer):
    collection_name = serializers.CharField()
    query = serializers.CharField()
    top_k = serializers.IntegerField(required=False, min_value=1, default=5)
    expr = serializers.CharField(required=False, allow_blank=True)
    model_name = serializers.CharField(required=False, allow_blank=True)


class DocumentChunksQuerySerializer(serializers.Serializer):
    collection_name = serializers.CharField()
    file_path = serializers.CharField()
    order_by_index = serializers.BooleanField(required=False, default=True)


class ReconstructQuerySerializer(serializers.Serializer):
    collection_name = serializers.CharField()
    file_path = serializers.CharField()

