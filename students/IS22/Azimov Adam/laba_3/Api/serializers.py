from rest_framework import serializers


class CollectionCreateSerializer(serializers.Serializer):
    """Сериализатор для создания коллекции."""

    collection_name = serializers.CharField(max_length=255, required=True)
    dimension = serializers.IntegerField(min_value=1, required=True)
    description = serializers.CharField(required=False, allow_blank=True)
    metric_type = serializers.ChoiceField(choices=["COSINE", "L2", "IP"], default="COSINE")


class CollectionDeleteSerializer(serializers.Serializer):
    """Сериализатор для удаления коллекции."""

    collection_name = serializers.CharField(max_length=255, required=True)


class SearchQuerySerializer(serializers.Serializer):
    """Сериализатор для поиска."""

    collection_name = serializers.CharField(max_length=255, required=True)
    query = serializers.CharField(max_length=255, required=True)


class TextAddSerializer(serializers.Serializer):
    """Сериализатор для создания коллекции."""

    collection_name = serializers.CharField(max_length=255, required=True)
    text = serializers.CharField(required=True)


class FileUploadSerializer(serializers.Serializer):
    """Сериализатор для загрузки файла."""

    collection_name = serializers.CharField(required=True)
    file = serializers.FileField(required=True)
    chunk_size = serializers.IntegerField(min_value=1, default=256)
    chunk_overlap = serializers.IntegerField(min_value=0, default=64)
