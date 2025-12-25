from rest_framework import serializers


class CreateCollectionSerializer(serializers.Serializer):
    name = serializers.CharField()
    dimension = serializers.IntegerField(default=768)
    metric_type = serializers.ChoiceField(
        choices=["COSINE", "L2", "IP"],
        default="COSINE"
    )
    force_delete = serializers.BooleanField(
        default=False)


class UploadDocumentsSerializer(serializers.Serializer):
    collection_name = serializers.CharField()
    directory = serializers.CharField(required=False)
    text = serializers.CharField(required=False)


class SearchSerializer(serializers.Serializer):
    collection_name = serializers.CharField()
    query = serializers.CharField()
    top_k = serializers.IntegerField(default=3)
