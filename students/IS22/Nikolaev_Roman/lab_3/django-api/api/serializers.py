from rest_framework import serializers


class SearchRequestSerializer(serializers.Serializer):
    query = serializers.CharField(
        required=True, help_text="Текст запроса для семантического поиска"
    )
    top_k = serializers.IntegerField(
        required=False,
        default=3,
        min_value=1,
        help_text="Количество результатов (по умолчанию 3)",
    )

class DocumentReconstructRequestSerializer(serializers.Serializer):
    file_name = serializers.CharField(
        required=True, 
        help_text="Полный путь к файлу-источнику для восстановления его текста"
    )
    collection_name = serializers.CharField(
        required=False,
        help_text="Имя коллекции (если не указано, используется значение по умолчанию)",
    )