from django.core.management.base import BaseCommand
from django.conf import settings
from milvus_client import MilvusClient
from embedder import Embedder
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Инициализация подключения к Milvus"

    def handle(self, *args, **options):
        self.stdout.write("Инициализация Milvus...")

        try:
            # Инициализация Milvus клиента
            milvus_client = MilvusClient(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

            self.stdout.write(self.style.SUCCESS(f"Успешное подключение к Milvus ({settings.MILVUS_HOST}:{settings.MILVUS_PORT})"))

            # Инициализация эмбеддера
            try:
                embedder = Embedder(model_name=settings.EMBEDDING_MODEL_PATH)
                self.stdout.write(self.style.SUCCESS(f"Эмбеддер инициализирован с моделью: {settings.EMBEDDING_MODEL_PATH}"))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Не удалось инициализировать эмбеддер: {e}"))

            # Проверка существующих коллекций
            from pymilvus import utility

            collections = utility.list_collections()

            if collections:
                self.stdout.write("Существующие коллекции:")
                for col in collections:
                    info = milvus_client.get_collection_info(col)
                    if info.get("exists"):
                        self.stdout.write(f'  - {col}: {info.get("num_entities", 0)} записей')
            else:
                self.stdout.write("Коллекции не найдены")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Ошибка инициализации: {e}"))
