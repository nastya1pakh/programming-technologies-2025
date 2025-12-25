# Лабораторная работа №3. Векторные базы данных и семантический поиск

## Цель

Целью работы было научиться использовать векторные базы данных и семантический поиск для обработки текстовых данных и потенциальной интеграции с LLM (Large Language Models).

## План

1. Настройка окружения;
2. Создание модуля для работы с Milvus;
3. Парсинг текстовых файлов;
4. Задания.

## 1. Настройка окружения

Для выполнения работы был использован **devcontainer**, конфиг которого был размещен по пути `[assets/lab3/.devcontainer/devcontainer.json](../assets/lab3/.devcontainer/devcontainer.json)`.

**Devcontainer** представлял собой контейнер, включающий все необходимые компоненты для работы, такие как:

* Python 3.12;
* Milvus;
* Attu.

Контейнер позволил работать с проектом в изолированном окружении, что предотвратило конфликты зависимостей и упростило процесс настройки окружения. Для его запуска было использовано контекстное меню в **VSCode** (Ctrl+Shift+P) с опцией "Reopen in Container". Установленное расширение "Dev Containers" обеспечило нужную среду.

Контейнеры оказались достаточно тяжелыми, и их загрузка заняла более 2 ГБ данных. Конфиг был подготовлен для использования CPU-версии эмбеддера, и для работы с GPU-версией потребовались изменения в конфиге.

## 2. Создание векторной базы данных

После настройки окружения был запущен контейнер с работающим **Milvus** и **Attu**. Для начала был инициализирован модуль для работы с **Milvus**, который установил подключение к серверу:

```python
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

class MilvusClient:
    def __init__(self, host: str = "standalone", port: int = 19530, alias: str = "default"):
        self.host = host
        self.port = port
        self.alias = alias
        self._connect()
    
    def _connect(self):
        try:
            connections.connect(host=self.host, port=self.port)
            print(f"Подключение к Milvus установлено ({self.host}:{self.port})")
        except Exception as e:
            print(f"Ошибка подключения к Milvus: {e}")
            raise
```

Метод для подключения был написан, и подключение к **Milvus** было успешно установлено.

Далее был реализован метод для создания коллекции в базе данных с параметрами: имя коллекции, размерность векторов, описание коллекции и тип метрики:

```python
def create_collection(self, collection_name: str, dimension: int, metric_type: str = "COSINE"):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": metric_type, "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    return collection
```

Этот метод создал коллекцию в **Milvus** с указанными параметрами и индексом для векторного поля. После этого была реализована функция для вставки данных в коллекцию:

```python
def insert_data(self, collection_name: str, texts: List[str], embeddings: List[List[float]]):
    collection = Collection(collection_name)
    collection.insert([texts, embeddings])
    collection.flush()
```

Метод позволил вставлять данные в коллекцию, после проверки на наличие коллекции и соответствие размерности эмбеддингов.

Также был реализован метод для поиска похожих векторов в коллекции:

```python
def search(self, collection_name: str, query_vectors: List[List[float]], top_k: int = 5):
    collection = Collection(collection_name)
    results = collection.search(query_vectors, anns_field="embedding", limit=top_k)
    return results
```

Поиск выполнялся с использованием метрики **cosine similarity**, но также была предусмотрена возможность использования других метрик, таких как **L2** или **IP**.

## 3. Парсинг текстовых файлов

После настройки работы с **Milvus** был создан класс **TextParser** для парсинга текстовых файлов с разбиением на чанки.

```python
import re
import os
from typing import List

class TextParser:
    """Парсер текстовых файлов с разбиением на чанки."""
    
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 64):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap
```

Класс позволил нормализовать текст, удаляя лишние пробелы и переносы строк:

```python
def normalize_text(self, text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

Текст делился на чанки с перекрытием, что позволяло сохранить контекст между частями текста:

```python
def chunk_text(self, text: str) -> List[str]:
    text = self.normalize_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + self.chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += self.step_size
    return chunks
```

Метод разделил текст на чанки, что обеспечило эффективную работу с большими объемами данных и контекстуальность каждого чанка.

После парсинга текстов был использован метод для обработки файлов и их преобразования в чанки:

```python
def parse_file(self, file_path: str, encoding: str = "utf-8") -> List[str]:
    text = self.read_file(file_path, encoding)
    chunks = self.chunk_text(text)
    return chunks
```

Метод вернул список чанков из текстового файла, готовых для загрузки в **Milvus**.

![запустил example usage](./screens/запустил%20example%20usage.png)

Также после настройки и запуска всех компонентов, через localhost:8000 открылся веб-интерфейс Attu для Milvus. Attu — это графический интерфейс для работы с Milvus, который позволяет выполнять различные операции с базой данных, такие как создание коллекций, управление данными и поиск по векторным представлениям.


Нашел коллецию documents, которая была создана с помощью кода.
Перешел в раздел "Data", где были отображены доступные чанки.

Выполнил поиск, задав параметры для поиска.

![search web attu](./screens/search%20web%20attu.png)

Интерфейс Attu оказался удобным для визуального взаимодействия с базой данных Milvus и эффективного выполнения поиска.

## 4. Задания

### 1. **Запуск сценария с использованием GPU**. 

**Изменение конфигурации Docker для использования GPU**

В уже существующем файле конфигурации **docker-compose.yaml** была добавлена строка `gpus: all`, чтобы использовать все доступные графические процессоры для контейнера. Это было сделано в разделе, отвечающем за контейнер **app** (который включает приложение для работы с **Milvus**).

```yaml
version: '3.8'

services:
  app:
    build:
      context: ..
      dockerfile: .devcontainer/Dockerfile
    container_name: milvus-lab-app
    working_dir: /workspaces
    volumes:
      - ../:/workspaces
    command: sleep infinity
    gpus: all  # Строка для использования GPU
    depends_on:
      - standalone
    networks:
      - internal-network
```

Эта строка гарантирует, что контейнер будет использовать все доступные графические процессоры в системе. Она добавлена под параметром **gpus**, что позволяет задействовать GPU для вычислений, например, при генерации **embeddings**.

**Дополнительные установки для работы с GPU**

Для уверенности в корректной работе с GPU, я также установил драйвера **NVIDIA Studio**. Эти драйвера необходимы для поддержки **CUDA** и работы с графическими процессорами в контейнерах **Docker**. Установка драйверов гарантирует, что система будет правильно использовать GPU для выполнения вычислений, таких как генерация embeddings в Milvus, с соответствующим ускорением на графическом процессоре.

**Перезапуск контейнеров**

После внесения изменений в конфигурацию контейнера, был выполнен перезапуск всех сервисов с помощью команды:

```bash
>Dev Containers: Rebuild Container
```

**Проверка работы GPU**

После перезапуска контейнера и настройки GPU, были выполнены тесты с использованием **CUDA**. Для этого был запущен сценарий с использованием **GPU**, в котором были обрабатывались данные для **embedding** с ускорением на графическом процессоре.

![запустил example usage gpu](./screens/запустил%20example%20usage%20gpu.png)

Таким образом, все необходимые шаги для включения GPU в конфигурацию и работы с ним были выполнены корректно, и система теперь использует графический процессор для вычислений, что ускоряет обработку векторных данных.

### 2. **Создание API для работы с Milvus**. 

**Создание проекта Django**

Первым шагом было создание нового проекта Django. Для этого была использована команда:

```bash
django-admin startproject milvus_api
```

Затем был создан новое приложение, в котором мы разместили логику для работы с **Milvus**:

```bash
python manage.py startapp search
```

**Подключение Django Rest Framework**

Для работы с **Django Rest Framework (DRF)**, который облегчил создание RESTful API, был добавлен в проект с помощью pip:

```bash
pip install djangorestframework
```

Далее, в файле **settings.py** в **INSTALLED_APPS** добавлен `rest_framework`:

```python
INSTALLED_APPS = [
    # другие приложения,
    'rest_framework',
]
```

**Создание `views.py` для обработки запроса**

В файле **views.py** было реализовано API для обработки запросов на поиск по векторной базе данных **Milvus**. Для этого был использован класс **APIView** из **Django Rest Framework**.

Основная логика обработки запроса заключается в следующем:

* Получение запроса от пользователя (поиск по тексту).
* Генерация **embedding** для текста с помощью модели **multilingual-e5-base**.
* Подключение к **Milvus** и выполнение поиска по векторному представлению текста.
* Формирование ответа с результатами поиска.

Пример кода в **views.py**:

```python
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
```

**Создание сериализатора в `serializers.py`**

Для обработки входных данных в виде векторов и получения правильного формата ответа был создан сериализатор в **serializers.py**. Это позволит валидировать запросы и ответы от API.

Пример кода для сериализатора:

```python
from rest_framework import serializers

class SearchSerializer(serializers.Serializer):
    query_vectors = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    top_k = serializers.IntegerField(default=5)
    expr = serializers.CharField(required=False, allow_blank=True)
```

**Создание маршрутов в `urls.py`**

Для подключения нашего API и создания маршрута, который будет обрабатывать запросы на поиск, был создан файл **urls.py** в приложении **search**:

```python
from django.urls import path
from .views import SearchView

urlpatterns = [
    path('search/', SearchView.as_view(), name='search'),
]
```

**Подключение URL-ов приложения к основному проекту**

В файле **urls.py** проекта был подключен URL-минимум для нашего приложения **search**:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('search.urls')),  # Подключение маршрутов из приложения search
]
```
**Проверка через Thunder Client в VS Code**

После реализации API была выполнена проверка с помощью **Thunder Client** в **VS Code**. Был отправлен POST-запрос с телом:

![search complete](./screens/search%20complete.png)

---

### Вывод:

В ходе выполнения лабораторной работы была успешно настроена и использована система для работы с векторными базами данных на основе Milvus и Attu. Через веб-интерфейс Attu был выполнен поиск по коллекциям Milvus, что подтвердило корректную работу системы. Все этапы, включая настройку окружения, создание коллекций и работу с данными, прошли успешно. Использование GPU для ускоренной обработки данных. Также был реализован API с помощью которого можно было взаимодейство с системой через запросы. В результате была реализована эффективная система для работы с векторными данными и их семантическим поиском.

