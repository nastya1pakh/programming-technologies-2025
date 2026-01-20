"""
Пример использования всей системы: парсинг, embeddings, Milvus.
"""

from milvus_client import MilvusClient
from document_processor import DocumentProcessor
from embedder import create_embedding_function


def main():
    """Основная функция с примером использования."""
    
    model_name=r"C:\stady\prog-tech\lab3\multilingual-e5-base"
    
    
    # 1. Инициализация Milvus клиента
    print("=" * 60)
    print("1. Подключение к Milvus")
    print("=" * 60)
    milvus = MilvusClient(host="localhost", port=19530)
    
    # 2. Создание функции для генерации embeddings
    print("\n" + "=" * 60)
    print("2. Инициализация эмбеддера (multilingual-e5-base)")
    print("=" * 60)
    embedding_func = create_embedding_function(
        model_name=model_name,
        batch_size=32
    )
    
    # 3. Создание обработчика документов
    print("\n" + "=" * 60)
    print("3. Создание обработчика документов")
    print("=" * 60)
    processor = DocumentProcessor(
        milvus_client=milvus,
        chunk_size=256,
        chunk_overlap=64,
        embedding_function=embedding_func
    )
    
    # 4. Создание коллекции в Milvus
    print("\n" + "=" * 60)
    print("4. Создание коллекции в Milvus")
    print("=" * 60)
    collection_name = "documents"
    
    # Удаляем коллекцию если существует (для чистого примера)
    milvus.delete_collection(collection_name)
    
    # Создаем новую коллекцию
    milvus.create_collection(
        collection_name=collection_name,
        dimension=768,  # multilingual-e5-base
        description="Коллекция документов для семантического поиска",
        metric_type="COSINE"
    )
    
    # 5. Обработка файлов из директории /files
    print("\n" + "=" * 60)
    print("5. Обработка файлов из директории /files")
    print("=" * 60)
    
    import os
    files_dir = "files"
    if os.path.exists(files_dir):
        txt_files = [f for f in os.listdir(files_dir) if f.endswith('.txt')]
        print(f"Найдено файлов: {len(txt_files)}")
        
        for txt_file in txt_files:
            file_path = os.path.join(files_dir, txt_file)
            print(f"\nОбработка: {txt_file}")
            result = processor.process_file(file_path, collection_name)
            
            if result['success']:
                print(f"  Успешно: {result['chunks_count']} чанков загружено")
            else:
                print(f"  Ошибка: {result.get('error', 'Неизвестная ошибка')}")
    else:
        print(f"Директория {files_dir} не найдена, используем тестовый текст")
        sample_text = """
        Python - это высокоуровневый язык программирования общего назначения.
        Машинное обучение использует алгоритмы для анализа данных.
        Векторные базы данных хранят embeddings для семантического поиска.
        """
        result = processor.process_text(sample_text, collection_name)
        
        if result['success']:
            print(f"\nУспешно обработано!")
            print(f"  Чанков: {result['chunks_count']}")
            print(f"  Вставлено записей: {len(result.get('inserted_ids', []))}")
        else:
            print(f"\nОшибка: {result.get('error', 'Неизвестная ошибка')}")
            return
    
    # 6. Поиск в коллекции
    print("\n" + "=" * 60)
    print("6. Семантический поиск")
    print("=" * 60)
    
    from embedder import Embedder
    embedder = Embedder(model_name=model_name)
    
    # Генерируем embedding для запроса
    query = "Что такое машинное обучение?"
    print(f"Запрос: '{query}'")
    query_embedding = embedder.encode_query(query)
    
    # Ищем похожие документы
    search_results = milvus.search(
        collection_name=collection_name,
        query_vectors=[query_embedding],
        top_k=3
    )
    
    print(f"\nНайдено результатов: {len(search_results[0])}")
    for i, hit in enumerate(search_results[0], 1):
        print(f"\n{i}. (Distance: {hit['distance']:.4f})")
        print(f"   Файл: {hit.get('file_name', 'N/A')}")
        print(f"   Чанк #{hit.get('chunk_index', -1)}")
        print(f"   Текст: {hit['text'][:150]}...")
    
    # 8. Пример получения исходного документа по найденному чанку
    if search_results[0]:
        first_hit = search_results[0][0]
        file_path = first_hit.get('file_path')

        if file_path:
            # 🔧 FIX: Milvus не понимает обратные слэши
            normalized_path = file_path.replace("\\", "/")

            print("\n" + "=" * 60)
            print("8. Получение всех чанков исходного документа")
            print("=" * 60)
            print(f"Файл: {first_hit.get('file_name', 'N/A')}")

            try:
                document_chunks = milvus.get_document_chunks(
                    collection_name,
                    normalized_path
                )

                print(f"Найдено чанков в документе: {len(document_chunks)}")
                print("\nПервые 3 чанка из документа:")

                for i, chunk in enumerate(document_chunks[:3], 1):
                    print(f"\n  Чанк {i} (индекс {chunk.get('chunk_index', -1)}):")
                    print(f"  {chunk['text'][:100]}...")

            except Exception as e:
                print(f"Ошибка при получении чанков: {e}")

    # 7. Информация о коллекции
    print("\n" + "=" * 60)
    print("7. Информация о коллекции")
    print("=" * 60)
    info = milvus.get_collection_info(collection_name)
    if info['exists']:
        print(f"Записей в коллекции: {info['num_entities']}")
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)


if __name__ == "__main__":
    main()

