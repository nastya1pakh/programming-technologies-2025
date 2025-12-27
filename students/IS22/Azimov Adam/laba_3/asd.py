{'exists': True, 'num_entities': 122, 'schema': 
    {'auto_id': True, 'description': 'Коллекция документов для семантического поиска', 'fields':
        [{'name': 'id', 'description': 'Автоинкрементный ID', 'type': <DataType.INT64: 5>, 
          'is_primary': True, 'auto_id': True}, {'name': 'text', 'description': 'Исходный текст', 'type': <DataType.VARCHAR: 21>, 
                                                 'params': {'max_length': 65535}}, {'name': 'embedding', 'description': 'Векторное представление текста', 
                                                                                    'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 768}}, {'name': 'file_name', 'description': 'Название файла-источника',
                                                                                                                                                    'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 512}}, 
                                                                                    {'name': 'file_path', 'description': 'Полный путь к файлу-источнику', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 1024}}, 
                                                                                    {'name': 'chunk_index', 'description': 'Индекс чанка в документе (начинается с 0)', 'type': <DataType.INT64: 5>}], 'enable_dynamic_field': False}, 
    'indexes': [<pymilvus.orm.index.Index object at 0x000002860096A890>]}