import json
import unittest

from pymilvus import CollectionSchema, DataType, FieldSchema

from milvus_client import MilvusClient


class MilvusJsonSerializationTests(unittest.TestCase):
    def test_collection_schema_to_dict_is_json_serializable(self):
        """
        Регрессия: DRF падает с
        TypeError: Object of type CollectionSchema is not JSON serializable
        если вернуть CollectionSchema напрямую.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=8),
        ]
        schema = CollectionSchema(fields=fields, description="test")

        payload = {"schema": MilvusClient._collection_schema_to_dict(schema)}
        json.dumps(payload)  # не должно падать

