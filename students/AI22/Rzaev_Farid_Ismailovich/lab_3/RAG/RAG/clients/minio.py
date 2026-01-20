import uuid
from minio import Minio
from minio.error import S3Error
from io import BytesIO


class MinioService:
    def __init__(self, host: str, access_key: str, secret_key: str, secure: bool = False, default_bucket: str = "files"):
        self.client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.default_bucket = default_bucket

    def ensure_bucket(self, bucket_name: str):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    def generate_unique_filename(self, filename: str) -> str:
        ext = filename.split(".")[-1] if "." in filename else ""
        unique_name = f"{uuid.uuid4().hex}"
        if ext:
            unique_name += f".{ext}"
        return unique_name

    def upload_file(self, file_path: str, bucket_name: str = None, object_name: str = None) -> str:
        bucket_name = bucket_name or self.default_bucket
        self.ensure_bucket(bucket_name)
        if object_name is None:
            object_name = self.generate_unique_filename(file_path)
        self.client.fput_object(bucket_name, object_name, file_path)
        return object_name

    def upload_file_RAM(self, file_name: str, mime_type: str, file_stream: BytesIO, bucket_name: str = None) -> str:
        bucket_name = bucket_name or self.default_bucket
        self.ensure_bucket(bucket_name)
        object_name = self.generate_unique_filename(file_name)
        self.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=file_stream,
            length=file_stream.getbuffer().nbytes,
            content_type=mime_type
        )
        return object_name

    def delete_file(self, object_name: str, bucket_name: str = None):
        bucket_name = bucket_name or self.default_bucket
        try:
            self.client.remove_object(bucket_name, object_name)
        except S3Error as e:
            print(f"Ошибка удаления {object_name}: {e}")

    def list_files(self, bucket_name: str = None):
        bucket_name = bucket_name or self.default_bucket
        self.ensure_bucket(bucket_name)
        return [obj.object_name for obj in self.client.list_objects(bucket_name, recursive=True)]

    def get_file_url(self, object_name: str, bucket_name: str = None) -> str:
        bucket_name = bucket_name or self.default_bucket
        return self.client.presigned_get_object(bucket_name, object_name)

    def get_file_stream(self, bucket: str, object_name: str) -> BytesIO:
        response = self.client.get_object(bucket, object_name)
        file_data = response.read()  
        response.close()
        response.release_conn()
        return BytesIO(file_data)
