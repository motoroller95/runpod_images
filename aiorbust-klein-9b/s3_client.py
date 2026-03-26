import logging
import os
import time

import boto3

logger = logging.getLogger("klein_9b.s3_client")


class S3Client:
    def __init__(self):
        self.bucket = os.environ["S3_BUCKET"]
        self.presign_ttl_seconds = int(os.getenv("S3_PRESIGN_TTL_SECONDS", "86400"))
        self.client = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION"),
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        )
        logger.info(
            "S3 client initialized: bucket=%s region=%s endpoint=%s presign_ttl=%s",
            self.bucket,
            os.getenv("AWS_REGION"),
            os.getenv("S3_ENDPOINT_URL"),
            self.presign_ttl_seconds,
        )

    def upload_and_presign(self, body: bytes, key: str, content_type: str | None = None) -> str:
        started_at = time.perf_counter()
        logger.info(
            "Uploading object to S3: key=%s bytes=%d content_type=%s",
            key,
            len(body),
            content_type,
        )
        extra_args = {"ContentType": content_type} if content_type else {}
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            **extra_args,
        )
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.presign_ttl_seconds,
        )
        duration = time.perf_counter() - started_at
        logger.info(
            "S3 upload complete: key=%s duration=%.2fs presign_ttl=%ss",
            key,
            duration,
            self.presign_ttl_seconds,
        )
        return url
