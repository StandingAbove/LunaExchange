import asyncio
import datetime as dt
from typing import Optional, Union

import polars as pl
from google.cloud import storage


class AsyncGCSWriter:
    """
    Minimal async uploader for various objects to GCS.
    - No queue, no worker thread mgmt.
    - Concurrency bounded by a semaphore.
    - Uses asyncio.to_thread() to offload blocking work.
    """

    def __init__(
            self,
            bucket_name: str,
            prefix: str = "",
            *,
            concurrency: int = 4,
            gzip: bool = False,
            chunk_size: Optional[int] = None,  # e.g., 8 * 1024 * 1024
            client: Optional[storage.Client] = None,
    ) -> None:
        if not bucket_name:
            raise ValueError("bucket_name must be provided")
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self._client = client or storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._tasks: set[asyncio.Task] = set()
        self._errors: list[tuple[str, BaseException]] = []
        self._gzip = gzip
        self._chunk_size = chunk_size
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def save_polars(
            self,
            frame: Union[pl.DataFrame, pl.LazyFrame],
            file_name: Optional[str] = None,
    ) -> None:
        """
        Schedule an upload. Returns when the task is scheduled, not finished.
        Call flush() or close() to wait for completion.
        """
        if self._closed:
            raise RuntimeError("writer is closed")

        if file_name is None:
            ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"frame_{ts}.csv"

        async def _task(name: str, f: Union[pl.DataFrame, pl.LazyFrame]) -> None:
            async with self._sem:
                try:
                    # Run the whole pipeline in one worker thread
                    await asyncio.to_thread(self._sync_polars, name, f)
                except Exception as e:  # keep going; surface later in flush()
                    self._errors.append((name, e))

        t = asyncio.create_task(_task(file_name, frame))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _sync_polars(self, name: str, frame: Union[pl.DataFrame, pl.LazyFrame]) -> None:
        # Collect if lazy (CPU-bound)
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()

        # Serialize to CSV bytes
        csv_bytes = frame.write_csv(None).encode("utf-8")
        if self._gzip:
            import gzip
            csv_bytes = gzip.compress(csv_bytes)

        # Build blob
        blob_name = f"{self.prefix}/{name}" if self.prefix else name
        blob = self._bucket.blob(blob_name)
        if self._chunk_size:
            blob.chunk_size = self._chunk_size  # can improve throughput on big objects
        if self._gzip:
            blob.content_encoding = "gzip"

        # Upload
        blob.upload_from_string(csv_bytes, content_type="text/csv")

    async def save_text(
            self,
            text: str,
            file_name: Optional[str] = None,
            *,
            content_type: str = "text/plain",
    ) -> None:
        """
        Schedule an upload of a plain text string to GCS. Returns when scheduled.
        Call flush() or close() to wait for completion.
        """
        if self._closed:
            raise RuntimeError("writer is closed")

        if file_name is None:
            ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"text_{ts}.txt"

        async def _task(name: str, txt: str) -> None:
            async with self._sem:
                try:
                    await asyncio.to_thread(self._sync_upload_text, name, txt, content_type)
                except Exception as e:
                    self._errors.append((name, e))

        t = asyncio.create_task(_task(file_name, text))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _sync_upload_text(self, name: str, text: str, content_type: str = "text/plain") -> None:
        data = text.encode("utf-8")
        if self._gzip:
            import gzip
            data = gzip.compress(data)

        blob_name = f"{self.prefix}/{name}" if self.prefix else name
        blob = self._bucket.blob(blob_name)
        if self._chunk_size:
            blob.chunk_size = self._chunk_size
        if self._gzip:
            blob.content_encoding = "gzip"

        blob.upload_from_string(data, content_type=content_type)

    async def flush(self) -> None:
        """Wait for all scheduled uploads to finish; raise on the first error."""
        while self._tasks:
            pending = list(self._tasks)
            await asyncio.gather(*pending, return_exceptions=False)
        if self._errors:
            name, err = self._errors[0]
            raise RuntimeError(f"GCS upload failed for {name}") from err

    async def close(self) -> None:
        """Flush and close the underlying GCS client."""
        if not self._closed:
            await self.flush()
            self._client.close()
            self._closed = True
