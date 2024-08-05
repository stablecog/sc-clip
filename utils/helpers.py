import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from PIL import Image
from io import BytesIO
import requests


@contextmanager
def time_log(after: str = "Completed", before: str | None = None):
    if before is not None:
        logging.info(before)
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        logging.info(f"{after}: {execution_time:.0f} ms")


def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    return Image.open(BytesIO(response.content)).convert("RGB")


def download_images(urls, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, url) for url in urls]
        images = [future.result() for future in futures]
    return images
