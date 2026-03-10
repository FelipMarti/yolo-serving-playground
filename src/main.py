import os
import sys
import time
from urllib.parse import urlparse

from infer_module import infer_image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def is_url(value: str) -> bool:
    """Check if a string looks like a URL."""
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and parsed.netloc != ""


def run_inference(input_value: str, input_type: str):
    start = time.perf_counter()

    if input_type == "url":
        result = infer_image(image_url=input_value)
    elif input_type == "path":
        result = infer_image(image_path=input_value)
    else:
        raise ValueError("Unsupported input type")

    elapsed = time.perf_counter() - start

    print("===================================")
    print(f"Input: {input_value}")
    print(f"Output: {result}")
    print(f"Time: {elapsed:.3f} seconds")
    print("===================================")


def run_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"Folder not found: {folder_path}")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file_name)[1].lower()
        if ext not in VALID_EXTENSIONS:
            continue

        run_inference(file_path, "path")


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python3 main.py <url | image_path | folder_path>")
        sys.exit(1)

    input_value = sys.argv[1]

    # URL
    if is_url(input_value):
        run_inference(input_value, "url")
        return

    # Folder
    if os.path.isdir(input_value):
        run_folder(input_value)
        return

    # File
    if os.path.isfile(input_value):
        ext = os.path.splitext(input_value)[1].lower()

        if ext not in VALID_EXTENSIONS:
            raise RuntimeError(f"Unsupported file type: {ext}")

        run_inference(input_value, "path")
        return

    raise RuntimeError(f"Input not recognized: {input_value}")


if __name__ == "__main__":
    main()
