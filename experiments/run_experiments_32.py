import os
import csv
import time
import numpy as np
from PIL import Image

from jpeg import encode_minijpeg, decode_minijpeg
from wht import encode_miniwht, decode_miniwht


DATA_DIR = "data"
OUTPUT_FILE = "results_32.csv"

QUALITIES = [20, 30, 40, 50, 60, 70, 80]
SUBSAMPLING = "420"
QUANT_MODE = "jpeg"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


METHODS = [
    {
        "name": "JPEG_DCT_8",
        "type": "jpeg",
        "block_size": 8,
        "matrix_class": "dct",
    },
    {
        "name": "WHT_8_sylvester",
        "type": "wht",
        "block_size": 8,
        "matrix_class": "sylvester",
    },
    {
        "name": "WHT_16_class1",
        "type": "wht",
        "block_size": 16,
        "matrix_class": "class1",
    },
    {
        "name": "WHT_32_sylvester",
        "type": "wht",
        "block_size": 32,
        "matrix_class": "sylvester",
    },
    {
        "name": "WHT_32_sequency",
        "type": "wht",
        "block_size": 32,
        "matrix_class": "sequency",
    },
]


def psnr(original, reconstructed, max_pixel=255.0):
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    mse = np.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10((max_pixel ** 2) / mse)


def run_method(img_rgb, original_bytes, quality, method):
    start = time.perf_counter()

    if method["type"] == "jpeg":
        encoded = encode_minijpeg(
            img_rgb,
            quality=quality,
            subsampling=SUBSAMPLING,
        )
    else:
        encoded = encode_miniwht(
            img_rgb,
            quality=quality,
            subsampling=SUBSAMPLING,
            block_size=method["block_size"],
            matrix_class=method["matrix_class"],
            quant_mode=QUANT_MODE,
        )

    encode_time = time.perf_counter() - start

    start = time.perf_counter()

    if method["type"] == "jpeg":
        decoded = decode_minijpeg(encoded)
    else:
        decoded = decode_miniwht(encoded)

    decode_time = time.perf_counter() - start

    compressed_bytes = len(encoded.bitstream) / 8
    compression_ratio = original_bytes / compressed_bytes
    reduction_percent = 100 * (1 - compressed_bytes / original_bytes)

    return {
        "method": method["name"],
        "quality": quality,
        "subsampling": SUBSAMPLING,
        "quant_mode": QUANT_MODE if method["type"] == "wht" else "jpeg",
        "block_size": method["block_size"],
        "matrix_class": method["matrix_class"],
        "psnr": psnr(img_rgb, decoded),
        "compression_ratio": compression_ratio,
        "reduction_percent": reduction_percent,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "bitstream_length": len(encoded.bitstream),
        "encode_time": encode_time,
        "decode_time": decode_time,
        "total_time": encode_time + decode_time,
    }


def run_experiments(data_dir=DATA_DIR, output_file=OUTPUT_FILE):
    results = []

    image_files = [
        filename for filename in sorted(os.listdir(data_dir))
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not image_files:
        raise ValueError(f"В папке {data_dir} нет изображений.")

    for filename in image_files:
        print(f"\n=== Processing: {filename} ===")

        path = os.path.join(data_dir, filename)
        img = Image.open(path).convert("RGB")
        img_rgb = np.array(img, dtype=np.uint8)

        h, w, _ = img_rgb.shape
        original_bytes = h * w * 3

        for quality in QUALITIES:
            print(f"Quality: {quality}")

            for method in METHODS:
                print(f"  {method['name']}")

                row = run_method(
                    img_rgb=img_rgb,
                    original_bytes=original_bytes,
                    quality=quality,
                    method=method,
                )

                row["image"] = filename
                results.append(row)

    fieldnames = [
        "image",
        "method",
        "quality",
        "subsampling",
        "quant_mode",
        "block_size",
        "matrix_class",
        "psnr",
        "compression_ratio",
        "reduction_percent",
        "original_bytes",
        "compressed_bytes",
        "bitstream_length",
        "encode_time",
        "decode_time",
        "total_time",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to: {output_file}")
    print(f"Rows: {len(results)}")


if __name__ == "__main__":
    run_experiments()
