import os
import csv
import time
import numpy as np
from PIL import Image

from jpeg import encode_minijpeg, decode_minijpeg
from wht_24 import encode_miniwht, decode_miniwht


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_pixel: float = 255.0) -> float:
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    mse = np.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10((max_pixel ** 2) / mse)


def run_method(
    img_rgb: np.ndarray,
    original_bytes: int,
    quality: int,
    method_name: str,
    method_type: str,
    block_size: int,
    matrix_class: str
) -> dict:

    if method_type == "dct":
        start = time.perf_counter()
        encoded = encode_minijpeg(
            img_rgb,
            quality=quality,
            subsampling="420"
        )
        encode_time = time.perf_counter() - start

        start = time.perf_counter()
        decoded = decode_minijpeg(encoded)
        decode_time = time.perf_counter() - start

    elif method_type == "wht":
        start = time.perf_counter()
        encoded = encode_miniwht(
            img_rgb,
            quality=quality,
            subsampling="420",
            block_size=block_size,
            matrix_class=matrix_class
        )
        encode_time = time.perf_counter() - start

        start = time.perf_counter()
        decoded = decode_miniwht(encoded)
        decode_time = time.perf_counter() - start

    else:
        raise ValueError(f"Неизвестный тип метода: {method_type}")

    compressed_bytes = len(encoded.bitstream) / 8
    compression_ratio = original_bytes / compressed_bytes
    psnr_value = psnr(img_rgb, decoded)

    return {
        "method": method_name,
        "quality": quality,
        "block_size": block_size,
        "matrix_class": matrix_class,
        "psnr": psnr_value,
        "compression_ratio": compression_ratio,
        "compressed_bytes": compressed_bytes,
        "encode_time": encode_time,
        "decode_time": decode_time,
        "bitstream_length": len(encoded.bitstream)
    }


def run_experiments(data_folder="data", output_file="results_paley.csv"):
    qualities = [20, 30, 40, 50, 60, 70, 80]

    methods = [
        {
            "method_name": "MiniJPEG_DCT_8",
            "method_type": "dct",
            "block_size": 8,
            "matrix_class": "dct"
        },
        {
            "method_name": "MiniJPEG_WHT_16_class1",
            "method_type": "wht",
            "block_size": 16,
            "matrix_class": "class1"
        },
        {
            "method_name": "MiniJPEG_WHT_24_paley",
            "method_type": "wht",
            "block_size": 24,
            "matrix_class": "paley"
        },
    ]

    results = []

    for filename in sorted(os.listdir(data_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        path = os.path.join(data_folder, filename)
        print(f"\n=== Processing: {filename} ===")

        img = Image.open(path).convert("RGB")
        img_rgb = np.array(img, dtype=np.uint8)

        original_bytes = img_rgb.shape[0] * img_rgb.shape[1] * 3

        for quality in qualities:
            print(f"Quality: {quality}")

            for method in methods:
                print(f"  Method: {method['method_name']}")

                row = run_method(
                    img_rgb=img_rgb,
                    original_bytes=original_bytes,
                    quality=quality,
                    method_name=method["method_name"],
                    method_type=method["method_type"],
                    block_size=method["block_size"],
                    matrix_class=method["matrix_class"]
                )

                row["image"] = filename
                results.append(row)

    if not results:
        raise RuntimeError("Не найдено изображений в папке data.")

    fieldnames = [
        "image",
        "method",
        "quality",
        "block_size",
        "matrix_class",
        "psnr",
        "compression_ratio",
        "compressed_bytes",
        "encode_time",
        "decode_time",
        "bitstream_length"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")
    print(f"Total rows: {len(results)}")


if __name__ == "__main__":
    run_experiments("data", "results_paley.csv")