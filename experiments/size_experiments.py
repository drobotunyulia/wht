import os
import csv
import time
import numpy as np
from PIL import Image

from jpeg import encode_minijpeg, decode_minijpeg
from wht_quant import encode_miniwht, decode_miniwht


DATA_DIR = "data"
DETAIL_OUTPUT = "size_experiments_detail.csv"
SUMMARY_OUTPUT = "size_experiments_summary.csv"

IMAGE_SIZES = [256, 512, 1024, 2048]
QUALITY = 50
SUBSAMPLING = "420"
QUANT_MODE = "jpeg"

# Можно поставить 20 или 30, чтобы эксперимент не шел слишком долго
MAX_IMAGES = 20

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


METHODS = [
    {
        "name": "JPEG_DCT_8",
        "type": "jpeg",
        "block_size": 8,
        "matrix_class": "dct",
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


def resize_image(img: Image.Image, size: int) -> np.ndarray:
    """
    Масштабирует изображение к размеру size x size.
    """
    img_resized = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.array(img_resized.convert("RGB"), dtype=np.uint8)


def run_method(img_rgb, original_bytes, method):
    start = time.perf_counter()

    if method["type"] == "jpeg":
        encoded = encode_minijpeg(
            img_rgb,
            quality=QUALITY,
            subsampling=SUBSAMPLING,
        )
    else:
        encoded = encode_miniwht(
            img_rgb,
            quality=QUALITY,
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

    return {
        "method": method["name"],
        "quality": QUALITY,
        "subsampling": SUBSAMPLING,
        "quant_mode": QUANT_MODE if method["type"] == "wht" else "jpeg",
        "block_size": method["block_size"],
        "matrix_class": method["matrix_class"],
        "psnr": psnr(img_rgb, decoded),
        "compression_ratio": compression_ratio,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compressed_kb": compressed_bytes / 1024,
        "bitstream_length": len(encoded.bitstream),
        "encode_time_sec": encode_time,
        "encode_time_ms": encode_time * 1000,
        "decode_time_sec": decode_time,
        "total_time_sec": encode_time + decode_time,
    }


def run_size_experiments():
    detailed_results = []

    image_files = [
        filename for filename in sorted(os.listdir(DATA_DIR))
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if MAX_IMAGES is not None:
        image_files = image_files[:MAX_IMAGES]

    if not image_files:
        raise ValueError(f"В папке {DATA_DIR} нет изображений.")

    print(f"Images selected: {len(image_files)}")
    print(f"Sizes: {IMAGE_SIZES}")
    print(f"Quality: {QUALITY}")

    for filename in image_files:
        path = os.path.join(DATA_DIR, filename)
        img = Image.open(path).convert("RGB")

        print(f"\n=== Image: {filename} ===")

        for size in IMAGE_SIZES:
            print(f"  Size: {size}x{size}")

            img_rgb = resize_image(img, size)
            h, w, _ = img_rgb.shape
            original_bytes = h * w * 3

            for method in METHODS:
                print(f"    {method['name']}")

                row = run_method(
                    img_rgb=img_rgb,
                    original_bytes=original_bytes,
                    method=method,
                )

                row["image"] = filename
                row["image_size"] = f"{size}x{size}"
                row["height"] = h
                row["width"] = w

                detailed_results.append(row)

    detail_fieldnames = [
        "image",
        "image_size",
        "height",
        "width",
        "method",
        "quality",
        "subsampling",
        "quant_mode",
        "block_size",
        "matrix_class",
        "psnr",
        "compression_ratio",
        "original_bytes",
        "compressed_bytes",
        "compressed_kb",
        "bitstream_length",
        "encode_time_sec",
        "encode_time_ms",
        "decode_time_sec",
        "total_time_sec",
    ]

    with open(DETAIL_OUTPUT, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detail_fieldnames)
        writer.writeheader()
        writer.writerows(detailed_results)

    print(f"\nSaved detailed results to: {DETAIL_OUTPUT}")

    make_summary(detailed_results)


def make_summary(detailed_results):
    summary_rows = []

    for size in IMAGE_SIZES:
        image_size = f"{size}x{size}"

        for method in METHODS:
            method_name = method["name"]

            rows = [
                row for row in detailed_results
                if row["image_size"] == image_size and row["method"] == method_name
            ]

            if not rows:
                continue

            summary_rows.append({
                "image_size": image_size,
                "method": method_name,
                "images": len(rows),
                "mean_cr": np.mean([r["compression_ratio"] for r in rows]),
                "mean_compressed_kb": np.mean([r["compressed_kb"] for r in rows]),
                "mean_encode_time_ms": np.mean([r["encode_time_ms"] for r in rows]),
                "mean_decode_time_ms": np.mean([r["decode_time_sec"] * 1000 for r in rows]),
                "mean_total_time_ms": np.mean([r["total_time_sec"] * 1000 for r in rows]),
                "mean_psnr": np.mean([r["psnr"] for r in rows]),
            })

    # Добавляем признак эффективности WHT относительно JPEG
    for size in IMAGE_SIZES:
        image_size = f"{size}x{size}"

        jpeg_row = next(
            (r for r in summary_rows if r["image_size"] == image_size and r["method"] == "JPEG_DCT_8"),
            None
        )

        wht_row = next(
            (r for r in summary_rows if r["image_size"] == image_size and r["method"] == "WHT_32_sequency"),
            None
        )

        if jpeg_row is not None:
            jpeg_row["effective_vs_jpeg"] = "-"

        if jpeg_row is not None and wht_row is not None:
            is_effective = (
                wht_row["mean_cr"] > jpeg_row["mean_cr"]
                and wht_row["mean_encode_time_ms"] <= jpeg_row["mean_encode_time_ms"]
                and wht_row["mean_psnr"] >= 25
            )

            wht_row["effective_vs_jpeg"] = "yes" if is_effective else "no"

    summary_fieldnames = [
        "image_size",
        "method",
        "images",
        "mean_cr",
        "mean_compressed_kb",
        "mean_encode_time_ms",
        "mean_decode_time_ms",
        "mean_total_time_ms",
        "mean_psnr",
        "effective_vs_jpeg",
    ]

    with open(SUMMARY_OUTPUT, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary results to: {SUMMARY_OUTPUT}")

    print("\n================ SIZE EXPERIMENT SUMMARY ================")
    for row in summary_rows:
        print(
            f"{row['image_size']:>9} | {row['method']:<16} | "
            f"CR={row['mean_cr']:.2f} | "
            f"size={row['mean_compressed_kb']:.2f} KB | "
            f"enc={row['mean_encode_time_ms']:.2f} ms | "
            f"PSNR={row['mean_psnr']:.2f} dB | "
            f"effective={row.get('effective_vs_jpeg', '')}"
        )


if __name__ == "__main__":
    run_size_experiments()
