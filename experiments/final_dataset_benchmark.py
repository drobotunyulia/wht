import os
import csv
import time
import numpy as np
from PIL import Image

from jpeg import encode_minijpeg, decode_minijpeg
from wht_quant import encode_miniwht, decode_miniwht


DATA_DIR = "data"
OUTPUT_CSV = "final_dataset_benchmark_32.csv"

QUALITY = 50
SUBSAMPLING = "420"

WHT_BLOCK_SIZE = 32
WHT_MATRIX_CLASS = "sequency"
WHT_QUANT_MODE = "jpeg"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def get_psnr(original, reconstructed):
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    mse = np.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10((255.0 ** 2) / mse)


def run_jpeg(img_rgb):
    start = time.perf_counter()

    encoded = encode_minijpeg(
        img_rgb,
        quality=QUALITY,
        subsampling=SUBSAMPLING
    )

    encode_time = time.perf_counter() - start

    start = time.perf_counter()
    decoded = decode_minijpeg(encoded)
    decode_time = time.perf_counter() - start

    compressed_bytes = len(encoded.bitstream) / 8

    return decoded, encode_time, decode_time, compressed_bytes


def run_wht(img_rgb):
    start = time.perf_counter()

    encoded = encode_miniwht(
        img_rgb,
        quality=QUALITY,
        subsampling=SUBSAMPLING,
        block_size=WHT_BLOCK_SIZE,
        matrix_class=WHT_MATRIX_CLASS,
        quant_mode=WHT_QUANT_MODE
    )

    encode_time = time.perf_counter() - start

    start = time.perf_counter()
    decoded = decode_miniwht(encoded)
    decode_time = time.perf_counter() - start

    compressed_bytes = len(encoded.bitstream) / 8

    return decoded, encode_time, decode_time, compressed_bytes


def main():
    image_files = [
        f for f in sorted(os.listdir(DATA_DIR))
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not image_files:
        raise ValueError("В папке data нет изображений.")

    method_jpeg = "JPEG_DCT_8"
    method_wht = "WHT_32_sequency"

    totals = {
        method_jpeg: {
            "original_bytes": 0,
            "compressed_bytes": 0,
            "encode_time": 0,
            "decode_time": 0,
            "psnr_sum": 0,
            "count": 0
        },
        method_wht: {
            "original_bytes": 0,
            "compressed_bytes": 0,
            "encode_time": 0,
            "decode_time": 0,
            "psnr_sum": 0,
            "count": 0
        }
    }

    rows = []

    for filename in image_files:
        print(f"\n=== Processing {filename} ===")

        path = os.path.join(DATA_DIR, filename)
        img = Image.open(path).convert("RGB")
        img_rgb = np.array(img, dtype=np.uint8)

        h, w, _ = img_rgb.shape
        original_bytes = h * w * 3

        jpeg_decoded, jpeg_enc_t, jpeg_dec_t, jpeg_comp_bytes = run_jpeg(img_rgb)
        jpeg_psnr = get_psnr(img_rgb, jpeg_decoded)

        wht_decoded, wht_enc_t, wht_dec_t, wht_comp_bytes = run_wht(img_rgb)
        wht_psnr = get_psnr(img_rgb, wht_decoded)

        for method, comp_bytes, enc_t, dec_t, psnr in [
            (method_jpeg, jpeg_comp_bytes, jpeg_enc_t, jpeg_dec_t, jpeg_psnr),
            (method_wht, wht_comp_bytes, wht_enc_t, wht_dec_t, wht_psnr),
        ]:
            compression_ratio = original_bytes / comp_bytes
            reduction_percent = 100 * (1 - comp_bytes / original_bytes)

            rows.append({
                "image": filename,
                "method": method,
                "quality": QUALITY,
                "subsampling": SUBSAMPLING,
                "block_size": 8 if method == method_jpeg else WHT_BLOCK_SIZE,
                "matrix_class": "dct" if method == method_jpeg else WHT_MATRIX_CLASS,
                "quant_mode": "jpeg",
                "original_bytes": original_bytes,
                "compressed_bytes": comp_bytes,
                "compression_ratio": compression_ratio,
                "reduction_percent": reduction_percent,
                "encode_time": enc_t,
                "decode_time": dec_t,
                "total_time": enc_t + dec_t,
                "psnr": psnr
            })

            totals[method]["original_bytes"] += original_bytes
            totals[method]["compressed_bytes"] += comp_bytes
            totals[method]["encode_time"] += enc_t
            totals[method]["decode_time"] += dec_t
            totals[method]["psnr_sum"] += psnr
            totals[method]["count"] += 1

    fieldnames = [
        "image",
        "method",
        "quality",
        "subsampling",
        "block_size",
        "matrix_class",
        "quant_mode",
        "original_bytes",
        "compressed_bytes",
        "compression_ratio",
        "reduction_percent",
        "encode_time",
        "decode_time",
        "total_time",
        "psnr"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n================ FINAL SUMMARY ================")

    for method, data in totals.items():
        original = data["original_bytes"]
        compressed = data["compressed_bytes"]

        compression_ratio = original / compressed
        reduction_percent = 100 * (1 - compressed / original)

        encode_time = data["encode_time"]
        decode_time = data["decode_time"]
        total_time = encode_time + decode_time

        mean_psnr = data["psnr_sum"] / data["count"]

        print(f"\n{method}")
        print(f"Images: {data['count']}")
        print(f"Original size:   {original:.0f} bytes")
        print(f"Compressed size: {compressed:.0f} bytes")
        print(f"Compression ratio: {compression_ratio:.2f} : 1")
        print(f"Dataset reduction: {reduction_percent:.2f}%")
        print(f"Encode time total: {encode_time:.6f} sec")
        print(f"Decode time total: {decode_time:.6f} sec")
        print(f"Total time:        {total_time:.6f} sec")
        print(f"Mean PSNR:         {mean_psnr:.2f} dB")

    print(f"\nSaved detailed results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
