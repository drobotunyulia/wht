import os
import csv
import time
import numpy as np
from PIL import Image

from jpeg import encode_minijpeg, decode_minijpeg
from wht_8_16 import encode_miniwht, decode_miniwht


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_pixel: float = 255.0) -> float:
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)

    mse = np.mean((original - reconstructed) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10((max_pixel ** 2) / mse)


def run_experiments(data_folder="data", output_file="results.csv"):
    qualities = [20, 30, 40, 50, 60, 70, 80]

    results = []

    for filename in sorted(os.listdir(data_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        path = os.path.join(data_folder, filename)
        print(f"\n=== Processing: {filename} ===")

        img = Image.open(path).convert("RGB")
        img_rgb = np.array(img, dtype=np.uint8)

        original_bytes = img_rgb.shape[0] * img_rgb.shape[1] * 3

        for q in qualities:
            print(f"Quality: {q}")

            # =========================
            # 1. MiniJPEG DCT
            # =========================
            start = time.perf_counter()
            encoded_dct = encode_minijpeg(
                img_rgb,
                quality=q,
                subsampling="420"
            )
            dct_encode_time = time.perf_counter() - start

            start = time.perf_counter()
            decoded_dct = decode_minijpeg(encoded_dct)
            dct_decode_time = time.perf_counter() - start

            dct_compressed_bytes = len(encoded_dct.bitstream) / 8
            dct_cr = original_bytes / dct_compressed_bytes
            dct_psnr = psnr(img_rgb, decoded_dct)

            results.append({
                "image": filename,
                "quality": q,
                "method": "MiniJPEG_DCT",
                "block_size": 8,
                "psnr": dct_psnr,
                "compression_ratio": dct_cr,
                "compressed_bytes": dct_compressed_bytes,
                "encode_time": dct_encode_time,
                "decode_time": dct_decode_time,
                "bitstream_length": len(encoded_dct.bitstream)
            })

            # =========================
            # 2. MiniJPEG WHT 8x8
            # =========================
            start = time.perf_counter()
            encoded_wht_8 = encode_miniwht(
                img_rgb,
                quality=q,
                subsampling="420",
                block_size=8
            )
            wht8_encode_time = time.perf_counter() - start

            start = time.perf_counter()
            decoded_wht_8 = decode_miniwht(encoded_wht_8)
            wht8_decode_time = time.perf_counter() - start

            wht8_compressed_bytes = len(encoded_wht_8.bitstream) / 8
            wht8_cr = original_bytes / wht8_compressed_bytes
            wht8_psnr = psnr(img_rgb, decoded_wht_8)

            results.append({
                "image": filename,
                "quality": q,
                "method": "MiniJPEG_WHT_8",
                "block_size": 8,
                "psnr": wht8_psnr,
                "compression_ratio": wht8_cr,
                "compressed_bytes": wht8_compressed_bytes,
                "encode_time": wht8_encode_time,
                "decode_time": wht8_decode_time,
                "bitstream_length": len(encoded_wht_8.bitstream)
            })

            # =========================
            # 3. MiniJPEG WHT 16x16
            # =========================
            start = time.perf_counter()
            encoded_wht_16 = encode_miniwht(
                img_rgb,
                quality=q,
                subsampling="420",
                block_size=16
            )
            wht16_encode_time = time.perf_counter() - start

            start = time.perf_counter()
            decoded_wht_16 = decode_miniwht(encoded_wht_16)
            wht16_decode_time = time.perf_counter() - start

            wht16_compressed_bytes = len(encoded_wht_16.bitstream) / 8
            wht16_cr = original_bytes / wht16_compressed_bytes
            wht16_psnr = psnr(img_rgb, decoded_wht_16)

            results.append({
                "image": filename,
                "quality": q,
                "method": "MiniJPEG_WHT_16",
                "block_size": 16,
                "psnr": wht16_psnr,
                "compression_ratio": wht16_cr,
                "compressed_bytes": wht16_compressed_bytes,
                "encode_time": wht16_encode_time,
                "decode_time": wht16_decode_time,
                "bitstream_length": len(encoded_wht_16.bitstream)
            })

    if not results:
        raise RuntimeError("Не найдено изображений в папке data.")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_file}")
    print(f"Total rows: {len(results)}")


if __name__ == "__main__":
    run_experiments("data", "results_jpeg_8_16.csv")