import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image


# ============================================================
# 1. Загрузка и сохранение RGB
# ============================================================

def load_rgb_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)


def save_rgb_image(image: np.ndarray, path: str) -> None:
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="RGB").save(path)


# ============================================================
# 2. RGB <-> YCbCr
# ============================================================

def rgb_to_ycbcr(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = img_rgb.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0

    return Y, Cb, Cr


def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0

    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


# ============================================================
# 3. Padding, блоки
# ============================================================

def pad_image_to_block_size(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape
    new_h = math.ceil(h / block_size) * block_size
    new_w = math.ceil(w / block_size) * block_size

    padded = np.pad(
        image,
        ((0, new_h - h), (0, new_w - w)),
        mode="edge"
    ).astype(np.float32)

    return padded, (h, w)


def crop_to_original_size(image: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    h, w = original_shape
    return image[:h, :w]


def split_into_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    h, w = image.shape
    bh = h // block_size
    bw = w // block_size
    return image.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)


def merge_blocks(blocks: np.ndarray) -> np.ndarray:
    bh, bw, bs1, bs2 = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(bh * bs1, bw * bs2)


# ============================================================
# 4. Матрица Адамара и WHT
# ============================================================

def hadamard_matrix(n: int) -> np.ndarray:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Размер матрицы Адамара должен быть степенью двойки.")

    h = np.array([[1.0]], dtype=np.float32)

    while h.shape[0] < n:
        h = np.block([
            [h,  h],
            [h, -h]
        ]).astype(np.float32)

    return h


def normalized_hadamard_matrix(n: int) -> np.ndarray:
    return hadamard_matrix(n) / math.sqrt(n)


def wht2d(block: np.ndarray, hmat: np.ndarray) -> np.ndarray:
    return hmat @ block @ hmat.T


def iwht2d(coeffs: np.ndarray, hmat: np.ndarray) -> np.ndarray:
    return hmat.T @ coeffs @ hmat


# ============================================================
# 5. Матрицы квантования
# ============================================================

def default_wht_quant_matrix(block_size: int = 8) -> np.ndarray:
    if block_size != 8:
        q = np.fromfunction(lambda i, j: 1 + i + j, (block_size, block_size))
        return q.astype(np.float32)

    return np.array([
        [8, 10, 12, 14, 18, 22, 26, 30],
        [10, 12, 14, 16, 20, 24, 28, 32],
        [12, 14, 16, 18, 22, 26, 30, 34],
        [14, 16, 18, 20, 24, 28, 32, 36],
        [18, 20, 22, 24, 28, 32, 36, 40],
        [22, 24, 26, 28, 32, 36, 40, 44],
        [26, 28, 30, 32, 36, 40, 44, 48],
        [30, 32, 34, 36, 40, 44, 48, 52],
    ], dtype=np.float32)


def quality_scale_matrix(qmat: np.ndarray, quality: int) -> np.ndarray:
    if not (1 <= quality <= 100):
        raise ValueError("quality должен быть в диапазоне 1..100")

    if quality < 50:
        scale = 50.0 / quality
    else:
        scale = 2.0 - quality / 50.0

    scaled = np.floor(qmat * scale + 0.5)
    scaled[scaled < 1] = 1

    return scaled.astype(np.float32)


def quantize(coeffs: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    return np.round(coeffs / qmat).astype(np.int32)


def dequantize(qcoeffs: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    return (qcoeffs * qmat).astype(np.float32)


# ============================================================
# 6. Метрики
# ============================================================

def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    return float(np.mean(diff ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_pixel: float = 255.0) -> float:
    err = mse(original, reconstructed)
    if err == 0:
        return float("inf")
    return 10.0 * math.log10((max_pixel ** 2) / err)


# ============================================================
# 7. Сжатие одного канала
# ============================================================

@dataclass
class WHTChannelResult:
    reconstructed: np.ndarray
    qcoeffs_blocks: np.ndarray
    encode_time_sec: float
    decode_time_sec: float
    compression_ratio_est: float


def compress_decompress_channel_wht(
    channel: np.ndarray,
    block_size: int,
    quality: int,
    quant_matrix: Optional[np.ndarray] = None,
    center_pixels: bool = True
) -> WHTChannelResult:

    padded, original_shape = pad_image_to_block_size(channel, block_size)

    if center_pixels:
        padded = padded - 128.0

    blocks = split_into_blocks(padded, block_size)

    hmat = normalized_hadamard_matrix(block_size)

    if quant_matrix is None:
        quant_matrix = default_wht_quant_matrix(block_size)

    qmat = quality_scale_matrix(quant_matrix, quality)

    bh, bw, _, _ = blocks.shape

    qcoeffs_blocks = np.zeros_like(blocks, dtype=np.int32)
    reconstructed_blocks = np.zeros_like(blocks, dtype=np.float32)

    encode_start = time.perf_counter()

    for i in range(bh):
        for j in range(bw):
            coeffs = wht2d(blocks[i, j], hmat)
            qcoeffs_blocks[i, j] = quantize(coeffs, qmat)

    encode_time_sec = time.perf_counter() - encode_start

    decode_start = time.perf_counter()

    for i in range(bh):
        for j in range(bw):
            deq = dequantize(qcoeffs_blocks[i, j], qmat)
            reconstructed_blocks[i, j] = iwht2d(deq, hmat)

    decode_time_sec = time.perf_counter() - decode_start

    reconstructed = merge_blocks(reconstructed_blocks)

    if center_pixels:
        reconstructed = reconstructed + 128.0

    reconstructed = crop_to_original_size(reconstructed, original_shape)
    reconstructed = np.clip(reconstructed, 0, 255)

    original_elements = channel.shape[0] * channel.shape[1]
    nonzero = np.count_nonzero(qcoeffs_blocks)

    compression_ratio_est = original_elements / nonzero if nonzero != 0 else float("inf")

    return WHTChannelResult(
        reconstructed=reconstructed,
        qcoeffs_blocks=qcoeffs_blocks,
        encode_time_sec=encode_time_sec,
        decode_time_sec=decode_time_sec,
        compression_ratio_est=compression_ratio_est
    )


# ============================================================
# 8. Цветное WHT-сжатие
# ============================================================

@dataclass
class WHTColorResult:
    reconstructed_rgb: np.ndarray
    psnr_rgb: float
    psnr_y: float
    compression_ratio_est: float
    encode_time_sec: float
    decode_time_sec: float


def compress_decompress_color_wht(
    img_rgb: np.ndarray,
    block_size: int = 8,
    quality_y: int = 60,
    quality_c: int = 40
) -> WHTColorResult:

    Y, Cb, Cr = rgb_to_ycbcr(img_rgb)

    y_result = compress_decompress_channel_wht(
        Y,
        block_size=block_size,
        quality=quality_y
    )

    cb_result = compress_decompress_channel_wht(
        Cb,
        block_size=block_size,
        quality=quality_c
    )

    cr_result = compress_decompress_channel_wht(
        Cr,
        block_size=block_size,
        quality=quality_c
    )

    reconstructed_rgb = ycbcr_to_rgb(
        y_result.reconstructed,
        cb_result.reconstructed,
        cr_result.reconstructed
    )

    psnr_rgb_value = psnr(img_rgb, reconstructed_rgb)
    psnr_y_value = psnr(Y, y_result.reconstructed)

    compression_ratio_est = (
        y_result.compression_ratio_est
        + cb_result.compression_ratio_est
        + cr_result.compression_ratio_est
    ) / 3.0

    encode_time = (
        y_result.encode_time_sec
        + cb_result.encode_time_sec
        + cr_result.encode_time_sec
    )

    decode_time = (
        y_result.decode_time_sec
        + cb_result.decode_time_sec
        + cr_result.decode_time_sec
    )

    return WHTColorResult(
        reconstructed_rgb=reconstructed_rgb,
        psnr_rgb=psnr_rgb_value,
        psnr_y=psnr_y_value,
        compression_ratio_est=compression_ratio_est,
        encode_time_sec=encode_time,
        decode_time_sec=decode_time
    )


# ============================================================
# 9. Пример запуска
# ============================================================

if __name__ == "__main__":
    input_path = "input.png"
    output_path = "reconstructed_color_wht.png"

    img_rgb = load_rgb_image(input_path)

    result = compress_decompress_color_wht(
        img_rgb,
        block_size=8,
        quality_y=60,
        quality_c=40
    )

    save_rgb_image(result.reconstructed_rgb, output_path)

    print("=== Color WHT Compression Result ===")
    print(f"PSNR RGB: {result.psnr_rgb:.2f} dB")
    print(f"PSNR Y:   {result.psnr_y:.2f} dB")
    print(f"Compression ratio estimate: {result.compression_ratio_est:.2f}")
    print(f"Encode time: {result.encode_time_sec:.6f} sec")
    print(f"Decode time: {result.decode_time_sec:.6f} sec")
