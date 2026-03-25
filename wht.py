import math
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from typing import Optional

import numpy as np
from PIL import Image


# =========================
# Вспомогательные функции
# =========================

def load_grayscale_image(path: str) -> np.ndarray:
    """Загружает изображение в оттенках серого как float32-массив."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def save_grayscale_image(image: np.ndarray, path: str) -> None:
    """Сохраняет изображение в диапазоне [0, 255]."""
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


def pad_image_to_block_size(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Дополняет изображение до размеров, кратных block_size.
    Возвращает дополненное изображение и исходный размер.
    """
    h, w = image.shape
    new_h = math.ceil(h / block_size) * block_size
    new_w = math.ceil(w / block_size) * block_size

    padded = np.zeros((new_h, new_w), dtype=np.float32)
    padded[:h, :w] = image
    return padded, (h, w)


def crop_to_original_size(image: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """Обрезает изображение до исходного размера."""
    h, w = original_shape
    return image[:h, :w]


def split_into_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    """
    Разбивает изображение на блоки.
    Возвращает массив формы:
    (num_blocks_h, num_blocks_w, block_size, block_size)
    """
    h, w = image.shape
    bh = h // block_size
    bw = w // block_size

    blocks = image.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)
    return blocks


def merge_blocks(blocks: np.ndarray) -> np.ndarray:
    """
    Собирает изображение из блоков формы:
    (num_blocks_h, num_blocks_w, block_size, block_size)
    """
    bh, bw, bs1, bs2 = blocks.shape
    image = blocks.transpose(0, 2, 1, 3).reshape(bh * bs1, bw * bs2)
    return image


# =========================
# WHT / Hadamard
# =========================

def hadamard_matrix(n: int) -> np.ndarray:
    """
    Рекурсивно строит матрицу Адамара порядка n.
    n должен быть степенью двойки.
    """
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
    """
    Нормированная матрица Адамара:
    H * H^T = I
    """
    h = hadamard_matrix(n)
    return h / math.sqrt(n)


def wht2d(block: np.ndarray, hmat: np.ndarray) -> np.ndarray:
    """
    2D WHT: F = H * X * H^T
    Для нормированной H обратное преобразование такое же.
    """
    return hmat @ block @ hmat.T


def iwht2d(coeffs: np.ndarray, hmat: np.ndarray) -> np.ndarray:
    """
    Обратное 2D WHT.
    Для нормированной матрицы Адамара:
    X = H^T * F * H
    Поскольку H симметрична, H^T = H.
    """
    return hmat.T @ coeffs @ hmat


# =========================
# Квантование
# =========================

def default_wht_quant_matrix(block_size: int = 8) -> np.ndarray:
    """
    Простая базовая матрица квантования для WHT.
    Это стартовый вариант для исследования, не "единственно правильный".
    Значения растут к более "высокочастотным" индексам.
    """
    if block_size != 8:
        # Для других размеров делаем простую возрастающую матрицу
        q = np.fromfunction(lambda i, j: 1 + i + j, (block_size, block_size), dtype=int)
        return q.astype(np.float32)

    q = np.array([
        [8, 10, 12, 14, 18, 22, 26, 30],
        [10, 12, 14, 16, 20, 24, 28, 32],
        [12, 14, 16, 18, 22, 26, 30, 34],
        [14, 16, 18, 20, 24, 28, 32, 36],
        [18, 20, 22, 24, 28, 32, 36, 40],
        [22, 24, 26, 28, 32, 36, 40, 44],
        [26, 28, 30, 32, 36, 40, 44, 48],
        [30, 32, 34, 36, 40, 44, 48, 52],
    ], dtype=np.float32)
    return q


def quality_scale_matrix(qmat: np.ndarray, quality: int) -> np.ndarray:
    """
    Масштабирует матрицу квантования по аналогии с JPEG-подходом.
    quality: 1..100
    Чем выше quality, тем слабее квантование.
    """
    if not (1 <= quality <= 100):
        raise ValueError("quality должен быть в диапазоне 1..100")

    if quality < 50:
        scale = 50.0 / quality
    else:
        scale = 2.0 - (quality / 50.0)

    scaled = np.floor(qmat * scale + 0.5)
    scaled[scaled < 1] = 1
    return scaled.astype(np.float32)


def quantize(block_coeffs: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    """Квантование коэффициентов."""
    return np.round(block_coeffs / qmat).astype(np.int32)


def dequantize(qcoeffs: np.ndarray, qmat: np.ndarray) -> np.ndarray:
    """Обратное квантование."""
    return (qcoeffs * qmat).astype(np.float32)


# =========================
# Метрики
# =========================

def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Среднеквадратичная ошибка."""
    diff = original.astype(np.float32) - reconstructed.astype(np.float32)
    return float(np.mean(diff ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_pixel: float = 255.0) -> float:
    """PSNR в дБ."""
    err = mse(original, reconstructed)
    if err == 0:
        return float("inf")
    return 10.0 * math.log10((max_pixel ** 2) / err)


def nonzero_compression_ratio(original_shape: Tuple[int, int], qcoeffs_blocks: np.ndarray) -> float:
    """
    Упрощённая оценка коэффициента сжатия:
    отношение числа исходных пикселей к числу ненулевых квантованных коэффициентов.
    Это не файловый размер, а исследовательская приближённая метрика.
    """
    original_elements = original_shape[0] * original_shape[1]
    nonzero = np.count_nonzero(qcoeffs_blocks)
    if nonzero == 0:
        return float("inf")
    return original_elements / nonzero


# =========================
# Параметры и результат
# =========================

@dataclass
class WHTCompressionResult:
    reconstructed: np.ndarray
    qcoeffs_blocks: np.ndarray
    quant_matrix: np.ndarray
    original_shape: Tuple[int, int]
    padded_shape: Tuple[int, int]
    encode_time_sec: float
    decode_time_sec: float
    psnr_db: float
    compression_ratio_est: float


# =========================
# Основной алгоритм
# =========================

def compress_decompress_wht(
    image: np.ndarray,
    block_size: int = 8,
    quality: int = 50,
    #quant_matrix: np.ndarray | None = None,
    quant_matrix: Optional[np.ndarray] = None,
    center_pixels: bool = True
) -> WHTCompressionResult:
    """
    JPEG-подобное сжатие/восстановление на основе WHT.

    Параметры:
    - image: полутоновое изображение float32/uint8
    - block_size: размер блока (обычно 8)
    - quality: 1..100
    - quant_matrix: пользовательская матрица квантования
    - center_pixels: вычитать ли 128 перед преобразованием

    Возвращает:
    - восстановленное изображение
    - квантованные коэффициенты
    - метрики
    """
    if image.ndim != 2:
        raise ValueError("Ожидается полутоновое изображение (2D массив).")

    if (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size должен быть степенью двойки для матрицы Адамара.")

    image = image.astype(np.float32)

    padded, original_shape = pad_image_to_block_size(image, block_size)
    padded_shape = padded.shape

    if center_pixels:
        padded = padded - 128.0

    blocks = split_into_blocks(padded, block_size)

    hmat = normalized_hadamard_matrix(block_size)

    if quant_matrix is None:
        quant_matrix = default_wht_quant_matrix(block_size)
    qmat = quality_scale_matrix(quant_matrix, quality)

    bh, bw, _, _ = blocks.shape
    transformed_blocks = np.zeros_like(blocks, dtype=np.float32)
    qcoeffs_blocks = np.zeros_like(blocks, dtype=np.int32)

    # Кодирование
    encode_start = time.perf_counter()
    for i in range(bh):
        for j in range(bw):
            coeffs = wht2d(blocks[i, j], hmat)
            transformed_blocks[i, j] = coeffs
            qcoeffs_blocks[i, j] = quantize(coeffs, qmat)
    encode_time_sec = time.perf_counter() - encode_start

    # Декодирование
    reconstructed_blocks = np.zeros_like(blocks, dtype=np.float32)
    decode_start = time.perf_counter()
    for i in range(bh):
        for j in range(bw):
            deq = dequantize(qcoeffs_blocks[i, j], qmat)
            recon_block = iwht2d(deq, hmat)
            reconstructed_blocks[i, j] = recon_block
    decode_time_sec = time.perf_counter() - decode_start

    reconstructed_padded = merge_blocks(reconstructed_blocks)

    if center_pixels:
        reconstructed_padded = reconstructed_padded + 128.0

    reconstructed = crop_to_original_size(reconstructed_padded, original_shape)
    reconstructed = np.clip(reconstructed, 0, 255)

    psnr_db = psnr(image, reconstructed)
    compression_ratio_est = nonzero_compression_ratio(original_shape, qcoeffs_blocks)

    return WHTCompressionResult(
        reconstructed=reconstructed,
        qcoeffs_blocks=qcoeffs_blocks,
        quant_matrix=qmat,
        original_shape=original_shape,
        padded_shape=padded_shape,
        encode_time_sec=encode_time_sec,
        decode_time_sec=decode_time_sec,
        psnr_db=psnr_db,
        compression_ratio_est=compression_ratio_est,
    )


# =========================
# Пример использования
# =========================

def run_example(input_path: str, output_path: str, quality: int = 50) -> Dict[str, Any]:
    """
    Пример запуска алгоритма.
    """
    image = load_grayscale_image(input_path)

    result = compress_decompress_wht(
        image=image,
        block_size=8,
        quality=quality,
        quant_matrix=None,
        center_pixels=True
    )

    save_grayscale_image(result.reconstructed, output_path)

    stats = {
        "original_shape": result.original_shape,
        "padded_shape": result.padded_shape,
        "quality": quality,
        "psnr_db": result.psnr_db,
        "compression_ratio_est": result.compression_ratio_est,
        "encode_time_sec": result.encode_time_sec,
        "decode_time_sec": result.decode_time_sec,
    }

    return stats


if __name__ == "__main__":
    input_image_path = "input.png"
    output_image_path = "reconstructed_wht.png"

    stats = run_example(
        input_path=input_image_path,
        output_path=output_image_path,
        quality=50
    )

    print("Результаты WHT-сжатия:")
    for key, value in stats.items():
        print(f"{key}: {value}")