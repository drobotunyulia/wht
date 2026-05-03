import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from jpeg import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    subsample_420,
    upsample_420,
    pad_to_multiple,
    scale_quant_table,
    QY_STD,
    QC_STD,
    image_to_blocks,
    blocks_to_image,
    block_to_zigzag,
    zigzag_to_block,
    encode_block_coeffs,
    decode_block_coeffs,
    build_huffman_codebook,
    huffman_encode,
    huffman_decode,
    HuffmanCodebook,
)


# ============================================================
# 1. Матрица Адамара и WHT
# ============================================================

def hadamard_matrix(n: int) -> np.ndarray:
    """
    Строит нормированную матрицу Адамара размера n x n.
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

    return h / np.sqrt(n)


H8 = hadamard_matrix(8)
H8_T = H8.T


def wht2(block: np.ndarray) -> np.ndarray:
    """
    Двумерное преобразование Уолша-Адамара.
    """
    return H8 @ block @ H8_T


def iwht2(coeff: np.ndarray) -> np.ndarray:
    """
    Обратное двумерное преобразование Уолша-Адамара.
    """
    return H8_T @ coeff @ H8


# ============================================================
# 2. Контейнер для WHT-кодека
# ============================================================

@dataclass
class MiniWHT:
    height: int
    width: int
    quality: int
    subsampling: str
    qy: np.ndarray
    qc: np.ndarray
    huff: HuffmanCodebook
    bitstream: str
    layout: Dict[str, Tuple[int, int]]


# ============================================================
# 3. Кодирование MiniJPEG-WHT
# ============================================================

def encode_miniwht(
    img_rgb: np.ndarray,
    quality: int = 75,
    subsampling: str = "420"
) -> MiniWHT:
    """
    JPEG-подобное кодирование изображения,
    где DCT заменено на WHT.
    """
    H0, W0, _ = img_rgb.shape

    if subsampling == "420":
        mult = 16
    else:
        mult = 8

    # RGB -> YCbCr
    Y, Cb, Cr = rgb_to_ycbcr(img_rgb)

    # Padding
    Y = pad_to_multiple(Y, mult, mult)
    Cb = pad_to_multiple(Cb, mult, mult)
    Cr = pad_to_multiple(Cr, mult, mult)

    # 4:2:0
    if subsampling == "420":
        Cb_s = subsample_420(Cb)
        Cr_s = subsample_420(Cr)
    else:
        Cb_s, Cr_s = Cb, Cr

    # Таблицы квантования
    qy = scale_quant_table(QY_STD, quality)
    qc = scale_quant_table(QC_STD, quality)

    # Разбиение на блоки
    Yb = image_to_blocks(Y)
    Cbb = image_to_blocks(Cb_s)
    Crb = image_to_blocks(Cr_s)

    layout = {
        "Y": (Yb.shape[0], Yb.shape[1]),
        "Cb": (Cbb.shape[0], Cbb.shape[1]),
        "Cr": (Crb.shape[0], Crb.shape[1]),
    }

    all_symbols = []
    prev_dc = {"Y": 0, "Cb": 0, "Cr": 0}

    def process_channel(blocks: np.ndarray, Q: np.ndarray, name: str):
        nonlocal all_symbols

        for by in range(blocks.shape[0]):
            for bx in range(blocks.shape[1]):
                block = blocks[by, bx].astype(np.float32) - 128.0

                # Главное отличие от JPEG:
                # здесь используется WHT вместо DCT
                F = wht2(block)

                q = np.round(F / Q).astype(np.int32)

                zz = block_to_zigzag(q)

                syms, new_prev = encode_block_coeffs(zz, prev_dc[name])
                prev_dc[name] = new_prev

                all_symbols.extend(syms)

    process_channel(Yb, qy, "Y")
    process_channel(Cbb, qc, "Cb")
    process_channel(Crb, qc, "Cr")

    huff = build_huffman_codebook(all_symbols)
    bitstream = huffman_encode(all_symbols, huff)

    return MiniWHT(
        height=H0,
        width=W0,
        quality=quality,
        subsampling=subsampling,
        qy=qy,
        qc=qc,
        huff=huff,
        bitstream=bitstream,
        layout=layout
    )


# ============================================================
# 4. Декодирование MiniJPEG-WHT
# ============================================================

def decode_miniwht(container: MiniWHT) -> np.ndarray:
    """
    Декодирование JPEG-подобного WHT-потока.
    """
    qy = container.qy
    qc = container.qc
    subsampling = container.subsampling

    nY = container.layout["Y"][0] * container.layout["Y"][1]
    nC = container.layout["Cb"][0] * container.layout["Cb"][1]
    n_blocks = nY + nC + nC

    blocks_syms = huffman_decode(
        container.bitstream,
        container.huff,
        n_blocks
    )

    prev_dc = {"Y": 0, "Cb": 0, "Cr": 0}

    def reconstruct_blocks(
        nh: int,
        nw: int,
        Q: np.ndarray,
        name: str,
        start_idx: int
    ):
        blocks = np.zeros((nh, nw, 8, 8), dtype=np.float32)
        idx = start_idx

        for by in range(nh):
            for bx in range(nw):
                syms = blocks_syms[idx]
                idx += 1

                zz, new_prev = decode_block_coeffs(
                    syms,
                    prev_dc[name]
                )
                prev_dc[name] = new_prev

                qblock = zigzag_to_block(zz).astype(np.float32)

                # Обратное квантование
                F = qblock * Q

                # Главное отличие от JPEG:
                # здесь используется обратное WHT вместо IDCT
                block = iwht2(F) + 128.0

                blocks[by, bx] = block

        return blocks, idx

    idx = 0

    Yb, idx = reconstruct_blocks(
        container.layout["Y"][0],
        container.layout["Y"][1],
        qy,
        "Y",
        idx
    )

    Cbb, idx = reconstruct_blocks(
        container.layout["Cb"][0],
        container.layout["Cb"][1],
        qc,
        "Cb",
        idx
    )

    Crb, idx = reconstruct_blocks(
        container.layout["Cr"][0],
        container.layout["Cr"][1],
        qc,
        "Cr",
        idx
    )

    # Сборка каналов
    Y = blocks_to_image(Yb)
    Cb = blocks_to_image(Cbb)
    Cr = blocks_to_image(Crb)

    # Upsampling для 4:2:0
    if subsampling == "420":
        Cb = upsample_420(Cb)
        Cr = upsample_420(Cr)

    # Обрезаем padding
    Y = Y[:container.height, :container.width]
    Cb = Cb[:container.height, :container.width]
    Cr = Cr[:container.height, :container.width]

    return ycbcr_to_rgb(Y, Cb, Cr)


# ============================================================
# 5. Тест одного изображения
# ============================================================

if __name__ == "__main__":
    from PIL import Image
    import time

    img = Image.open("data/airplane1.tif").convert("RGB")
    arr = np.array(img)

    start = time.perf_counter()
    encoded = encode_miniwht(arr, quality=60, subsampling="420")
    encode_time = time.perf_counter() - start

    start = time.perf_counter()
    recon = decode_miniwht(encoded)
    decode_time = time.perf_counter() - start

    Image.fromarray(recon).save("reconstructed_wht.png")

    original_bytes = arr.shape[0] * arr.shape[1] * 3
    compressed_bytes = len(encoded.bitstream) / 8
    compression_ratio = original_bytes / compressed_bytes

    print("=== MiniWHT test ===")
    print("Bitstream length:", len(encoded.bitstream), "bits")
    print("Approx bytes:", compressed_bytes)
    print(f"Compression ratio: {compression_ratio:.2f} : 1")
    print(f"Encode time: {encode_time:.6f} sec")
    print(f"Decode time: {decode_time:.6f} sec")