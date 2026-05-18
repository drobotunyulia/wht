import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

from jpeg import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    subsample_420,
    upsample_420,
    pad_to_multiple,
    scale_quant_table,
    QY_STD,
    QC_STD,
    build_huffman_codebook,
    huffman_encode,
    HuffmanCodebook,
    EOB,
    ZRL,
    magnitude_category,
    amplitude_bits,
    bits_to_amplitude,
)

from classes_16 import HADAMARD_16_CLASSES
from classes_24 import HADAMARD_24_CLASSES
from wht_32 import HADAMARD_32_CLASSES


# ============================================================
# 1. Матрицы Адамара и WHT
# ============================================================

def sylvester_hadamard_matrix(n: int) -> np.ndarray:
    """
    Строит нормированную матрицу Адамара конструкции Сильвестра.
    n должен быть степенью двойки: 2, 4, 8, 16, ...
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


def normalize_hadamard(H: np.ndarray) -> np.ndarray:
    """
    Нормирует матрицу Адамара:
    H_norm = H / sqrt(n)
    """
    n = H.shape[0]
    return H.astype(np.float32) / np.sqrt(n)


def check_hadamard_matrix(H: np.ndarray) -> bool:
    """
    Проверяет, что матрица является матрицей Адамара:
    - квадратная;
    - состоит только из +1 и -1;
    - H @ H.T = n * I.
    """
    if H.ndim != 2:
        return False

    if H.shape[0] != H.shape[1]:
        return False

    n = H.shape[0]

    if not np.all(np.isin(H, [-1, 1])):
        return False

    return np.allclose(H @ H.T, n * np.eye(n))


def get_hadamard_matrix(block_size: int, matrix_class: str = "sylvester") -> np.ndarray:
    """
    Возвращает нормированную матрицу Адамара для WHT.

    Для block_size=8:
        используется только матрица Сильвестра.

    Для block_size=16:
        используются только представители пяти классов:
        "class1", "class2", "class3", "class4", "class5".
    """
    if block_size == 8:
        if matrix_class != "sylvester":
            raise ValueError(
                "Для block_size=8 доступна только матрица 'sylvester'."
            )
        return sylvester_hadamard_matrix(8)

    if block_size == 16:
        if matrix_class == "sylvester":
            raise ValueError(
                "Для block_size=16 используйте только классы: "
                "'class1', 'class2', 'class3', 'class4', 'class5'."
            )

        if matrix_class not in HADAMARD_16_CLASSES:
            raise ValueError(
                f"Неизвестный класс матрицы: {matrix_class}. "
                f"Доступны: {list(HADAMARD_16_CLASSES.keys())}"
            )

        H = HADAMARD_16_CLASSES[matrix_class]

        if not check_hadamard_matrix(H):
            raise ValueError(f"Матрица {matrix_class} не прошла проверку Адамара.")

        return normalize_hadamard(H)

    if block_size == 24:
        if matrix_class not in HADAMARD_24_CLASSES:
            raise ValueError(
                f"Для block_size=24 доступна только матрица: "
                f"{list(HADAMARD_24_CLASSES.keys())}"
            )

        H = HADAMARD_24_CLASSES[matrix_class]

        if not check_hadamard_matrix(H):
            raise ValueError(f"Матрица {matrix_class} не прошла проверку Адамара.")

        return normalize_hadamard(H)

    if block_size == 32:
        if matrix_class not in HADAMARD_32_CLASSES:
            raise ValueError(
                f"Для block_size=32 доступны только: "
                f"{list(HADAMARD_32_CLASSES.keys())}"
            )

        H = HADAMARD_32_CLASSES[matrix_class]

        if not check_hadamard_matrix(H):
            raise ValueError(
                f"Матрица 32x32 '{matrix_class}' "
                f"не прошла проверку Адамара."
            )

        return normalize_hadamard(H)

    raise ValueError("Поддерживаются block_size=8, block_size=16 и block_size=24.")


def wht2(block: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Двумерное преобразование Уолша-Адамара:
    F = H * X * H^T
    """
    return H @ block @ H.T


def iwht2(coeff: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Обратное двумерное преобразование Уолша-Адамара:
    X = H^T * F * H
    """
    return H.T @ coeff @ H


# ============================================================
# 2. Блоки произвольного размера
# ============================================================

def image_to_blocks_n(img: np.ndarray, block_size: int) -> np.ndarray:
    H, W = img.shape

    assert H % block_size == 0 and W % block_size == 0

    return img.reshape(
        H // block_size,
        block_size,
        W // block_size,
        block_size
    ).transpose(0, 2, 1, 3)


def blocks_to_image_n(blocks: np.ndarray) -> np.ndarray:
    nH, nW, bs1, bs2 = blocks.shape

    return blocks.transpose(0, 2, 1, 3).reshape(
        nH * bs1,
        nW * bs2
    )


# ============================================================
# 3. Zigzag для произвольного размера блока
# ============================================================

def zigzag_indices(n: int) -> List[Tuple[int, int]]:
    idx = []

    for s in range(2 * n - 1):
        if s % 2 == 0:
            x = min(s, n - 1)
            y = s - x

            while x >= 0 and y < n:
                idx.append((x, y))
                x -= 1
                y += 1
        else:
            y = min(s, n - 1)
            x = s - y

            while y >= 0 and x < n:
                idx.append((x, y))
                x += 1
                y -= 1

    return idx


def block_to_zigzag_n(block: np.ndarray) -> np.ndarray:
    n = block.shape[0]
    zz = zigzag_indices(n)

    return np.array([block[i, j] for i, j in zz], dtype=np.int32)


def zigzag_to_block_n(vec: np.ndarray, block_size: int) -> np.ndarray:
    block = np.zeros((block_size, block_size), dtype=np.int32)
    zz = zigzag_indices(block_size)

    for k, (i, j) in enumerate(zz):
        block[i, j] = vec[k]

    return block


# ============================================================
# 4. Матрицы квантования для WHT 8x8 и 16x16
# ============================================================

def base_wht_quant_matrix(block_size: int, chroma: bool = False, quant_mode: str = "wht") -> np.ndarray:
    """
    Матрица квантования для WHT.

    quant_mode:
        "wht"  — текущая возрастающая матрица для WHT;
        "jpeg" — JPEG-матрица, адаптированная под размер блока.
    """

    base_q = QC_STD if chroma else QY_STD

    if quant_mode == "jpeg":
        if block_size == 8:
            return base_q.astype(np.int32)

        if block_size == 16:
            return np.tile(base_q, (2, 2)).astype(np.int32)

        if block_size == 24:
            return np.tile(base_q, (3, 3)).astype(np.int32)

        if block_size == 32:
            return np.tile(base_q, (4, 4)).astype(np.int32)

    if quant_mode == "wht":
        if block_size == 8:
            return base_q.astype(np.int32)

        if block_size == 16:
            Q = np.fromfunction(
                lambda i, j: 8 + 2.2 * (i + j),
                (16, 16),
                dtype=np.float32
            )

            if chroma:
                Q = Q * 1.4

            return np.clip(np.round(Q), 1, 255).astype(np.int32)

        if block_size == 24:
            Q = np.fromfunction(
                lambda i, j: 8 + 1.6 * (i + j),
                (24, 24),
                dtype=np.float32
            )

            if chroma:
                Q = Q * 1.4

            return np.clip(np.round(Q), 1, 255).astype(np.int32)

        if block_size == 32:
            Q = np.fromfunction(
                lambda i, j: 8 + 1.2 * (i + j),
                (32, 32),
                dtype=np.float32
            )

            if chroma:
                Q = Q * 1.4

            return np.clip(np.round(Q), 1, 255).astype(np.int32)

    raise ValueError("Некорректные block_size или quant_mode.")


# ============================================================
# 5. DC/AC кодирование для произвольного размера блока
# ============================================================

def encode_block_coeffs_n(
    zz: np.ndarray,
    prev_dc: int
) -> Tuple[List[Any], int]:
    dc = int(zz[0])
    diff = dc - prev_dc

    dc_size = magnitude_category(diff)
    symbols = [("DC", dc_size, amplitude_bits(diff, dc_size))]

    run = 0

    for k in range(1, len(zz)):
        a = int(zz[k])

        if a == 0:
            run += 1

            if run == 16:
                symbols.append(ZRL)
                run = 0
        else:
            size = magnitude_category(a)
            symbols.append(("AC", run, size, amplitude_bits(a, size)))
            run = 0

    if run > 0:
        symbols.append(EOB)

    return symbols, dc


def decode_block_coeffs_n(
    symbols: List[Any],
    prev_dc: int,
    block_size: int
) -> Tuple[np.ndarray, int]:
    total_len = block_size * block_size
    zz = np.zeros(total_len, dtype=np.int32)

    sym = symbols[0]
    assert sym[0] == "DC"

    diff = bits_to_amplitude(sym[2])
    dc = prev_dc + diff
    zz[0] = dc

    idx = 1

    for sym in symbols[1:]:
        if sym == EOB:
            break

        if sym == ZRL:
            idx += 16
            continue

        assert sym[0] == "AC"

        run, size, bits = sym[1], sym[2], sym[3]

        idx += run

        if idx >= total_len:
            break

        zz[idx] = bits_to_amplitude(bits)
        idx += 1

        if idx >= total_len:
            break

    return zz, dc


# ============================================================
# 6. Huffman decode для произвольного размера блока
# ============================================================

def invert_codebook(cb: HuffmanCodebook) -> Dict[str, Any]:
    return {v: k for k, v in cb.codes.items()}


def huffman_decode_n(
    bitstream: str,
    cb: HuffmanCodebook,
    n_blocks: int,
    block_size: int
) -> List[List[Any]]:
    inv = invert_codebook(cb)

    blocks = []
    i = 0
    total_len = block_size * block_size

    def read_huff_symbol() -> Any:
        nonlocal i

        acc = ""

        while i < len(bitstream):
            acc += bitstream[i]
            i += 1

            if acc in inv:
                return inv[acc]

        raise ValueError("Unexpected end of bitstream")

    def read_bits(k: int) -> str:
        nonlocal i

        if k == 0:
            return ""

        bits = bitstream[i:i + k]

        if len(bits) != k:
            raise ValueError("Unexpected end of bitstream amplitude")

        i += k

        return bits

    for _ in range(n_blocks):
        block_syms = []

        sym = read_huff_symbol()

        if sym[0] != "DC":
            raise ValueError("Expected DC symbol")

        dc_size = sym[1]
        amp = read_bits(dc_size)

        block_syms.append(("DC", dc_size, amp))

        filled = 1

        while filled < total_len:
            sym = read_huff_symbol()

            if sym == EOB:
                block_syms.append(EOB)
                break

            if sym == ZRL:
                block_syms.append(ZRL)
                filled += 16
                continue

            run, size = sym[1], sym[2]
            amp = read_bits(size)

            block_syms.append(("AC", run, size, amp))

            filled += run + 1

        blocks.append(block_syms)

    return blocks


# ============================================================
# 7. Контейнер WHT
# ============================================================

@dataclass
class MiniWHT:
    height: int
    width: int
    quality: int
    subsampling: str
    block_size: int
    matrix_class: str
    quant_mode: str
    qy: np.ndarray
    qc: np.ndarray
    huff: HuffmanCodebook
    bitstream: str
    layout: Dict[str, Tuple[int, int]]


# ============================================================
# 8. Кодирование MiniWHT
# ============================================================

def encode_miniwht(
    img_rgb: np.ndarray,
    quality: int = 75,
    subsampling: str = "420",
    block_size: int = 8,
    matrix_class: str = "sylvester",
    quant_mode: str = "wht"
) -> MiniWHT:
    """
    JPEG-подобное кодирование изображения на основе WHT.

    block_size:
        8 или 16

    matrix_class:
    для block_size=8 используется "sylvester";
    для block_size=16 используются:
    "class1", "class2", "class3", "class4", "class5".
    """
    if block_size not in (8, 16, 24, 32):
        raise ValueError("Поддерживаются только block_size=8, block_size=16 и block_size=24.")

    H0, W0, _ = img_rgb.shape

    mult = block_size * 2 if subsampling == "420" else block_size

    Y, Cb, Cr = rgb_to_ycbcr(img_rgb)

    Y = pad_to_multiple(Y, mult, mult)
    Cb = pad_to_multiple(Cb, mult, mult)
    Cr = pad_to_multiple(Cr, mult, mult)

    if subsampling == "420":
        Cb_s = subsample_420(Cb)
        Cr_s = subsample_420(Cr)
    else:
        Cb_s, Cr_s = Cb, Cr

    qy_base = base_wht_quant_matrix(block_size, chroma=False, quant_mode=quant_mode)
    qc_base = base_wht_quant_matrix(block_size, chroma=True, quant_mode=quant_mode)

    qy = scale_quant_table(qy_base, quality)
    qc = scale_quant_table(qc_base, quality)

    Yb = image_to_blocks_n(Y, block_size)
    Cbb = image_to_blocks_n(Cb_s, block_size)
    Crb = image_to_blocks_n(Cr_s, block_size)

    layout = {
        "Y": (Yb.shape[0], Yb.shape[1]),
        "Cb": (Cbb.shape[0], Cbb.shape[1]),
        "Cr": (Crb.shape[0], Crb.shape[1]),
    }

    Hmat = get_hadamard_matrix(block_size, matrix_class)

    all_symbols = []
    prev_dc = {"Y": 0, "Cb": 0, "Cr": 0}

    def process_channel(blocks: np.ndarray, Q: np.ndarray, name: str):
        nonlocal all_symbols

        for by in range(blocks.shape[0]):
            for bx in range(blocks.shape[1]):
                block = blocks[by, bx].astype(np.float32) - 128.0

                F = wht2(block, Hmat)
                q = np.round(F / Q).astype(np.int32)

                zz = block_to_zigzag_n(q)

                syms, new_prev = encode_block_coeffs_n(
                    zz,
                    prev_dc[name]
                )

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
        block_size=block_size,
        matrix_class=matrix_class,
        quant_mode=quant_mode,
        qy=qy,
        qc=qc,
        huff=huff,
        bitstream=bitstream,
        layout=layout
    )


# ============================================================
# 9. Декодирование MiniWHT
# ============================================================

def decode_miniwht(container: MiniWHT) -> np.ndarray:
    block_size = container.block_size
    qy = container.qy
    qc = container.qc

    nY = container.layout["Y"][0] * container.layout["Y"][1]
    nC = container.layout["Cb"][0] * container.layout["Cb"][1]
    n_blocks = nY + nC + nC

    blocks_syms = huffman_decode_n(
        container.bitstream,
        container.huff,
        n_blocks,
        block_size
    )

    Hmat = get_hadamard_matrix(
        container.block_size,
        container.matrix_class
    )

    prev_dc = {"Y": 0, "Cb": 0, "Cr": 0}

    def reconstruct_blocks(
        nh: int,
        nw: int,
        Q: np.ndarray,
        name: str,
        start_idx: int
    ):
        blocks = np.zeros((nh, nw, block_size, block_size), dtype=np.float32)
        idx = start_idx

        for by in range(nh):
            for bx in range(nw):
                syms = blocks_syms[idx]
                idx += 1

                zz, new_prev = decode_block_coeffs_n(
                    syms,
                    prev_dc[name],
                    block_size
                )

                prev_dc[name] = new_prev

                qblock = zigzag_to_block_n(zz, block_size).astype(np.float32)

                F = qblock * Q

                block = iwht2(F, Hmat) + 128.0

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

    Y = blocks_to_image_n(Yb)
    Cb = blocks_to_image_n(Cbb)
    Cr = blocks_to_image_n(Crb)

    if container.subsampling == "420":
        Cb = upsample_420(Cb)
        Cr = upsample_420(Cr)

    Y = Y[:container.height, :container.width]
    Cb = Cb[:container.height, :container.width]
    Cr = Cr[:container.height, :container.width]

    return ycbcr_to_rgb(Y, Cb, Cr)


# ============================================================
# 10. Быстрый тест
# ============================================================

if __name__ == "__main__":
    from PIL import Image
    import time

    img = Image.open("data/airplane2.tif").convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    tests = [
        (8, "sylvester"),
        (16, "class1"),
        (24, "paley"),
        (32, "sylvester"),
        (32, "sequency")
    ]

    for bs, cls in tests:
        start = time.perf_counter()

        encoded = encode_miniwht(
            arr,
            quality=30,
            subsampling="420",
            block_size=bs,
            matrix_class=cls
        )

        encode_time = time.perf_counter() - start

        start = time.perf_counter()
        recon = decode_miniwht(encoded)
        decode_time = time.perf_counter() - start

        Image.fromarray(recon).save(f"reconstructed_wht_{bs}_{cls}.png")

        original_bytes = arr.shape[0] * arr.shape[1] * 3
        compressed_bytes = len(encoded.bitstream) / 8
        cr = original_bytes / compressed_bytes

        print(f"\n=== MiniWHT block_size={bs}, matrix_class={cls} ===")
        print("Bitstream length:", len(encoded.bitstream), "bits")
        print("Approx bytes:", compressed_bytes)
        print(f"Compression ratio: {cr:.2f} : 1")
        print(f"Encode time: {encode_time:.6f} sec")
        print(f"Decode time: {decode_time:.6f} sec")
