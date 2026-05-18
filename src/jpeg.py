import numpy as np
from dataclasses import dataclass
from collections import Counter
import heapq
from typing import Dict, List, Tuple, Any

# ============================================================
# 1) Цветовые преобразования RGB <-> YCbCr (BT.601, full range)
# ============================================================

def rgb_to_ycbcr(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    img_rgb: uint8, shape (H, W, 3), RGB 0..255
    Returns: Y, Cb, Cr as float32 (still in 0..255-ish range)
    """
    img = img_rgb.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    # BT.601 (популярная формула)
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0
    return Y, Cb, Cr

def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Inputs: float arrays
    Output: uint8 RGB image
    """
    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0

    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb

    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

# ============================================================
# 2) Субдискретизация 4:2:0 (простая: average 2x2 + upsample)
# ============================================================

def subsample_420(C: np.ndarray) -> np.ndarray:
    """
    Downsample by 2 in both directions using 2x2 averaging.
    C: (H, W) float
    Return: (H/2, W/2)
    """
    H, W = C.shape
    assert H % 2 == 0 and W % 2 == 0
    C00 = C[0::2, 0::2]
    C01 = C[0::2, 1::2]
    C10 = C[1::2, 0::2]
    C11 = C[1::2, 1::2]
    return (C00 + C01 + C10 + C11) / 4.0

def upsample_420(Cs: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor upsample back to full resolution.
    Cs: (H/2, W/2)
    Return: (H, W)
    """
    return np.repeat(np.repeat(Cs, 2, axis=0), 2, axis=1)

# ============================================================
# 3) Подготовка: паддинг до кратности 8 (и 16 для 4:2:0)
# ============================================================

def pad_to_multiple(img: np.ndarray, mult_h: int, mult_w: int, mode="edge") -> np.ndarray:
    """
    Pad 2D array to multiples of mult_h, mult_w.
    mode='edge' повторяет крайние значения (часто хорошо для кодеков)
    """
    H, W = img.shape
    H2 = int(np.ceil(H / mult_h) * mult_h)
    W2 = int(np.ceil(W / mult_w) * mult_w)
    pad_h = H2 - H
    pad_w = W2 - W
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, ((0, pad_h), (0, pad_w)), mode=mode)

# ============================================================
# 4) DCT через матрицу преобразования (точно как в теории)
#    D = T @ block @ T.T
# ============================================================

def dct_matrix(N: int = 8) -> np.ndarray:
    """
    Возвращает матрицу T размера NxN для DCT-II с ортонормировкой:
    T[u, x] = alpha(u) * cos(pi/N * (x + 1/2) * u)
    """
    T = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        alpha = np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)
        for x in range(N):
            T[u, x] = alpha * np.cos((np.pi / N) * (x + 0.5) * u)
    return T

T8 = dct_matrix(8)
T8_T = T8.T

def dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT: F = T * f * T^T"""
    return T8 @ block @ T8_T

def idct2(coeff: np.ndarray) -> np.ndarray:
    """Inverse 2D DCT: f = T^T * F * T"""
    return T8_T @ coeff @ T8

# ============================================================
# 5) Таблицы квантования (базовые JPEG, качество можно масштабировать)
# ============================================================

QY_STD = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.int32)

QC_STD = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
], dtype=np.int32)

def scale_quant_table(Q: np.ndarray, quality: int) -> np.ndarray:
    """
    Масштабирование таблицы квантования как в популярных реализациях JPEG.
    quality: 1..100
    """
    quality = int(np.clip(quality, 1, 100))
    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality
    Qs = np.floor((S * Q + 50) / 100).astype(np.int32)
    Qs = np.clip(Qs, 1, 255)
    return Qs

# ============================================================
# 6) Zigzag порядок (8x8 -> 64)
# ============================================================

def zigzag_indices(n=8) -> List[Tuple[int, int]]:
    idx = []
    for s in range(2*n - 1):
        if s % 2 == 0:
            # even: go down-left
            x = min(s, n-1)
            y = s - x
            while x >= 0 and y < n:
                idx.append((x, y))
                x -= 1
                y += 1
        else:
            # odd: go up-right
            y = min(s, n-1)
            x = s - y
            while y >= 0 and x < n:
                idx.append((x, y))
                x += 1
                y -= 1
    return idx

ZZ = zigzag_indices(8)

def block_to_zigzag(block: np.ndarray) -> np.ndarray:
    return np.array([block[i, j] for (i, j) in ZZ], dtype=np.int32)

def zigzag_to_block(vec: np.ndarray) -> np.ndarray:
    block = np.zeros((8, 8), dtype=np.int32)
    for k, (i, j) in enumerate(ZZ):
        block[i, j] = vec[k]
    return block

# ============================================================
# 7) Категории величины (SIZE) как в JPEG: сколько бит нужно на амплитуду
# ============================================================

def magnitude_category(x: int) -> int:
    if x == 0:
        return 0
    return int(np.floor(np.log2(abs(x))) + 1)

def amplitude_bits(x: int, size: int) -> str:
    """
    JPEG кодирует амплитуду:
    - положительное: обычный двоичный код
    - отрицательное: инверсия (2^size - 1 + x) (эквивалентно JPEG "ones complement" трюку)
    """
    if size == 0:
        return ""
    if x >= 0:
        return format(x, f"0{size}b")
    # для отрицательных:
    val = (1 << size) - 1 + x  # x отрицательное
    return format(val, f"0{size}b")

def bits_to_amplitude(bits: str) -> int:
    if not bits:
        return 0
    size = len(bits)
    v = int(bits, 2)
    # если старший бит 1 -> положительное
    if v >= (1 << (size - 1)):
        return v
    # иначе отрицательное
    return v - ((1 << size) - 1)

# ============================================================
# 8) RLE для AC коэффициентов + DC дифференцирование
# ============================================================

EOB = ("EOB",)     # End of Block
ZRL = ("ZRL",)     # Zero Run Length (16 zeros)

def encode_block_coeffs(zz: np.ndarray, prev_dc: int) -> Tuple[List[Any], int]:
    """
    zz: 64 ints in zigzag order (quantized).
    Returns:
      symbols: список "символов" для Хаффмана + отдельные биты амплитуд
      new_prev_dc: DC текущего блока
    Представление символов сделаем так:
      - DC: ("DC", size, amp_bits)
      - AC: ("AC", run, size, amp_bits)
      - EOB, ZRL как отдельные маркеры
    """
    dc = int(zz[0])
    diff = dc - prev_dc
    dc_size = magnitude_category(diff)
    symbols = [("DC", dc_size, amplitude_bits(diff, dc_size))]

    # AC coefficients:
    run = 0
    for k in range(1, 64):
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

def decode_block_coeffs(symbols: List[Any], prev_dc: int) -> Tuple[np.ndarray, int]:
    """
    Обратно из символов -> zigzag vector длины 64.
    """
    zz = np.zeros(64, dtype=np.int32)

    # DC должен быть первым
    sym = symbols[0]
    assert sym[0] == "DC"
    size = sym[1]
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
        # AC
        assert sym[0] == "AC"
        run, size, bits = sym[1], sym[2], sym[3]
        idx += run
        zz[idx] = bits_to_amplitude(bits)
        idx += 1
        if idx >= 64:
            break

    return zz, dc

# ============================================================
# 9) Хаффман для наших символов
#    (мы кодируем "символ" без амплитудных битов, а амплитуды приклеиваем)
# ============================================================

@dataclass
class HuffmanCodebook:
    codes: Dict[Any, str]  # symbol -> bitstring

def build_huffman_codebook(symbols: List[Any]) -> HuffmanCodebook:
    """
    Строим Хаффман по частотам "символов", где:
      DC символ = ("DC", dc_size)
      AC символ = ("AC", run, size)
      EOB, ZRL как есть
    Амплитудные биты НЕ участвуют в выборе символа (как в JPEG).
    """
    def symbol_key(s):
        if s == EOB or s == ZRL:
            return s
        if s[0] == "DC":
            return ("DC", s[1])
        if s[0] == "AC":
            return ("AC", s[1], s[2])
        raise ValueError(s)

    freq = Counter(symbol_key(s) for s in symbols)

    # Классический Хаффман через очередь
    heap = [[w, [sym, ""]] for sym, w in freq.items()]
    heapq.heapify(heap)

    # крайний случай: один символ
    if len(heap) == 1:
        w, [sym, _] = heap[0]
        return HuffmanCodebook({sym: "0"})

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    _, *pairs = heap[0]
    codes = {sym: code for sym, code in pairs}
    return HuffmanCodebook(codes)

def huffman_encode(symbols: List[Any], cb: HuffmanCodebook) -> str:
    """
    Возвращает одну большую строку бит (для учебных целей).
    """
    out_bits = []
    for s in symbols:
        if s == EOB or s == ZRL:
            key = s
            out_bits.append(cb.codes[key])
        elif s[0] == "DC":
            key = ("DC", s[1])
            out_bits.append(cb.codes[key])
            out_bits.append(s[2])  # amplitude bits
        else:  # AC
            key = ("AC", s[1], s[2])
            out_bits.append(cb.codes[key])
            out_bits.append(s[3])  # amplitude bits
    return "".join(out_bits)

# Для декодирования Хаффмана сделаем обратную таблицу
def invert_codebook(cb: HuffmanCodebook) -> Dict[str, Any]:
    return {v: k for k, v in cb.codes.items()}

def huffman_decode(bitstream: str, cb: HuffmanCodebook, n_blocks: int) -> List[List[Any]]:
    """
    Декодируем обратно в список блоков, каждый блок = список символов (с амплитудными битами).
    Нам нужно знать n_blocks (сколько блоков ожидаем).
    """
    inv = invert_codebook(cb)

    blocks = []
    i = 0

    def read_huff_symbol() -> Any:
        nonlocal i
        # читаем биты пока не совпадет с кодом
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
        bits = bitstream[i:i+k]
        if len(bits) != k:
            raise ValueError("Unexpected end of bitstream (amplitude)")
        i += k
        return bits

    for _ in range(n_blocks):
        block_syms = []

        # DC
        sym = read_huff_symbol()
        if sym[0] != "DC":
            raise ValueError("Expected DC symbol")
        dc_size = sym[1]
        amp = read_bits(dc_size)
        block_syms.append(("DC", dc_size, amp))

        # AC until EOB or filled
        filled = 1
        while filled < 64:
            sym = read_huff_symbol()
            if sym == EOB:
                block_syms.append(EOB)
                break
            if sym == ZRL:
                block_syms.append(ZRL)
                filled += 16
                continue
            # AC
            run, size = sym[1], sym[2]
            amp = read_bits(size)
            block_syms.append(("AC", run, size, amp))
            filled += run + 1

        blocks.append(block_syms)

    return blocks

# ============================================================
# 10) Разбиение на блоки 8x8 и сборка обратно
# ============================================================

def image_to_blocks(img: np.ndarray) -> np.ndarray:
    """
    img: 2D float, shape (H, W), where H,W multiples of 8
    Returns: blocks shape (nH, nW, 8, 8)
    """
    H, W = img.shape
    assert H % 8 == 0 and W % 8 == 0
    return img.reshape(H//8, 8, W//8, 8).transpose(0, 2, 1, 3)

def blocks_to_image(blocks: np.ndarray) -> np.ndarray:
    """
    blocks: (nH, nW, 8, 8)
    Returns: (H, W)
    """
    nH, nW, _, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(nH*8, nW*8)

# ============================================================
# 11) Основной encode/decode (мини-JPEG)
# ============================================================

@dataclass
class MiniJPEG:
    height: int
    width: int
    quality: int
    subsampling: str  # "444" or "420"
    qy: np.ndarray
    qc: np.ndarray
    huff: HuffmanCodebook
    bitstream: str
    # Нужно знать, сколько блоков и в каком порядке мы их складывали:
    layout: Dict[str, Tuple[int, int]]  # channel -> (nH_blocks, nW_blocks)

def encode_minijpeg(img_rgb: np.ndarray, quality: int = 75, subsampling: str = "420") -> MiniJPEG:
    """
    Кодирует изображение (учебный формат).
    subsampling: "444" или "420"
    """
    H0, W0, _ = img_rgb.shape

    # Для 4:2:0 нужно, чтобы размеры были кратны 16 (потому что хрома вдвое меньше,
    # а мы хотим, чтобы и хрома тоже делилась на 8).
    if subsampling == "420":
        mult = 16
    else:
        mult = 8

    Y, Cb, Cr = rgb_to_ycbcr(img_rgb)

    Y  = pad_to_multiple(Y,  mult, mult)
    Cb = pad_to_multiple(Cb, mult, mult)
    Cr = pad_to_multiple(Cr, mult, mult)

    if subsampling == "420":
        Cb_s = subsample_420(Cb)
        Cr_s = subsample_420(Cr)
    else:
        Cb_s, Cr_s = Cb, Cr

    # Квант. таблицы
    qy = scale_quant_table(QY_STD, quality)
    qc = scale_quant_table(QC_STD, quality)

    # Преобразуем в блоки 8×8
    Yb  = image_to_blocks(Y)
    Cbb = image_to_blocks(Cb_s)
    Crb = image_to_blocks(Cr_s)

    layout = {
        "Y":  (Yb.shape[0],  Yb.shape[1]),
        "Cb": (Cbb.shape[0], Cbb.shape[1]),
        "Cr": (Crb.shape[0], Crb.shape[1]),
    }

    # Идём по блокам в фиксированном порядке: все Y, потом Cb, потом Cr
    all_symbols = []
    prev_dc = {"Y": 0, "Cb": 0, "Cr": 0}

    def process_channel(blocks: np.ndarray, Q: np.ndarray, name: str):
        nonlocal all_symbols
        for by in range(blocks.shape[0]):
            for bx in range(blocks.shape[1]):
                block = blocks[by, bx].astype(np.float32) - 128.0  # level shift

                F = dct2(block)  # DCT
                q = np.round(F / Q).astype(np.int32)  # quantize

                zz = block_to_zigzag(q)
                syms, new_prev = encode_block_coeffs(zz, prev_dc[name])
                prev_dc[name] = new_prev

                all_symbols.extend(syms)

    process_channel(Yb,  qy, "Y")
    process_channel(Cbb, qc, "Cb")
    process_channel(Crb, qc, "Cr")

    # Хаффман по всему потоку (для простоты — единый кодбук)
    huff = build_huffman_codebook(all_symbols)
    bitstream = huffman_encode(all_symbols, huff)

    return MiniJPEG(
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

def decode_minijpeg(container: MiniJPEG) -> np.ndarray:
    """
    Возвращает восстановленное RGB uint8.
    """
    qy, qc = container.qy, container.qc
    subsampling = container.subsampling

    nY = container.layout["Y"][0] * container.layout["Y"][1]
    nC = container.layout["Cb"][0] * container.layout["Cb"][1]
    n_blocks = nY + nC + nC  # Y + Cb + Cr

    # Декодируем Хаффман -> получаем список блоков символов
    blocks_syms = huffman_decode(container.bitstream, container.huff, n_blocks)

    # Восстанавливаем блоки коэффициентов и затем пиксели через IDCT
    prev_dc = {"Y": 0, "Cb": 0, "Cr": 0}

    def reconstruct_blocks(nh: int, nw: int, Q: np.ndarray, name: str, start_idx: int) -> Tuple[np.ndarray, int]:
        blocks = np.zeros((nh, nw, 8, 8), dtype=np.float32)
        idx = start_idx
        for by in range(nh):
            for bx in range(nw):
                syms = blocks_syms[idx]
                idx += 1

                zz, new_prev = decode_block_coeffs(syms, prev_dc[name])
                prev_dc[name] = new_prev

                qblock = zigzag_to_block(zz).astype(np.float32)
                F = qblock * Q  # dequantize
                block = idct2(F) + 128.0  # IDCT + unshift

                blocks[by, bx] = block
        return blocks, idx

    idx = 0
    Yb, idx  = reconstruct_blocks(container.layout["Y"][0],  container.layout["Y"][1],  qy, "Y",  idx)
    Cbb, idx = reconstruct_blocks(container.layout["Cb"][0], container.layout["Cb"][1], qc, "Cb", idx)
    Crb, idx = reconstruct_blocks(container.layout["Cr"][0], container.layout["Cr"][1], qc, "Cr", idx)

    # Склеиваем блоки обратно в картинки
    Y  = blocks_to_image(Yb)
    Cb = blocks_to_image(Cbb)
    Cr = blocks_to_image(Crb)

    # Если 4:2:0 — апсемплим хрому
    if subsampling == "420":
        Cb = upsample_420(Cb)
        Cr = upsample_420(Cr)

    # Обрезаем паддинг до исходного размера
    Y  = Y[:container.height, :container.width]
    Cb = Cb[:container.height, :container.width]
    Cr = Cr[:container.height, :container.width]

    return ycbcr_to_rgb(Y, Cb, Cr)

# ============================================================
# 12) Пример использования
# ============================================================

if __name__ == "__main__":
    from PIL import Image

    img = Image.open("input.png").convert("RGB")
    arr = np.array(img)

    encoded = encode_minijpeg(arr, quality=90, subsampling="420")
    recon = decode_minijpeg(encoded)

    Image.fromarray(recon).save("reconstructed.png")

    # "оценка" размера: длина битового потока
    print("Bitstream length:", len(encoded.bitstream), "bits")
    print("Approx bytes:", len(encoded.bitstream) / 8)

    # =============================
    # Сравнение сжатия
    # =============================

    # исходный размер (в памяти)
    H, W, _ = arr.shape
    original_bytes = H * W * 3

    # размер после сжатия
    compressed_bytes = len(encoded.bitstream) / 8

    # коэффициент сжатия
    compression_ratio = original_bytes / compressed_bytes

    # процент уменьшения
    reduction_percent = 100 * (1 - compressed_bytes / original_bytes)

    print("\n=== Compression Stats ===")
    print(f"Original size:   {original_bytes:.0f} bytes ({original_bytes / 1024:.2f} KB)")
    print(f"Compressed size: {compressed_bytes:.0f} bytes ({compressed_bytes / 1024:.2f} KB)")
    print(f"Compression ratio: {compression_ratio:.2f} : 1")
    print(f"Size reduction: {reduction_percent:.2f}%")

