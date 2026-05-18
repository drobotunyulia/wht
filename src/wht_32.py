import numpy as np


def sylvester_hadamard_matrix(n: int) -> np.ndarray:
    """
    Строит ненормированную матрицу Адамара конструкции Сильвестра.
    n должен быть степенью двойки.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Размер матрицы Адамара должен быть степенью двойки.")

    h = np.array([[1]], dtype=np.float32)

    while h.shape[0] < n:
        h = np.block([
            [h,  h],
            [h, -h]
        ]).astype(np.float32)

    return h


def normalize_hadamard(H: np.ndarray) -> np.ndarray:
    """
    Нормировка матрицы Адамара:
    H_norm = H / sqrt(n)
    """
    n = H.shape[0]
    return H.astype(np.float32) / np.sqrt(n)


def check_hadamard_matrix(H: np.ndarray) -> bool:
    """
    Проверяет, что матрица является матрицей Адамара:
    H @ H.T = n * I
    """
    if H.ndim != 2:
        return False

    if H.shape[0] != H.shape[1]:
        return False

    n = H.shape[0]

    if not np.all(np.isin(H, [-1, 1])):
        return False

    return np.allclose(H @ H.T, n * np.eye(n))


def bit_reverse(x: int, bits: int) -> int:
    """
    Разворот битов числа x.
    Например, для bits=3:
    3 = 011 -> 110 = 6
    """
    result = 0

    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1

    return result


def gray_code(x: int) -> int:
    """
    Код Грея.
    """
    return x ^ (x >> 1)


def sequency_hadamard_matrix(n: int) -> np.ndarray:
    """
    Строит ненормированную матрицу Адамара с упорядочением строк по sequency.

    Матрица получается перестановкой строк матрицы Сильвестра
    через bit-reversed Gray code.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Размер матрицы должен быть степенью двойки.")

    H = sylvester_hadamard_matrix(n)

    bits = int(np.log2(n))
    order = []

    for s in range(n):
        g = gray_code(s)
        idx = bit_reverse(g, bits)
        order.append(idx)

    return H[order, :]


H_32_SYLVESTER = sylvester_hadamard_matrix(32)

H_32_SEQUENCY = sequency_hadamard_matrix(32)


HADAMARD_32_CLASSES = {
    "sylvester": H_32_SYLVESTER,
    "sequency": H_32_SEQUENCY,
}


def get_hadamard_32(matrix_class: str = "sylvester", normalize: bool = True) -> np.ndarray:
    """
    Возвращает матрицу Адамара 32x32.

    matrix_class:
        "sylvester" — конструкция Сильвестра;
        "sequency"  — строки упорядочены по sequency.

    normalize:
        True  — вернуть H / sqrt(32);
        False — вернуть ненормированную матрицу.
    """
    if matrix_class not in HADAMARD_32_CLASSES:
        raise ValueError(
            f"Неизвестная матрица 32x32: {matrix_class}. "
            f"Доступны: {list(HADAMARD_32_CLASSES.keys())}"
        )

    H = HADAMARD_32_CLASSES[matrix_class]

    if not check_hadamard_matrix(H):
        raise ValueError(f"Матрица {matrix_class} не прошла проверку Адамара.")

    if normalize:
        return normalize_hadamard(H)

    return H


if __name__ == "__main__":
    for name, H in HADAMARD_32_CLASSES.items():
        print(name, check_hadamard_matrix(H))
        print(H.shape)