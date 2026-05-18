import pandas as pd
import matplotlib.pyplot as plt


def plot_results(csv_path="results_jpeg_8.csv"):
    df = pd.read_csv(csv_path)

    summary = df.groupby(["method", "quality"], as_index=False).agg({
        "psnr": "mean",
        "compression_ratio": "mean",
        "compressed_bytes": "mean",
        "encode_time": "mean",
        "decode_time": "mean",
        "bitstream_length": "mean"
    })

    summary.to_csv("summary_by_quality.csv", index=False)

    # 1. PSNR vs Quality
    plt.figure(figsize=(8, 5))
    for method in summary["method"].unique():
        part = summary[summary["method"] == method]
        plt.plot(part["quality"], part["psnr"], marker="o", label=method)

    plt.xlabel("Quality")
    plt.ylabel("PSNR, dB")
    plt.title("Зависимость PSNR от параметра качества")
    plt.grid(True)
    plt.legend()
    plt.savefig("psnr_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Compression Ratio vs Quality
    plt.figure(figsize=(8, 5))
    for method in summary["method"].unique():
        part = summary[summary["method"] == method]
        plt.plot(part["quality"], part["compression_ratio"], marker="o", label=method)

    plt.xlabel("Quality")
    plt.ylabel("Коэффициент сжатия")
    plt.title("Зависимость коэффициента сжатия от параметра качества")
    plt.grid(True)
    plt.legend()
    plt.savefig("compression_ratio_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Encode Time vs Quality
    plt.figure(figsize=(8, 5))
    for method in summary["method"].unique():
        part = summary[summary["method"] == method]
        plt.plot(part["quality"], part["encode_time"], marker="o", label=method)

    plt.xlabel("Quality")
    plt.ylabel("Время кодирования, с")
    plt.title("Зависимость времени кодирования от параметра качества")
    plt.grid(True)
    plt.legend()
    plt.savefig("encode_time_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 4. Decode Time vs Quality
    plt.figure(figsize=(8, 5))
    for method in summary["method"].unique():
        part = summary[summary["method"] == method]
        plt.plot(part["quality"], part["decode_time"], marker="o", label=method)

    plt.xlabel("Quality")
    plt.ylabel("Время декодирования, с")
    plt.title("Зависимость времени декодирования от параметра качества")
    plt.grid(True)
    plt.legend()
    plt.savefig("decode_time_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 5. PSNR vs Compression Ratio
    plt.figure(figsize=(8, 5))
    for method in summary["method"].unique():
        part = summary[summary["method"] == method]
        plt.plot(part["compression_ratio"], part["psnr"], marker="o", label=method)

    plt.xlabel("Коэффициент сжатия")
    plt.ylabel("PSNR, dB")
    plt.title("Компромисс качество–сжатие")
    plt.grid(True)
    plt.legend()
    plt.savefig("psnr_vs_compression_ratio.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Графики сохранены.")
    print("Сводная таблица сохранена в summary_by_quality.csv")
    print(summary)


if __name__ == "__main__":
    plot_results("results_jpeg_8.csv")
