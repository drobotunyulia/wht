import pandas as pd
import matplotlib.pyplot as plt


def plot_bar(summary, metric, ylabel, title, output_name):
    plt.figure(figsize=(8, 5))

    plt.bar(summary["method"], summary[metric])

    plt.xlabel("Метод")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.4)

    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_quality50(csv_path="results_jpegq_paley.csv"):
    df = pd.read_csv(csv_path)

    # Берём только quality=50
    df50 = df[df["quality"] == 50].copy()

    # Оставляем только DCT и WHT 16 class1
    methods = [
        "DCT_JPEG_Q",
        "WHT_16_class1_JPEG_Q"
    ]

    df50 = df50[df50["method"].isin(methods)]

    summary = df50.groupby("method", as_index=False).agg({
        "psnr": "mean",
        "compression_ratio": "mean",
        "compressed_bytes": "mean",
        "encode_time": "mean",
        "decode_time": "mean",
        "bitstream_length": "mean"
    })

    summary["total_time"] = summary["encode_time"] + summary["decode_time"]

    summary["method"] = pd.Categorical(
        summary["method"],
        categories=methods,
        ordered=True
    )
    summary = summary.sort_values("method")

    summary.to_csv("summary_quality50_dct_wht16.csv", index=False)

    plot_bar(
        summary,
        "psnr",
        "PSNR, dB",
        "Среднее значение PSNR при quality=50",
        "quality50_psnr_dct_vs_wht16.png"
    )

    plot_bar(
        summary,
        "compression_ratio",
        "Коэффициент сжатия",
        "Средний коэффициент сжатия при quality=50",
        "quality50_compression_ratio_dct_vs_wht16.png"
    )

    plot_bar(
        summary,
        "compressed_bytes",
        "Размер сжатого потока, байт",
        "Средний размер сжатого потока при quality=50",
        "quality50_compressed_bytes_dct_vs_wht16.png"
    )

    plot_bar(
        summary,
        "encode_time",
        "Время кодирования, с",
        "Среднее время кодирования при quality=50",
        "quality50_encode_time_dct_vs_wht16.png"
    )

    plot_bar(
        summary,
        "decode_time",
        "Время декодирования, с",
        "Среднее время декодирования при quality=50",
        "quality50_decode_time_dct_vs_wht16.png"
    )

    plot_bar(
        summary,
        "total_time",
        "Суммарное время, с",
        "Среднее суммарное время кодирования и декодирования при quality=50",
        "quality50_total_time_dct_vs_wht16.png"
    )

    # Компромисс качество–сжатие
    plt.figure(figsize=(7, 5))
    plt.scatter(summary["compression_ratio"], summary["psnr"])

    for _, row in summary.iterrows():
        plt.annotate(
            row["method"],
            (row["compression_ratio"], row["psnr"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9
        )

    plt.xlabel("Коэффициент сжатия")
    plt.ylabel("PSNR, dB")
    plt.title("Компромисс качество–сжатие при quality=50")
    plt.grid(True, alpha=0.4)

    plt.savefig("quality50_psnr_vs_compression_ratio_dct_vs_wht16.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Графики сохранены.")
    print("Сводная таблица сохранена в summary_quality50_dct_wht16.csv")
    print(summary)


if __name__ == "__main__":
    plot_quality50("results_jpegq_paley.csv")