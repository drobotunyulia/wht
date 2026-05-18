import pandas as pd
import matplotlib.pyplot as plt


def plot_quality20(csv_path="results_all_classes.csv"):
    df = pd.read_csv(csv_path)

    df20 = df[df["quality"] == 20].copy()

    summary = df20.groupby("method", as_index=False).agg({
        "psnr": "mean",
        "compression_ratio": "mean",
        "compressed_bytes": "mean",
        "encode_time": "mean",
        "decode_time": "mean",
        "bitstream_length": "mean"
    })

    method_order = [
        "MiniJPEG_DCT",
        "MiniJPEG_WHT_8_sylvester",
        "MiniJPEG_WHT_16_class1",
        "MiniJPEG_WHT_16_class2",
        "MiniJPEG_WHT_16_class3",
        "MiniJPEG_WHT_16_class4",
        "MiniJPEG_WHT_16_class5",
    ]

    summary["method"] = pd.Categorical(
        summary["method"],
        categories=method_order,
        ordered=True
    )
    summary = summary.sort_values("method")

    summary.to_csv("summary_quality20.csv", index=False)

    def bar_plot(metric, ylabel, title, filename):
        plt.figure(figsize=(11, 6))
        plt.bar(summary["method"].astype(str), summary[metric])

        plt.xlabel("Метод")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.4)

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()

    bar_plot(
        "psnr",
        "PSNR, dB",
        "Среднее значение PSNR при quality=20",
        "quality20_psnr_by_method.png"
    )

    bar_plot(
        "compression_ratio",
        "Коэффициент сжатия",
        "Средний коэффициент сжатия при quality=20",
        "quality20_compression_ratio_by_method.png"
    )

    bar_plot(
        "compressed_bytes",
        "Размер сжатого потока, байт",
        "Средний размер сжатого потока при quality=20",
        "quality20_compressed_bytes_by_method.png"
    )

    bar_plot(
        "encode_time",
        "Время кодирования, с",
        "Среднее время кодирования при quality=20",
        "quality20_encode_time_by_method.png"
    )

    bar_plot(
        "decode_time",
        "Время декодирования, с",
        "Среднее время декодирования при quality=20",
        "quality20_decode_time_by_method.png"
    )

    # Компромисс качество–сжатие
    plt.figure(figsize=(9, 6))
    plt.scatter(summary["compression_ratio"], summary["psnr"])

    for _, row in summary.iterrows():
        label = str(row["method"]).replace("MiniJPEG_", "")
        plt.annotate(
            label,
            (row["compression_ratio"], row["psnr"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel("Коэффициент сжатия")
    plt.ylabel("PSNR, dB")
    plt.title("Компромисс качество–сжатие при quality=20")
    plt.grid(True, alpha=0.4)

    plt.savefig("quality20_psnr_vs_compression_ratio.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Графики сохранены.")
    print("Сводная таблица сохранена в summary_quality20.csv")
    print(summary)


if __name__ == "__main__":
    plot_quality20("results_all_classes.csv")