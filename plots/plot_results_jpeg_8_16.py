import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(summary, metric, ylabel, title, output_name):
    plt.figure(figsize=(8, 5))

    for method in summary["method"].unique():
        part = summary[summary["method"] == method]
        plt.plot(part["quality"], part[metric], marker="o", label=method)

    plt.xlabel("Quality")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_results(csv_path="results.csv"):
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

    plot_metric(
        summary,
        metric="psnr",
        ylabel="PSNR, dB",
        title="Зависимость PSNR от параметра качества",
        output_name="psnr_vs_quality.png"
    )

    plot_metric(
        summary,
        metric="compression_ratio",
        ylabel="Коэффициент сжатия",
        title="Зависимость коэффициента сжатия от параметра качества",
        output_name="compression_ratio_vs_quality.png"
    )

    plot_metric(
        summary,
        metric="compressed_bytes",
        ylabel="Размер сжатого потока, байт",
        title="Зависимость размера сжатого потока от параметра качества",
        output_name="compressed_bytes_vs_quality.png"
    )

    plot_metric(
        summary,
        metric="encode_time",
        ylabel="Время кодирования, с",
        title="Зависимость времени кодирования от параметра качества",
        output_name="encode_time_vs_quality.png"
    )

    plot_metric(
        summary,
        metric="decode_time",
        ylabel="Время декодирования, с",
        title="Зависимость времени декодирования от параметра качества",
        output_name="decode_time_vs_quality.png"
    )

    # Компромисс качество–сжатие
    plt.figure(figsize=(8, 5))

    for method in summary["method"].unique():
        part = summary[summary["method"] == method]
        plt.plot(
            part["compression_ratio"],
            part["psnr"],
            marker="o",
            label=method
        )

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
    plot_results("results_jpeg_8_16.csv")