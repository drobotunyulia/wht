import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(summary, metric, ylabel, title, output_name):
    plt.figure(figsize=(10, 6))

    for method in summary["method"].unique():
        part = summary[summary["method"] == method]
        plt.plot(part["quality"], part[metric], marker="o", label=method)

    plt.xlabel("Quality")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_results(csv_path="results_all_classes.csv"):
    df = pd.read_csv(csv_path)

    summary = df.groupby(["method", "quality"], as_index=False).agg({
        "psnr": "mean",
        "compression_ratio": "mean",
        "compressed_bytes": "mean",
        "encode_time": "mean",
        "decode_time": "mean",
        "bitstream_length": "mean"
    })

    method_summary = df.groupby("method", as_index=False).agg({
        "psnr": "mean",
        "compression_ratio": "mean",
        "compressed_bytes": "mean",
        "encode_time": "mean",
        "decode_time": "mean"
    })

    summary.to_csv("summary_by_quality_all_classes.csv", index=False)
    method_summary.to_csv("summary_by_method_all_classes.csv", index=False)

    plot_metric(
        summary,
        "psnr",
        "PSNR, dB",
        "Зависимость PSNR от параметра качества",
        "psnr_vs_quality_all_classes.png"
    )

    plot_metric(
        summary,
        "compression_ratio",
        "Коэффициент сжатия",
        "Зависимость коэффициента сжатия от параметра качества",
        "compression_ratio_vs_quality_all_classes.png"
    )

    plot_metric(
        summary,
        "compressed_bytes",
        "Размер сжатого потока, байт",
        "Зависимость размера сжатого потока от параметра качества",
        "compressed_bytes_vs_quality_all_classes.png"
    )

    plot_metric(
        summary,
        "encode_time",
        "Время кодирования, с",
        "Зависимость времени кодирования от параметра качества",
        "encode_time_vs_quality_all_classes.png"
    )

    plot_metric(
        summary,
        "decode_time",
        "Время декодирования, с",
        "Зависимость времени декодирования от параметра качества",
        "decode_time_vs_quality_all_classes.png"
    )

    plt.figure(figsize=(10, 6))

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
    plt.legend(fontsize=8)
    plt.savefig("psnr_vs_compression_ratio_all_classes.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n=== Средние значения по методам ===")
    print(method_summary.sort_values("psnr", ascending=False))


if __name__ == "__main__":
    plot_results("results_all_classes.csv")