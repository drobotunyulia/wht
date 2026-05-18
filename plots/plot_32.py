import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "results_32.csv"


def load_summary(csv_file=CSV_FILE):
    df = pd.read_csv(csv_file)

    # убираем WHT 8x8 из финальных графиков
    df = df[df["method"] != "WHT_8_sylvester"]

    summary = df.groupby(["method", "quality"], as_index=False).agg({
        "psnr": "mean",
        "compression_ratio": "mean",
        "reduction_percent": "mean",
        "encode_time": "mean",
        "decode_time": "mean",
        "total_time": "mean",
        "compressed_bytes": "mean",
        "bitstream_length": "mean",
    })

    summary.to_csv("summary_32.csv", index=False)
    return summary


def plot_metric(summary, metric, ylabel, title, filename):
    plt.figure(figsize=(10, 6))

    for method in summary["method"].unique():
        data = summary[summary["method"] == method].sort_values("quality")
        plt.plot(data["quality"], data[metric], marker="o", label=method)

    plt.xlabel("Quality")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_psnr_vs_compression(summary):
    plt.figure(figsize=(9, 6))

    for method in summary["method"].unique():
        data = summary[summary["method"] == method].sort_values("quality")
        plt.plot(
            data["compression_ratio"],
            data["psnr"],
            marker="o",
            label=method
        )

    plt.xlabel("Коэффициент сжатия")
    plt.ylabel("PSNR, дБ")
    plt.title("Компромисс между качеством и коэффициентом сжатия")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("32_psnr_vs_compression.png", dpi=300)
    plt.show()


def main():
    summary = load_summary(CSV_FILE)

    plot_metric(
        summary,
        "psnr",
        "PSNR, дБ",
        "Зависимость PSNR от quality",
        "32_psnr.png"
    )

    plot_metric(
        summary,
        "compression_ratio",
        "Коэффициент сжатия",
        "Зависимость коэффициента сжатия от quality",
        "32_compression_ratio.png"
    )

    plot_metric(
        summary,
        "encode_time",
        "Время кодирования, с",
        "Зависимость времени кодирования от quality",
        "32_encode_time.png"
    )

    plot_metric(
        summary,
        "decode_time",
        "Время декодирования, с",
        "Зависимость времени декодирования от quality",
        "32_decode_time.png"
    )

    plot_metric(
        summary,
        "total_time",
        "Общее время, с",
        "Зависимость общего времени обработки от quality",
        "32_total_time.png"
    )

    plot_psnr_vs_compression(summary)

    print("Готово.")
    print("Сводная таблица сохранена в summary_32.csv")
    print("Графики сохранены:")
    print("32_psnr.png")
    print("32_compression_ratio.png")
    print("32_encode_time.png")
    print("32_decode_time.png")
    print("32_total_time.png")
    print("32_psnr_vs_compression.png")


if __name__ == "__main__":
    main()