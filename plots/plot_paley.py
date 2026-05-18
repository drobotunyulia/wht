import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "results_paley.csv"


def prepare_data(csv_file=CSV_FILE):
    df = pd.read_csv(csv_file)

    # усредняем значения по всем изображениям
    summary = df.groupby(["method", "quality"], as_index=False).agg({
        "psnr": "mean",
        "compression_ratio": "mean",
        "encode_time": "mean",
        "decode_time": "mean",
        "compressed_bytes": "mean",
        "bitstream_length": "mean"
    })

    return summary


def plot_metric(summary, metric, ylabel, title, output_file):
    plt.figure(figsize=(10, 6))

    for method in summary["method"].unique():
        data = summary[summary["method"] == method]
        data = data.sort_values("quality")

        plt.plot(
            data["quality"],
            data[metric],
            marker="o",
            label=method
        )

    plt.xlabel("Quality")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.show()


def main():
    summary = prepare_data(CSV_FILE)

    summary.to_csv("summary_paley.csv", index=False)

    plot_metric(
        summary,
        metric="psnr",
        ylabel="PSNR, дБ",
        title="Зависимость PSNR от параметра качества",
        output_file="paley_psnr.png"
    )

    plot_metric(
        summary,
        metric="compression_ratio",
        ylabel="Коэффициент сжатия",
        title="Зависимость коэффициента сжатия от параметра качества",
        output_file="paley_compression_ratio.png"
    )

    plot_metric(
        summary,
        metric="encode_time",
        ylabel="Время кодирования, с",
        title="Зависимость времени кодирования от параметра качества",
        output_file="paley_encode_time.png"
    )

    plot_metric(
        summary,
        metric="decode_time",
        ylabel="Время декодирования, с",
        title="Зависимость времени декодирования от параметра качества",
        output_file="paley_decode_time.png"
    )

    # график компромисса качество-сжатие
    plt.figure(figsize=(9, 6))

    for method in summary["method"].unique():
        data = summary[summary["method"] == method]
        plt.plot(
            data["compression_ratio"],
            data["psnr"],
            marker="o",
            label=method
        )

    plt.xlabel("Коэффициент сжатия")
    plt.ylabel("PSNR, дБ")
    plt.title("Компромисс между качеством и сжатием")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("paley_psnr_vs_compression.png", dpi=300)
    plt.show()

    print("Готово.")
    print("Сводная таблица сохранена в summary_paley.csv")
    print("Графики сохранены:")
    print("paley_psnr.png")
    print("paley_compression_ratio.png")
    print("paley_encode_time.png")
    print("paley_decode_time.png")
    print("paley_psnr_vs_compression.png")


if __name__ == "__main__":
    main()