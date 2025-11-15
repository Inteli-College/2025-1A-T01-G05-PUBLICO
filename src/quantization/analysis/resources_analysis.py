from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from shared import (
    DEFAULT_METRICS_ROOT,
    VARIANTES,
    get_metrics_subdir,
    resolve_existing_dir,
    resolve_existing_file,
    resolve_output_dir,
)


def _plot_time_series(df: pd.DataFrame, output_path: Path) -> None:
    indices = range(1, len(df) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(indices, df["latency_ms"], marker="o", label="Latency (ms)")
    plt.plot(indices, df["cpu_percent"], marker="o", label="CPU (%)")
    plt.plot(indices, df["memory_mb"], marker="o", label="Memory (MB)")
    plt.xlabel("Inference index")
    plt.ylabel("Value")
    plt.title("Resources by inference")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_distribution(df: pd.DataFrame, column: str, label: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=25, kde=True, color="#ff7f0e")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {label.lower()}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def gerar_graficos_recursos(metrics_dir: Path, output_dir: Path | None = None) -> list[Path]:
    metrics_dir = resolve_existing_dir(metrics_dir)
    resources_path = resolve_existing_file(metrics_dir / "resources.csv")

    df = pd.read_csv(resources_path)
    if df.empty:
        raise ValueError(f"Arquivo {resources_path} está vazio.")

    output_dir = resolve_output_dir(output_dir, metrics_dir / "images")

    caminhos = []

    time_series_path = output_dir / "resources_time_series.png"
    _plot_time_series(df, time_series_path)
    caminhos.append(time_series_path)

    latency_dist_path = output_dir / "resources_latency_distribution.png"
    _plot_distribution(df, "latency_ms", "Latency (ms)", latency_dist_path)
    caminhos.append(latency_dist_path)

    cpu_dist_path = output_dir / "resources_cpu_distribution.png"
    _plot_distribution(df, "cpu_percent", "CPU Usage (%)", cpu_dist_path)
    caminhos.append(cpu_dist_path)

    memory_dist_path = output_dir / "resources_memory_distribution.png"
    _plot_distribution(df, "memory_mb", "Memory (MB)", memory_dist_path)
    caminhos.append(memory_dist_path)

    return caminhos


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera gráficos a partir das métricas de recursos exportadas pelo benchmark."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help=(
            "Diretório contendo o arquivo resources.csv. "
            "Use em conjunto com --metrics-root/--variant se preferir apontar para a raiz."
        ),
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=DEFAULT_METRICS_ROOT,
        help="Diretório raiz com as métricas exportadas (por padrão, <projeto>/metrics).",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTES,
        default=None,
        help="Variante do modelo cuja pasta de recursos será analisada (original/dynamic/static).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório para salvar os gráficos (padrão: metrics_dir/images).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    metrics_dir = get_metrics_subdir(
        args.metrics_dir,
        args.metrics_root,
        args.variant,
        "resources",
    )

    caminhos = gerar_graficos_recursos(
        metrics_dir=metrics_dir,
        output_dir=args.output_dir,
    )

    print("Gráficos gerados:")
    for caminho in caminhos:
        print(f" - {caminho}")


if __name__ == "__main__":
    main()

