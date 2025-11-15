from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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


def _plot_predictions_per_image(df: pd.DataFrame, output_path: Path) -> None:
    per_image = df.groupby("image").size().reset_index(name="total_predicoes")
    per_image = per_image.sort_values("total_predicoes", ascending=False).reset_index(
        drop=True
    )
    per_image["indice_imagem"] = per_image.index + 1

    num_bins = min(50, len(per_image))
    if num_bins <= 0:
        raise ValueError("Não há predições suficientes para gerar o gráfico.")

    per_image["faixa"] = pd.cut(
        per_image["indice_imagem"],
        bins=num_bins,
        labels=range(1, num_bins + 1),
        include_lowest=True,
    )
    agregados = (
        per_image.groupby("faixa", observed=True)["total_predicoes"]
        .sum()
        .reset_index()
        .rename(columns={"faixa": "faixa_indice"})
    )
    agregados["faixa_indice"] = agregados["faixa_indice"].astype(int)
    agregados = agregados.sort_values("faixa_indice")
    agregados["total_predicoes_normalizado"] = (
        agregados["total_predicoes"] / agregados["total_predicoes"].sum()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=agregados,
        x="faixa_indice",
        y="total_predicoes_normalizado",
        color="#4c72b0",
    )
    plt.xlabel("Faixa sequencial de imagens ordenadas")
    plt.ylabel("Proporção de predições")
    plt.title("Distribuição normalizada de predições por faixas de imagens")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))

    xtick_step = max(1, num_bins // 10)
    plt.xticks(range(1, num_bins + 1, xtick_step))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_confidence_distribution(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.histplot(df["confidence"], bins=30, kde=True, color="#1f77b4")
    plt.xlabel("Confiança")
    plt.ylabel("Frequência")
    plt.title("Distribuição da confiança das predições")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def gerar_graficos_predicoes(metrics_dir: Path, output_dir: Path | None = None) -> list[Path]:
    metrics_dir = resolve_existing_dir(metrics_dir)
    predictions_path = resolve_existing_file(metrics_dir / "predictions.csv")

    df = pd.read_csv(predictions_path)
    if df.empty:
        raise ValueError(f"Arquivo {predictions_path} está vazio.")

    output_dir = resolve_output_dir(output_dir, metrics_dir / "images")

    caminhos = []

    per_image_path = output_dir / "predictions_per_image.png"
    _plot_predictions_per_image(df, per_image_path)
    caminhos.append(per_image_path)

    confidence_path = output_dir / "predictions_confidence_distribution.png"
    _plot_confidence_distribution(df, confidence_path)
    caminhos.append(confidence_path)

    return caminhos


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gera gráficos a partir das métricas de predições exportadas pelo benchmark."
        )
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help=(
            "Diretório contendo o arquivo predictions.csv. "
            "Use em conjunto com --metrics-root/--variant para localizar automaticamente."
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
        help="Variante do modelo cuja pasta de predições será analisada (original/dynamic/static).",
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
        "preds",
    )

    caminhos = gerar_graficos_predicoes(
        metrics_dir=metrics_dir,
        output_dir=args.output_dir,
    )

    print("Gráficos gerados:")
    for caminho in caminhos:
        print(f" - {caminho}")


if __name__ == "__main__":
    main()

