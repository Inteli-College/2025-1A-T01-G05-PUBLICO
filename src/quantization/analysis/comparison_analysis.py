from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from shared import (
    DEFAULT_IMAGES_ROOT,
    DEFAULT_METRICS_ROOT,
    VARIANTES,
    resolve_existing_dir,
    resolve_existing_file,
    resolve_output_dir,
)


def _carregar_recursos(metrics_root: Path) -> dict[str, pd.DataFrame]:
    recursos: dict[str, pd.DataFrame] = {}
    for variante in VARIANTES:
        caminho = metrics_root / variante / "resources" / "resources.csv"
        csv_path = resolve_existing_file(caminho)
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"O arquivo {caminho} está vazio.")
        recursos[variante] = df.reset_index(drop=True)
    return recursos


def _carregar_predicoes(metrics_root: Path) -> dict[str, pd.DataFrame]:
    predicoes: dict[str, pd.DataFrame] = {}
    for variante in VARIANTES:
        caminho = metrics_root / variante / "preds" / "predictions.csv"
        csv_path = resolve_existing_file(caminho)
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"O arquivo {caminho} está vazio.")
        predicoes[variante] = df.reset_index(drop=True)
    return predicoes


def _plotar_serie_temporal(
    dados_recursos: dict[str, pd.DataFrame],
    metrica: str,
    rotulo_y: str,
    titulo: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    for variante, df in dados_recursos.items():
        indices = range(1, len(df) + 1)
        plt.plot(indices, df[metrica], marker="o", linewidth=1.5, label=variante.title())

    plt.xlabel("Inference index")
    plt.ylabel(rotulo_y)
    plt.title(titulo)
    plt.legend(title="Modelo")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plotar_cdf_confianca(
    dados_predicoes: dict[str, pd.DataFrame], output_path: Path
) -> None:
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    for variante, df in dados_predicoes.items():
        confidences = df["confidence"].dropna().sort_values().reset_index(drop=True)
        if confidences.empty:
            continue
        proporcao = (pd.Series(range(1, len(confidences) + 1), dtype=float)) / len(
            confidences
        )
        plt.plot(confidences, proporcao, linewidth=1.8, label=variante.title())

    plt.xlabel("Confidence")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of Confidence by Model")
    plt.xlim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def gerar_graficos_comparacao(
    metrics_root: Path, output_dir: Path | None = None
) -> list[Path]:
    metrics_root = resolve_existing_dir(metrics_root)
    dados_recursos = _carregar_recursos(metrics_root)
    dados_predicoes = _carregar_predicoes(metrics_root)

    output_dir = resolve_output_dir(output_dir, DEFAULT_IMAGES_ROOT / "comparison")

    caminhos: list[Path] = []

    cpu_path = output_dir / "comparison_cpu_series.png"
    _plotar_serie_temporal(
        dados_recursos,
        "cpu_percent",
        "CPU (%)",
        "Comparison of CPU Usage by Inference",
        cpu_path,
    )
    caminhos.append(cpu_path)

    memoria_path = output_dir / "comparison_memory_series.png"
    _plotar_serie_temporal(
        dados_recursos,
        "memory_mb",
        "Memory (MB)",
        "Comparison of Memory Usage by Inference",
        memoria_path,
    )
    caminhos.append(memoria_path)

    latencia_path = output_dir / "comparison_latency_series.png"
    _plotar_serie_temporal(
        dados_recursos,
        "latency_ms",
        "Latency (ms)",
        "Comparison of Latency by Inference",
        latencia_path,
    )
    caminhos.append(latencia_path)

    confianca_path = output_dir / "comparison_confidence_cdf.png"
    _plotar_cdf_confianca(dados_predicoes, confianca_path)
    caminhos.append(confianca_path)

    return caminhos


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera gráficos comparativos entre modelos original, dinâmico e estático."
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=DEFAULT_METRICS_ROOT,
        help="Diretório raiz contendo as métricas de cada modelo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório para salvar os gráficos comparativos (padrão: images/comparison).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    caminhos = gerar_graficos_comparacao(
        metrics_root=args.metrics_root,
        output_dir=args.output_dir,
    )

    print("Gráficos comparativos gerados:")
    for caminho in caminhos:
        print(f" - {caminho}")


if __name__ == "__main__":
    main()
