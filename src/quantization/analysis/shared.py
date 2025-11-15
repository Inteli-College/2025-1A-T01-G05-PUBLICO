from __future__ import annotations

from pathlib import Path
from typing import Iterable

VARIANTES: tuple[str, ...] = ("original", "dynamic", "static")
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_ROOT: Path = PROJECT_ROOT / "metrics"
DEFAULT_IMAGES_ROOT: Path = PROJECT_ROOT / "images"


def _candidate_paths(path: Path) -> Iterable[Path]:
    yield (Path.cwd() / path).resolve()
    yield (PROJECT_ROOT / path).resolve()


def resolve_existing_dir(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate
    else:
        for possibility in _candidate_paths(candidate):
            if possibility.exists():
                resolved = possibility
                break
        else:
            resolved = (Path.cwd() / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Esperava um diretório, mas encontrei: {resolved}")
    return resolved


def resolve_existing_file(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate
    else:
        for possibility in _candidate_paths(candidate):
            if possibility.exists():
                resolved = possibility
                break
        else:
            resolved = (Path.cwd() / candidate).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {resolved}")
    return resolved


def resolve_output_dir(output_dir: Path | str | None, default: Path) -> Path:
    target = Path(output_dir) if output_dir is not None else default
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_metrics_subdir(
    metrics_dir: Path | None,
    metrics_root: Path,
    variant: str | None,
    *relative_parts: str,
) -> Path:
    if metrics_dir is not None:
        return resolve_existing_dir(metrics_dir)
    if variant is None:
        options = ", ".join(VARIANTES)
        raise ValueError(
            f"Informe --variant (opções: {options}) ou --metrics-dir para localizar as métricas."
        )
    root_dir = resolve_existing_dir(metrics_root)
    target = root_dir.joinpath(variant, *relative_parts)
    return resolve_existing_dir(target)


