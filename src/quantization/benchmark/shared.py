from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from psutil import Process
from tqdm import tqdm
from ultralytics import YOLO


@dataclass
class Measurement:
    image_name: str
    latency_ms: float
    cpu_percent: float
    memory_mb: float


@dataclass
class VideoMeasurement:
    frame_number: int
    latency_ms: float
    cpu_percent: float
    memory_mb: float


@dataclass
class BenchmarkReport:
    model_name: str
    num_images: int
    warmup_runs: int
    summary: dict
    detailed: List[Measurement]

    def to_dict(self, include_detailed: bool = False) -> dict:
        payload = {
            "model_name": self.model_name,
            "num_images": self.num_images,
            "warmup_runs": self.warmup_runs,
            "summary": self.summary,
        }
        if include_detailed:
            payload["per_image"] = [asdict(item) for item in self.detailed]
        return payload


def build_parser(
    description: str,
    default_model_format: str,
    default_model_path: Path,
    default_image_dir: Path,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model-format",
        choices=("yolo", "onnx"),
        default=default_model_format,
        help="Formato do modelo a ser avaliado.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path,
        help="Caminho para o arquivo do modelo.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Diretório com as imagens de teste.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Dispositivo utilizado na inferência (somente para modelos YOLO).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de imagens a serem utilizadas no benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Quantidade de inferências iniciais para aquecimento (sem registro de métricas).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Extensões de arquivos consideradas como imagens.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desativa a barra de progresso.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("metrics"),
        help="Diretório para exportar as métricas de predições em CSV.",
    )
    parser.add_argument(
        "--resource-metrics-dir",
        type=Path,
        default=Path("metrics"),
        help="Diretório para exportar métricas de recursos (CPU/memória/latência).",
    )
    parser.add_argument(
        "--export-predictions",
        action="store_true",
        help="Exporta métricas detalhadas das predições em CSV.",
    )
    parser.add_argument(
        "--export-resources",
        action="store_true",
        help="Exporta métricas de recursos (CPU/memória/latência) em CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Arquivo para salvar os resultados em formato JSON.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Exibe o resultado completo em JSON no stdout.",
    )
    parser.add_argument(
        "--include-detailed",
        action="store_true",
        help="Inclui resultados por imagem no JSON.",
    )
    return parser


def validate_paths(image_dir: Path, model_path: Path) -> None:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")


def list_image_paths(image_dir: Path, extensions: Sequence[str]) -> List[Path]:
    normalized_exts = tuple(ext.lower() for ext in extensions)
    return [
        path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in normalized_exts
    ]


def process_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
    return image


def create_onnx_runner(model_path: Path) -> Callable[[Path], None]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def _run(image_path: Path) -> None:
        image_tensor = process_image(image_path)
        session.run([output_name], {input_name: image_tensor})

    return _run


def create_yolo_runner(model_path: Path, device: str) -> Callable[[Path], None]:
    yolo_model = YOLO(str(model_path))

    def _run(image_path: Path) -> None:
        yolo_model.predict(source=str(image_path), device=device, verbose=False)

    return _run


def warmup_model(run_inference: Callable[[Path], None], image_paths: Sequence[Path], warmup_runs: int) -> None:
    if warmup_runs <= 0:
        return
    for image_path in image_paths[: min(warmup_runs, len(image_paths))]:
        run_inference(image_path)


def collect_measurements(
    run_inference: Callable[[Path], None],
    image_paths: Iterable[Path],
    show_progress: bool,
    process: Process,
) -> List[Measurement]:
    process.cpu_percent(interval=None)

    iterator = image_paths
    if show_progress:
        iterator = tqdm(image_paths, desc="Benchmarking", unit="image")

    measurements: List[Measurement] = []

    for image_path in iterator:
        start = perf_counter()
        run_inference(image_path)
        latency = (perf_counter() - start) * 1000

        cpu_percent = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024**2)

        measurements.append(
            Measurement(
                image_name=image_path.name,
                latency_ms=latency,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
            )
        )

    return measurements


def summarize(metric_values: Sequence[float]) -> dict:
    return {
        "min": min(metric_values),
        "avg": mean(metric_values),
        "max": max(metric_values),
    }


def build_summary(measurements: Sequence[Measurement]) -> dict:
    latencies = [item.latency_ms for item in measurements]
    cpu_values = [item.cpu_percent for item in measurements]
    mem_values = [item.memory_mb for item in measurements]

    return {
        "latency_ms": summarize(latencies),
        "cpu_percent": summarize(cpu_values),
        "memory_mb": summarize(mem_values),
    }


def format_summary(report: BenchmarkReport) -> str:
    lines = [
        f"Modelo: {report.model_name}",
        f"Imagens avaliadas: {report.num_images}",
        f"Warmup: {report.warmup_runs}",
        "",
        "Latência (ms): "
        f"mín={report.summary['latency_ms']['min']:.2f}, "
        f"médio={report.summary['latency_ms']['avg']:.2f}, "
        f"máx={report.summary['latency_ms']['max']:.2f}",
        "CPU (%): "
        f"mín={report.summary['cpu_percent']['min']:.2f}, "
        f"médio={report.summary['cpu_percent']['avg']:.2f}, "
        f"máx={report.summary['cpu_percent']['max']:.2f}",
        "Memória (MB): "
        f"mín={report.summary['memory_mb']['min']:.2f}, "
        f"médio={report.summary['memory_mb']['avg']:.2f}, "
        f"máx={report.summary['memory_mb']['max']:.2f}",
    ]
    return "\n".join(lines)


def _prepare_images(image_dir: Path, extensions: Sequence[str], limit: int | None) -> List[Path]:
    image_paths = list_image_paths(image_dir, extensions)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem encontrada em {image_dir}")
    return image_paths


def _select_runner(model_format: str, model_path: Path, device: str) -> Tuple[Callable[[Path], None], str]:
    if model_format == "yolo":
        run_inference = create_yolo_runner(model_path, device)
        model_name = f"YOLO::{model_path.name}"
    else:
        run_inference = create_onnx_runner(model_path)
        model_name = f"ONNX::{model_path.name}"
    return run_inference, model_name


def execute_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    predictions_dir = args.predictions_dir.resolve()
    resource_dir = args.resource_metrics_dir.resolve()
    validate_paths(args.image_dir, args.model_path)
    image_paths = _prepare_images(args.image_dir, args.extensions, args.limit)

    run_inference, model_name = _select_runner(args.model_format, args.model_path, args.device)

    warmup_model(run_inference, image_paths, args.warmup)

    process = Process(os.getpid())
    process.cpu_percent(interval=None)

    measurements = collect_measurements(
        run_inference=run_inference,
        image_paths=image_paths,
        show_progress=not args.no_progress,
        process=process,
    )

    summary = build_summary(measurements)
    if args.export_predictions:
        export_prediction_metrics(
            model_path=args.model_path,
            image_paths=image_paths,
            device=args.device,
            metrics_root=predictions_dir,
            show_progress=not args.no_progress,
        )
    if args.export_resources:
        export_resource_metrics(
            measurements=measurements,
            metrics_root=resource_dir,
        )
    return BenchmarkReport(
        model_name=model_name,
        num_images=len(image_paths),
        warmup_runs=args.warmup,
        summary=summary,
        detailed=measurements,
    )


def export_report(
    report: BenchmarkReport,
    print_json: bool,
    output_json: Path | None,
    include_detailed: bool,
) -> None:
    if not (print_json or output_json):
        return

    json_payload = json.dumps(report.to_dict(include_detailed=include_detailed), indent=2)
    if print_json:
        print("\nJSON:")
        print(json_payload)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json_payload, encoding="utf-8")


def export_prediction_metrics(
    model_path: Path,
    image_paths: Sequence[Path],
    device: str,
    metrics_root: Path,
    show_progress: bool,
) -> None:
    metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_root / "predictions.csv"

    try:
        predictor = YOLO(str(model_path))
    except Exception as exc:
        print(f"Não foi possível carregar o modelo para exportar métricas ({model_path}): {exc}")
        return

    iterator: Iterable[Path] = image_paths
    if show_progress:
        iterator = tqdm(image_paths, desc="Exportando métricas", unit="image")

    records: List[dict] = []
    for image_path in iterator:
        results = predictor.predict(
            source=str(image_path),
            device=device,
            verbose=False,
            save=False,
        )
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            confs = result.boxes.conf.tolist()
            classes = result.boxes.cls.tolist()
            for box, conf, cls in zip(boxes, confs, classes):
                records.append(
                    {
                        "image": image_path.name,
                        "class_id": int(cls),
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "confidence": conf,
                    }
                )

    df = pd.DataFrame(
        records,
        columns=["image", "class_id", "x1", "y1", "x2", "y2", "confidence"],
    )
    df.to_csv(csv_path, index=False)


def export_resource_metrics(
    measurements: Sequence[Measurement],
    metrics_root: Path,
) -> None:
    metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_root / "resources.csv"

    df = pd.DataFrame(
        [
            {
                "image": item.image_name,
                "latency_ms": item.latency_ms,
                "cpu_percent": item.cpu_percent,
                "memory_mb": item.memory_mb,
            }
            for item in measurements
        ],
        columns=["image", "latency_ms", "cpu_percent", "memory_mb"],
    )
    df.to_csv(csv_path, index=False)


def run_cli(
    description: str,
    default_model_format: str,
    default_model_path: Path,
    default_image_dir: Path,
) -> None:
    parser = build_parser(
        description=description,
        default_model_format=default_model_format,
        default_model_path=default_model_path,
        default_image_dir=default_image_dir,
    )
    args = parser.parse_args()
    report = execute_benchmark(args)
    print(format_summary(report))
    export_report(
        report=report,
        print_json=args.print_json,
        output_json=args.output_json,
        include_detailed=args.include_detailed,
    )

def list_video_devices(max_index: int = 50) -> list[dict]:
    """Lista todos os dispositivos de vídeo disponíveis e suas informações.
    
    Args:
        max_index: Índice máximo a ser testado.
    
    Returns:
        Lista de dicionários com informações sobre cada dispositivo encontrado.
    """
    devices = []
    cv2.setLogLevel(0)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        for index in range(max_index):
            try:
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    ret, frame = cap.read()
                    working = ret and frame is not None and frame.size > 0
                    
                    devices.append({
                        'index': index,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'working': working,
                    })
                cap.release()
            except:
                continue
    
    return devices


def find_available_camera(max_index: int = 50, prefer_working: bool = True) -> Optional[int]:
    """Procura por uma câmera disponível testando índices de 0 até max_index.
    
    Args:
        max_index: Índice máximo a ser testado.
        prefer_working: Se True, prioriza câmeras que conseguem fornecer frames.
    
    Returns:
        Índice da primeira câmera disponível encontrada, ou None se nenhuma for encontrada.
    """
    cv2.setLogLevel(0)
    
    devices = list_video_devices(max_index)
    
    if not devices:
        return None
    
    if prefer_working:
        working_devices = [d for d in devices if d['working']]
        if working_devices:
            print(f"Encontradas {len(working_devices)} câmera(s) funcionando de {len(devices)} dispositivo(s) de vídeo")
            return working_devices[0]['index']
    
    print(f"Encontrados {len(devices)} dispositivo(s) de vídeo, testando...")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        for device in devices:
            index = device['index']
            try:
                cap = None
                backends = [
                    cv2.CAP_V4L2,
                    cv2.CAP_ANY,
                ]
                
                for backend in backends:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        break
                    cap.release()
                    cap = None
                
                if cap is None or not cap.isOpened():
                    continue
                
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except:
                    pass
                
                success_count = 0
                valid_frames = []
                
                for attempt in range(15):
                    try:
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                                success_count += 1
                                valid_frames.append(frame.shape)
                    except:
                        pass
                    
                    time.sleep(0.03)
                
                cap.release()
                
                if success_count >= 1:
                    print(f"✓ Câmera encontrada no índice {index} ({success_count}/15 frames lidos)")
                    return index
                else:
                    print(f"  Testando índice {index}... sem frames válidos")
                    
            except Exception:
                if cap:
                    try:
                        cap.release()
                    except:
                        pass
                continue
    
    return None


def build_video_parser(description: str = "Ferramenta de benchmark para modelos YOLO ou ONNX em vídeo ou câmera.") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model-format",
        choices=("yolo", "onnx"),
        default="yolo",
        help="Formato do modelo a ser avaliado.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=False,
        default=None,
        help="Caminho para o arquivo do modelo (não necessário com --list-cameras).",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Caminho para o arquivo de vídeo. Se não fornecido, procura automaticamente por uma câmera disponível.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Lista todas as câmeras disponíveis e sai.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Dispositivo utilizado na inferência (somente para modelos YOLO).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de frames a serem processados no benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Quantidade de inferências iniciais para aquecimento (sem registro de métricas).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desativa a barra de progresso.",
    )
    parser.add_argument(
        "--resource-metrics-dir",
        type=Path,
        default=Path("metrics"),
        help="Diretório para exportar métricas de recursos (CPU/memória/latência).",
    )
    parser.add_argument(
        "--export-resources",
        action="store_true",
        help="Exporta métricas de recursos (CPU/memória/latência) em CSV.",
    )
    parser.add_argument(
        "--export-predictions",
        action="store_true",
        help="Exporta métricas detalhadas das predições em CSV.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("metrics"),
        help="Diretório para exportar as métricas de predições em CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Arquivo para salvar os resultados em formato JSON.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Exibe o resultado completo em JSON no stdout.",
    )
    parser.add_argument(
        "--include-detailed",
        action="store_true",
        help="Inclui resultados por frame no JSON.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Salva o vídeo processado com predições na pasta 'videos/processed'.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Diretório para salvar vídeos processados (padrão: videos/processed na raiz do projeto).",
    )
    return parser


def validate_video_paths(video_path: Optional[Path], model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if video_path is not None and not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")


def process_frame(frame: np.ndarray) -> np.ndarray:
    """Processa um frame para o formato esperado pelo modelo."""
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.expand_dims(frame.transpose(2, 0, 1), 0).astype(np.float32) / 255.0
    return frame


def process_onnx_output(
    output: np.ndarray, 
    conf_threshold: float = 0.25, 
    iou_threshold: float = 0.45,
    original_width: int = 640,
    original_height: int = 640,
    model_input_size: int = 640
) -> list:
    """Processa saída ONNX do YOLO e retorna detecções no formato esperado.
    
    Args:
        output: Saída do modelo ONNX (pode ter diferentes formatos)
        conf_threshold: Limiar de confiança mínimo
        iou_threshold: Limiar de IoU para NMS
        original_width: Largura original do frame
        original_height: Altura original do frame
        model_input_size: Tamanho do input do modelo (geralmente 640)
    
    Returns:
        Lista de objetos ONNXResult com boxes, confidences e classes
    """
    if output is None:
        return []
    
    if not isinstance(output, np.ndarray):
        output = np.array(output)
    
    if len(output.shape) == 3:
        if output.shape[0] == 1:
            output = output[0]
        else:
            return []
    
    if len(output.shape) == 2:
        num_first, num_second = output.shape
        
        if num_first <= 10 and num_second > 100:
            output = output.T
            num_detections, num_features = output.shape
        elif num_first > 100 and num_second <= 10:
            num_detections, num_features = output.shape
        else:
            num_detections, num_features = output.shape
    elif len(output.shape) == 4:
        output = output[0]
        if len(output.shape) != 2:
            return []
        num_detections, num_features = output.shape
    else:
        return []
    
    scale_x = original_width / model_input_size
    scale_y = original_height / model_input_size
    
    boxes = []
    confidences = []
    classes = []
    for detection in output:
        if num_features < 5:
            continue
        
        coord1, coord2, coord3, coord4 = detection[:4]
        
        if coord3 > coord1 and coord4 > coord2:
            if coord3 < 1.0 and coord4 < 1.0:
                x1 = coord1 * model_input_size
                y1 = coord2 * model_input_size
                x2 = coord3 * model_input_size
                y2 = coord4 * model_input_size
            elif coord3 < model_input_size and coord4 < model_input_size:
                x1, y1, x2, y2 = float(coord1), float(coord2), float(coord3), float(coord4)
            else:
                x1, y1, x2, y2 = float(coord1), float(coord2), float(coord3), float(coord4)
        else:
            x_center, y_center, width, height = float(coord1), float(coord2), float(coord3), float(coord4)
            
            if width < 1.0 and height < 1.0:
                x_center = x_center * model_input_size
                y_center = y_center * model_input_size
                width = width * model_input_size
                height = height * model_input_size
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
        
        if num_features >= 6:
            confidence_val = detection[4]
            
            if num_features > 6:
                class_scores = detection[5:]
                class_id = int(np.argmax(class_scores))
                max_class_score = class_scores[class_id]
                confidence = float(confidence_val * max_class_score)
            else:
                confidence = float(confidence_val)
                class_id = int(detection[5])
        else:
            confidence = float(detection[4])
            class_id = 0
        
        if confidence < conf_threshold:
            continue
        
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y
        
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        confidences.append(confidence)
        classes.append(class_id)
    
    if not boxes:
        return []
    
    boxes_np = np.array(boxes)
    confidences_np = np.array(confidences)
    
    try:
        indices = cv2.dnn.NMSBoxes(
            boxes_np.tolist(),
            confidences_np.tolist(),
            conf_threshold,
            iou_threshold
        )
        
        if indices is None or len(indices) == 0:
            return []
        
        indices = indices.flatten() if hasattr(indices, 'flatten') else indices
    except Exception:
        indices = np.arange(len(boxes))
    
    class ONNXResult:
        def __init__(self, boxes, confidences, classes):
            self.boxes = type('Boxes', (), {
                'xyxy': np.array(boxes),
                'conf': np.array(confidences),
                'cls': np.array(classes)
            })()
    
    filtered_boxes = boxes_np[indices]
    filtered_confidences = confidences_np[indices]
    filtered_classes = np.array(classes)[indices]
    
    return [ONNXResult(filtered_boxes, filtered_confidences, filtered_classes)]


def create_onnx_runner_video(model_path: Path) -> Callable[[np.ndarray], tuple[np.ndarray, Optional[list]]]:
    """Cria uma função de inferência para modelos ONNX em vídeo."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    model_input_size = 640

    def _run(frame: np.ndarray) -> tuple[np.ndarray, Optional[list]]:
        original_height, original_width = frame.shape[:2]
        
        frame_tensor = process_frame(frame)
        outputs = session.run([output_name], {input_name: frame_tensor})
        
        processed_results = process_onnx_output(
            outputs[0],
            original_width=original_width,
            original_height=original_height,
            model_input_size=model_input_size,
        )
        
        return frame, processed_results if processed_results else None

    return _run


def create_yolo_runner_video(model_path: Path, device: str) -> tuple[Callable[[np.ndarray], tuple[np.ndarray, Optional[list]]], dict]:
    """Cria uma função de inferência para modelos YOLO em vídeo.
    
    Returns:
        Tupla com (função de inferência, dicionário de nomes de classes)
    """
    yolo_model = YOLO(str(model_path))
    class_names = {}
    if hasattr(yolo_model, 'names'):
        names = yolo_model.names
        if isinstance(names, dict):
            class_names = names
        elif isinstance(names, (list, tuple)):
            class_names = {i: str(name) for i, name in enumerate(names)}

    def _run(frame: np.ndarray) -> tuple[np.ndarray, Optional[list]]:
        results = yolo_model.predict(source=frame.copy(), device=device, verbose=False)
        return frame, results

    return _run, class_names


def warmup_model_video(
    run_inference: Callable[[np.ndarray], tuple[np.ndarray, Optional[list]]],
    cap: cv2.VideoCapture,
    warmup_runs: int,
    is_video: bool,
) -> None:
    """Executa warmup do modelo com frames iniciais."""
    if warmup_runs <= 0:
        return

    frame_count = 0
    while frame_count < warmup_runs:
        ret, frame = cap.read()
        if not ret:
            if is_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        run_inference(frame)[0]
        frame_count += 1


def draw_predictions(
    frame: np.ndarray, 
    results: Optional[list], 
    class_names: Optional[dict] = None, 
    fps: Optional[float] = None,
    model_name: Optional[str] = None
) -> np.ndarray:
    """Desenha predições YOLO no frame.
    
    Args:
        frame: Frame de vídeo em BGR
        results: Resultados da predição YOLO
        class_names: Dicionário mapeando IDs de classe para nomes (opcional)
        fps: FPS atual para exibir no frame (opcional)
        model_name: Nome do modelo para exibir no frame (opcional)
    """
    frame_bgr = frame.copy()
    height, width = frame_bgr.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        color = (0, 255, 0)
        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        
        overlay = frame_bgr.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + text_width + 10, 10 + text_height + baseline + 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)
        
        cv2.putText(
            frame_bgr,
            fps_text,
            (15, 15 + text_height),
            font,
            font_scale,
            color,
            thickness
        )
    
    if model_name is not None:
        model_text = model_name
        color = (255, 255, 0)
        (text_width, text_height), baseline = cv2.getTextSize(model_text, font, font_scale, thickness)
        
        x_pos = width - text_width - 20
        y_pos = 15 + text_height
        
        overlay = frame_bgr.copy()
        cv2.rectangle(
            overlay,
            (x_pos - 10, 10),
            (width - 10, 10 + text_height + baseline + 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)
        
        cv2.putText(
            frame_bgr,
            model_text,
            (x_pos, y_pos),
            font,
            font_scale,
            color,
            thickness
        )
    
    if results is None or len(results) == 0:
        return frame_bgr
    
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            if hasattr(result.boxes.xyxy, 'cpu'):
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
            else:
                boxes = np.array(result.boxes.xyxy)
                confidences = np.array(result.boxes.conf)
                classes = np.array(result.boxes.cls).astype(int)
            
            if boxes.max() <= 1.0:
                boxes = boxes * np.array([width, height, width, height])
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if class_names and isinstance(class_names, dict) and cls in class_names:
                    class_label = str(class_names[cls])
                elif class_names and isinstance(class_names, (list, tuple)) and cls < len(class_names):
                    class_label = str(class_names[cls])
                else:
                    class_label = f"Class {cls}"
                
                label = f"{class_label} {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(frame_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return frame_bgr


def collect_video_measurements(
    run_inference: Callable[[np.ndarray], tuple[np.ndarray, Optional[list]]],
    cap: cv2.VideoCapture,
    show_progress: bool,
    process: Process,
    limit: Optional[int] = None,
    save_video: bool = False,
    class_names: Optional[dict] = None,
    model_name: Optional[str] = None,
    collect_predictions: bool = False,
) -> tuple[list[VideoMeasurement], float, list[np.ndarray], list[dict]]:
    """Coleta métricas de performance durante o processamento de frames.
    
    Returns:
        Tuple com (lista de medições, tempo total em segundos, lista de frames processados, lista de predições)
    """
    process.cpu_percent(interval=None)

    measurements: list[VideoMeasurement] = []
    processed_frames: list[np.ndarray] = []
    predictions: list[dict] = []
    frame_number = 0
    start_time = perf_counter()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None
    elif limit is not None:
        total_frames = min(limit, total_frames)

    if limit is not None:
        iterator = range(limit)
    else:
        iterator = range(total_frames) if total_frames is not None else iter(int, 1)

    if show_progress:
        if total_frames is not None:
            iterator = tqdm(iterator, desc="Benchmarking", unit="frame", total=total_frames)
        else:
            iterator = tqdm(iterator, desc="Benchmarking", unit="frame")

    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        start = perf_counter()
        processed_frame, results = run_inference(frame)
        latency = (perf_counter() - start) * 1000

        cpu_percent = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024**2)

        current_fps = 1000.0 / latency if latency > 0 else 0.0

        measurements.append(
            VideoMeasurement(
                frame_number=frame_number,
                latency_ms=latency,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
            )
        )

        if collect_predictions and results:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    if hasattr(result.boxes.xyxy, 'cpu'):
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                    else:
                        boxes = np.array(result.boxes.xyxy)
                        confidences = np.array(result.boxes.conf)
                        classes = np.array(result.boxes.cls).astype(int)
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        predictions.append({
                            "frame_number": frame_number,
                            "class_id": int(cls),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "confidence": float(conf),
                        })

        if save_video:
            frame_with_predictions = draw_predictions(
                frame.copy(), 
                results, 
                class_names, 
                fps=current_fps,
                model_name=model_name
            )
            processed_frames.append(frame_with_predictions)

        frame_number += 1
        if limit is not None and frame_number >= limit:
            break

    total_time = perf_counter() - start_time
    return measurements, total_time, processed_frames, predictions


def build_video_summary(
    measurements: list[VideoMeasurement],
    total_time_seconds: float,
) -> dict:
    """Constrói resumo com métricas de vídeo incluindo FPS e MPS."""
    latencies = [item.latency_ms for item in measurements]
    cpu_values = [item.cpu_percent for item in measurements]
    mem_values = [item.memory_mb for item in measurements]

    fps = len(measurements) / total_time_seconds if total_time_seconds > 0 else 0.0

    avg_latency_ms = mean(latencies) if latencies else 0.0
    mps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    summary = {
        "latency_ms": summarize(latencies),
        "cpu_percent": summarize(cpu_values),
        "memory_mb": summarize(mem_values),
        "fps": fps,
        "mps": mps,
    }

    return summary


def format_video_summary(report: BenchmarkReport) -> str:
    """Formata o resumo do benchmark de vídeo para exibição."""
    lines = [
        f"Modelo: {report.model_name}",
        f"Frames avaliados: {report.num_images}",
        f"Warmup: {report.warmup_runs}",
        "",
        "Latência (ms): "
        f"mín={report.summary['latency_ms']['min']:.2f}, "
        f"médio={report.summary['latency_ms']['avg']:.2f}, "
        f"máx={report.summary['latency_ms']['max']:.2f}",
        "CPU (%): "
        f"mín={report.summary['cpu_percent']['min']:.2f}, "
        f"médio={report.summary['cpu_percent']['avg']:.2f}, "
        f"máx={report.summary['cpu_percent']['max']:.2f}",
        "Memória (MB): "
        f"mín={report.summary['memory_mb']['min']:.2f}, "
        f"médio={report.summary['memory_mb']['avg']:.2f}, "
        f"máx={report.summary['memory_mb']['max']:.2f}",
    ]
    
    if "fps" in report.summary:
        lines.append(
            f"FPS: {report.summary['fps']:.2f}"
        )
    if "mps" in report.summary:
        lines.append(
            f"MPS: {report.summary['mps']:.2f}"
        )
    
    return "\n".join(lines)


def convert_video_to_measurements(video_measurements: list[VideoMeasurement]) -> list[Measurement]:
    """Converte VideoMeasurement para Measurement (formato do shared.py)."""
    return [
        Measurement(
            image_name=f"frame_{item.frame_number:06d}",
            latency_ms=item.latency_ms,
            cpu_percent=item.cpu_percent,
            memory_mb=item.memory_mb,
        )
        for item in video_measurements
    ]


def save_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: float = 30.0,
) -> None:
    """Salva frames processados como vídeo usando codec mp4v."""
    if not frames:
        raise ValueError("Nenhum frame para salvar.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_frames = []
    for frame in frames:
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3:
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        processed_frames.append(frame)
    
    height, width = processed_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(
            f"Não foi possível criar o VideoWriter com codec mp4v. "
            f"Verifique se o codec está disponível no sistema."
        )
    
    for frame in processed_frames:
        out.write(frame)
    
    out.release()
    
    if not output_path.exists():
        raise RuntimeError(f"Vídeo não foi salvo: {output_path}")
    
    file_size = output_path.stat().st_size
    if file_size == 0:
        raise RuntimeError(f"Vídeo foi criado mas está vazio: {output_path}")
    
    print(
        f"Vídeo salvo: {output_path} "
        f"(codec: mp4v, {len(processed_frames)} frames, "
        f"{file_size / (1024*1024):.2f} MB)"
    )


def export_video_resource_metrics(
    measurements: list[VideoMeasurement],
    metrics_root: Path,
    total_time: float,
) -> None:
    """Exporta métricas de recursos para CSV."""
    metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_root / "video_resources.csv"

    avg_latency_ms = mean([item.latency_ms for item in measurements]) if measurements else 0.0
    mps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    fps = len(measurements) / total_time if total_time > 0 else 0.0

    df = pd.DataFrame(
        [
            {
                "frame_number": item.frame_number,
                "latency_ms": item.latency_ms,
                "cpu_percent": item.cpu_percent,
                "memory_mb": item.memory_mb,
                "fps": fps,
                "mps": mps,
            }
            for item in measurements
        ],
        columns=["frame_number", "latency_ms", "cpu_percent", "memory_mb", "fps", "mps"],
    )
    df.to_csv(csv_path, index=False)


def export_video_prediction_metrics(
    predictions: list[dict],
    metrics_root: Path,
) -> None:
    """Exporta métricas de predições de vídeo para CSV."""
    if not predictions:
        return
    
    metrics_root.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_root / "video_predictions.csv"

    df = pd.DataFrame(
        predictions,
        columns=["frame_number", "class_id", "x1", "y1", "x2", "y2", "confidence"],
    )
    df.to_csv(csv_path, index=False)


def execute_video_benchmark(args: argparse.Namespace) -> BenchmarkReport:
    """Executa o benchmark em vídeo ou câmera."""
    if args.model_path is None:
        raise ValueError("--model-path é obrigatório para executar o benchmark.")
    
    resource_dir = args.resource_metrics_dir.resolve()
    validate_video_paths(args.video_path, args.model_path)

    if args.videos_dir is not None:
        videos_dir = args.videos_dir.resolve()
    else:
        project_root = Path(__file__).parent.parent.parent
        videos_dir = project_root / "videos" / "processed"
    
    is_video = args.video_path is not None
    if is_video:
        cap = cv2.VideoCapture(str(args.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {args.video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        print("Procurando por câmeras disponíveis...")
        camera_index = find_available_camera()
        if camera_index is None:
            error_msg = (
                "Nenhuma câmera disponível encontrada após testar índices 0-49.\n\n"
                "Possíveis soluções:\n"
                "  1. Liste todas as câmeras disponíveis:\n"
                "     $ python src/benchmark/model_benchmark_on_video.py --list-cameras\n"
                "  2. Verifique se a câmera está conectada:\n"
                "     $ lsusb | grep -i camera\n"
                "  3. Verifique se há processos usando a câmera:\n"
                "     $ lsof | grep /dev/video*\n"
                "     $ fuser /dev/video*\n"
                "  4. Feche outros aplicativos que possam estar usando a câmera\n"
                "  5. Tente desligar e religar a câmera USB\n"
                "  6. Verifique permissões:\n"
                "     $ ls -l /dev/video*\n"
                "  7. Se a câmera aparecer como ocupada, tente:\n"
                "     $ sudo modprobe -r uvcvideo && sudo modprobe uvcvideo"
            )
            raise RuntimeError(error_msg)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera no índice {camera_index}.")
        video_fps = 30.0
        print(f"✓ Câmera encontrada e conectada no índice {camera_index}")
        
        if args.limit is None:
            args.limit = 300
            print(f"Limite padrão de 300 frames aplicado para webcam (use --limit para alterar)")

    try:
        class_names = {}
        if args.model_format == "yolo":
            run_inference, class_names = create_yolo_runner_video(args.model_path, args.device)
            model_name = f"YOLO::{args.model_path.name}"
        else:
            run_inference = create_onnx_runner_video(args.model_path)
            model_name = f"ONNX::{args.model_path.name}"

        warmup_model_video(run_inference, cap, args.warmup, is_video)
        if args.warmup > 0 and is_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        process = Process(os.getpid())
        process.cpu_percent(interval=None)

        predictions_dir = args.predictions_dir.resolve()

        video_measurements, total_time, processed_frames, predictions = collect_video_measurements(
            run_inference=run_inference,
            cap=cap,
            show_progress=not args.no_progress,
            process=process,
            limit=args.limit,
            save_video=args.save_video,
            class_names=class_names,
            model_name=model_name,
            collect_predictions=args.export_predictions,
        )

        if not video_measurements:
            raise RuntimeError("Nenhum frame foi processado.")

        if args.save_video and processed_frames:
            model_stem = args.model_path.stem
            if is_video:
                video_stem = args.video_path.stem
                output_name = f"{model_stem}_{video_stem}_processed.mp4"
            else:
                output_name = f"{model_stem}_camera_processed.mp4"
            
            output_path = videos_dir / output_name
            save_video(processed_frames, output_path, fps=video_fps)
            print(f"Vídeo salvo em: {output_path}")

        measurements = convert_video_to_measurements(video_measurements)

        summary = build_video_summary(video_measurements, total_time)

        if args.export_resources:
            export_video_resource_metrics(
                measurements=video_measurements,
                metrics_root=resource_dir,
                total_time=total_time,
            )
        
        if args.export_predictions:
            export_video_prediction_metrics(
                predictions=predictions,
                metrics_root=predictions_dir,
            )

        return BenchmarkReport(
            model_name=model_name,
            num_images=len(measurements),
            warmup_runs=args.warmup,
            summary=summary,
            detailed=measurements,
        )

    finally:
        cap.release()


def run_video_cli(description: str = "Ferramenta de benchmark para modelos YOLO ou ONNX em vídeo ou câmera.") -> None:
    """Função principal que executa o benchmark de vídeo via CLI."""
    parser = build_video_parser(description)
    args = parser.parse_args()
    
    if args.list_cameras:
        print("Procurando por câmeras disponíveis...\n")
        devices = list_video_devices(max_index=50)
        
        if not devices:
            print("Nenhum dispositivo de vídeo encontrado.")
            return
        
        print(f"Encontrados {len(devices)} dispositivo(s) de vídeo:\n")
        print(f"{'Índice':<8} {'Resolução':<15} {'FPS':<8} {'Status':<12}")
        print("-" * 45)
        
        working_count = 0
        for device in devices:
            status = "✓ Funcionando" if device['working'] else "✗ Não funciona"
            if device['working']:
                working_count += 1
            resolution = f"{device['width']}x{device['height']}" if device['width'] > 0 else "Desconhecida"
            fps_str = f"{device['fps']:.1f}" if device['fps'] > 0 else "N/A"
            print(f"{device['index']:<8} {resolution:<15} {fps_str:<8} {status:<12}")
        
        print(f"\nTotal: {len(devices)} dispositivo(s), {working_count} funcionando")
        if working_count > 0:
            working_devices = [d for d in devices if d['working']]
            print(f"\nCâmera recomendada: índice {working_devices[0]['index']}")
        return
    
    report = execute_video_benchmark(args)
    print(format_video_summary(report))
    export_report(
        report=report,
        print_json=args.print_json,
        output_json=args.output_json,
        include_detailed=args.include_detailed,
    )
