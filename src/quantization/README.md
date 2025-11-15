# Benchmark e Analysis CLI

Este repositório contém os scripts, notebooks e utilidades usados para avaliar modelos YOLO (FP32) e variações quantizadas (INT8 dinâmico e estático) exportadas para ONNX.

## Pré‑requisitos

- Python 3.12+
- [Poetry](https://python-poetry.org/) **ou** `pip` para gerenciamento de dependências
- CUDA opcional (se desejar executar benchmarks YOLO em GPU)

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/Inteli-College/2025-1A-T01-G05-INTERNO.git interno
cd interno/src/quantization

# Criar ambiente virtual (opcional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt  # ou poetry install
```

## Estrutura relevante

- `src/notebooks/` — notebooks de treinamento e referência.
- `src/benchmark/` — scripts CLI para automatizar benchmarks.
  - `original_fp32_benchmark.py` — benchmark de imagens com modelo YOLO FP32
  - `dynamic_int8_quant_benchmark.py` — benchmark de imagens com modelo ONNX INT8 dinâmico
  - `static_int8_quant_benchmark.py` — benchmark de imagens com modelo ONNX INT8 estático
  - `model_on_video_benchmark.py` — benchmark de vídeo ou câmera
  - `shared.py` — utilidades compartilhadas entre os CLIs.
- `notebooks/models/` — modelos (`best_fp32.pt`, `dynamic_int8_quant.onnx`, `static_int8_quant.onnx`).
- `notebooks/tcc-1/valid/images` — imagens usadas para os benchmarks.
- `videos/original/` — vídeos de entrada para benchmarks.
- `videos/processed/` — vídeos processados com predições desenhadas.

## Executando Benchmarks

Todos os CLIs compartilham a mesma interface. Execute a partir da raiz do projeto:

```bash
# YOLO FP32
python benchmark/original_fp32_benchmark.py

# ONNX INT8 dinâmico
python benchmark/dynamic_int8_quant_benchmark.py

# ONNX INT8 estático
python benchmark/static_int8_quant_benchmark.py
```

### Opções comuns

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `--model-format {yolo,onnx}` | Formato do modelo | Definido por script |
| `--model-path PATH` | Caminho para o modelo | Valor padrão do script |
| `--image-dir PATH` | Diretório com imagens | `notebooks/tcc-1/valid/images` |
| `--device DEVICE` | Dispositivo (`cpu`, `0`, `0,1`, etc.) | `cpu` |
| `--limit N` | Limite de imagens | Todas |
| `--warmup N` | Execuções iniciais ignoradas | `0` |
| `--extensions` | Extensões aceitas | `.jpg .jpeg .png` |
| `--no-progress` | Remove barras de progresso | Desabilitado |
| `--output-json PATH` | Salva resumo em JSON | Não salva |
| `--print-json` | Exibe JSON completo | Não |
| `--include-detailed` | Adiciona resultados por imagem ao JSON | Não |

### Métricas opcionais

Os scripts podem exportar métricas adicionais em CSV:

| Flag | Resultado |
|------|-----------|
| `--export-predictions` | Gera `predictions.csv` com bounding boxes (colunas `image`, `class_id`, `x1`, `y1`, `x2`, `y2`, `confidence`). |
| `--predictions-dir DIR` | Define onde salvar `predictions.csv` (default `metrics`). |
| `--export-resources` | Gera `resources.csv` com latência, CPU e memória por imagem. |
| `--resource-metrics-dir DIR` | Define onde salvar `resources.csv` (default `metrics`). |

Exemplos:

```bash
# Exporta apenas as predições para metrics/dynamic/
python benchmark/dynamic_int8_quant_benchmark.py \
  --export-predictions \
  --predictions-dir metrics/dynamic

# Exporta apenas métricas de recursos
python benchmark/static_int8_quant_benchmark.py \
  --export-resources \
  --resource-metrics-dir metrics/static

# Exporta ambos em diretórios distintos
python benchmark/original_fp32_benchmark.py \
  --export-predictions --predictions-dir metrics/fp32/preds \
  --export-resources --resource-metrics-dir metrics/fp32/resources
```

Os diretórios especificados são criados automaticamente, e o arquivo CSV é sempre salvo diretamente neles.

## Benchmarks de Vídeo

O script `model_on_video_benchmark.py` permite executar benchmarks em arquivos de vídeo ou câmeras ao vivo, processando frames individualmente e coletando métricas de performance.

### Comando básico

```bash
# Executar benchmark em vídeo ou câmera
python src/benchmark/model_on_video_benchmark.py \
  --model-path notebooks/models/best_fp32.pt \
  --model-format yolo
```

### Opções disponíveis

| Flag | Descrição | Padrão |
|------|-----------|--------|
| `--model-format {yolo,onnx}` | Formato do modelo | `yolo` |
| `--model-path PATH` | Caminho para o modelo | Obrigatório |
| `--video-path PATH` | Caminho para arquivo de vídeo | Se não fornecido, usa câmera |
| `--list-cameras` | Lista câmeras disponíveis e sai | Desabilitado |
| `--device DEVICE` | Dispositivo para inferência (YOLO apenas) | `cpu` |
| `--limit N` | Limite de frames a processar | Todos |
| `--warmup N` | Execuções iniciais ignoradas | `0` |
| `--no-progress` | Remove barras de progresso | Desabilitado |
| `--save-video` | Salva vídeo processado com predições | Desabilitado |
| `--videos-dir PATH` | Diretório para salvar vídeos processados | `videos/processed` |
| `--export-resources` | Exporta métricas de recursos em CSV | Desabilitado |
| `--resource-metrics-dir DIR` | Diretório para salvar `video_resources.csv` | `metrics` |
| `--export-predictions` | Exporta predições detalhadas em CSV | Desabilitado |
| `--predictions-dir DIR` | Diretório para salvar `video_predictions.csv` | `metrics` |
| `--output-json PATH` | Salva resumo em JSON | Não salva |
| `--print-json` | Exibe JSON completo | Não |
| `--include-detailed` | Inclui resultados por frame no JSON | Não |

### Exemplos de uso

#### Listar câmeras disponíveis

```bash
python src/benchmark/model_on_video_benchmark.py --list-cameras
```

#### Processar arquivo de vídeo com modelo YOLO

```bash
python benchmark/model_on_video_benchmark.py \
  --model-path notebooks/models/best_fp32.pt \
  --model-format yolo \
  --video-path videos/original/test_file.mp4 \
  --device cpu
```

#### Processar arquivo de vídeo com modelo ONNX quantizado

```bash
python benchmark/model_on_video_benchmark.py \
  --model-path notebooks/models/dynamic_int8_quant.onnx \
  --model-format onnx \
  --video-path videos/original/test_file.mp4
```

#### Processar vídeo e salvar resultado com predições desenhadas

```bash
python benchmark/model_on_video_benchmark.py \
  --model-path notebooks/models/best_fp32.pt \
  --model-format yolo \
  --video-path videos/original/test_file.mp4 \
  --save-video \
  --videos-dir videos/processed
```

#### Processar câmera ao vivo (sem arquivo de vídeo)

```bash
python benchmark/model_on_video_benchmark.py \
  --model-path notebooks/models/best_fp32.pt \
  --model-format yolo \
  --device cpu \
  --limit 100
```

#### Exportar métricas de recursos e predições

```bash
python benchmark/model_on_video_benchmark.py \
  --model-path notebooks/models/static_int8_quant.onnx \
  --model-format onnx \
  --video-path videos/original/test_file.mp4 \
  --export-resources \
  --resource-metrics-dir metrics/static \
  --export-predictions \
  --predictions-dir metrics/static/preds
```

#### Benchmark completo com todas as métricas

```bash
python benchmark/model_on_video_benchmark.py \
  --model-path notebooks/models/best_fp32.pt \
  --model-format yolo \
  --video-path videos/original/test_file.mp4 \
  --save-video \
  --export-resources \
  --export-predictions \
  --output-json metrics/original/video_report.json \
  --include-detailed \
  --warmup 5
```

### Métricas exportadas

- **`video_resources.csv`**: Contém métricas por frame (quando `--export-resources` é usado):
  - `frame_number`: Número do frame
  - `latency_ms`: Latência de inferência em milissegundos
  - `cpu_percent`: Uso de CPU em percentual
  - `memory_mb`: Uso de memória em MB
  - `fps`: FPS médio calculado
  - `mps`: Frames por segundo (MPS) baseado na latência

- **`video_predictions.csv`**: Contém predições detalhadas (quando `--export-predictions` é usado):
  - `frame_number`: Número do frame
  - `class_id`: ID da classe detectada
  - `x1`, `y1`, `x2`, `y2`: Coordenadas do bounding box
  - `confidence`: Confiança da predição

Os vídeos processados são salvos em `videos/processed/` (ou no diretório especificado via `--videos-dir`) com o nome formatado como `{modelo}_{nome_original}_processed.mp4`.

## Análises e gráficos

Os scripts em `analysis/` consomem os CSVs gerados pelos benchmarks e produzem gráficos comparativos ou específicos por variante. Eles funcionam a partir de qualquer diretório do projeto graças à resolução automática de caminhos.

### Gráficos por variante

Substitua `original` por `dynamic` ou `static` conforme necessário:

```bash
# Recursos (latência, CPU, memória)
python analysis/resources_analysis.py --variant original

# Predições (distribuição de confiança e volume por imagem)
python analysis/predictions_analysis.py --variant original
```

Parâmetros úteis:

| Flag | Descrição |
|------|-----------|
| `--metrics-root PATH` | Raiz onde estão as pastas `original/`, `dynamic/`, `static/` (default: `<projeto>/metrics`). |
| `--metrics-dir PATH` | Aponta diretamente para uma pasta que contenha `resources.csv` ou `predictions.csv`, ignorando `--variant`. |
| `--output-dir PATH` | Define onde salvar as imagens (default: `metrics/<variante>/images`). |

### Comparação entre variantes

```bash
python analysis/comparison_analysis.py
```

Esse comando gera séries temporais e uma CDF de confiança para as três variantes, salvando os gráficos em `images/comparison/` (ou no diretório informado via `--output-dir`). Para usar métricas em outro local, utilize `--metrics-root PATH`.
