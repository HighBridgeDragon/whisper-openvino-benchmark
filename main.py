import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import List, Tuple

import cpuinfo
import librosa
import numpy as np
import psutil
import requests
from openvino_genai import WhisperPipeline
from tabulate import tabulate


@dataclass
class ChunkMetrics:
    """Metrics for a single audio chunk"""

    chunk_index: int
    start_time: float
    end_time: float
    processing_time: float
    audio_duration: float
    rtf: float
    transcription: str
    cumulative_time: float
    cumulative_rtf: float


@dataclass
class StreamingBenchmarkResults:
    """Results from streaming benchmark"""

    chunk_metrics: List[ChunkMetrics]
    total_audio_duration: float
    total_processing_time: float
    overall_rtf: float
    first_chunk_latency: float
    avg_chunk_processing_time: float
    min_chunk_processing_time: float
    max_chunk_processing_time: float
    std_chunk_processing_time: float
    throughput_audio_per_sec: float
    inter_chunk_latencies: List[float]
    avg_inter_chunk_latency: float
    model_memory_gb: float
    avg_inference_memory_mb: float
    full_transcription: str


def get_cpu_info():
    """Get CPU information using py-cpuinfo"""
    try:
        cpu_info_data = cpuinfo.get_cpu_info()

        cpu_name = cpu_info_data.get("brand_raw", "Unknown")
        cores = psutil.cpu_count(logical=False) or 1
        threads = psutil.cpu_count(logical=True) or 1

        # Get CPU flags
        detected_flags = set(cpu_info_data.get("flags", []))

        # Map flags to feature names
        features = []

        flag_mapping = {
            # Vector instructions (OpenVINO performance critical)
            "avx": "AVX",
            "avx2": "AVX2",
            "fma": "FMA",
            # AVX512 family (high performance inference)
            "avx512f": "AVX512F",
            "avx512dq": "AVX512DQ",
            "avx512cd": "AVX512CD",
            "avx512bw": "AVX512BW",
            "avx512vl": "AVX512VL",
            "avx512vnni": "AVX512VNNI",
            "avx512_bf16": "AVX512-BF16",
            # AI-optimized instructions (neural network acceleration)
            "vnni": "VNNI",
            "amx_tile": "AMX-TILE",
            "amx_int8": "AMX-INT8",
            "amx_bf16": "AMX-BF16",
        }

        for flag, feature_name in flag_mapping.items():
            if flag in detected_flags:
                features.append(feature_name)

        # Windows WMI fallback for CPU name if needed
        if cpu_name == "Unknown" and platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name", "/value"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Name="):
                        cpu_name = line.split("=", 1)[1].strip()
                        break
            except Exception:
                pass

        return {
            "name": cpu_name,
            "cores": cores,
            "threads": threads,
            "features": features,
            "detection_methods": "py-cpuinfo",
        }

    except Exception as e:
        print(f"Warning: Could not get CPU info: {e}")
        return {
            "name": "Unknown",
            "cores": psutil.cpu_count(logical=False) or 1,
            "threads": psutil.cpu_count(logical=True) or 1,
            "features": [],
            "detection_methods": "Failed",
        }


def get_audio_file(audio_file_path=None, cache_dir="cache"):
    """Get audio file from specified path or download default if needed"""
    # If audio file path is specified, try to use it
    if audio_file_path:
        if os.path.exists(audio_file_path):
            print(f"Using specified audio file: {audio_file_path}")
            return audio_file_path
        else:
            print(f"Warning: Specified audio file not found: {audio_file_path}")
            print("Falling back to default audio file...")

    # Download default audio file
    return download_audio_file(
        "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav",
        cache_dir,
    )


def download_audio_file(url, cache_dir="cache"):
    """Download audio file with caching"""
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(cache_dir, filename)

    if os.path.exists(filepath):
        print(f"Using cached audio file: {filepath}")
        return filepath

    print(f"Downloading audio file from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)

        print("\nDownload completed!")
        return filepath
    except Exception as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise


def validate_model_files(model_path):
    """Validate that all required model files exist"""
    required_files = [
        "openvino_encoder_model.xml",
        "openvino_encoder_model.bin",
        "openvino_decoder_model.xml",
        "openvino_decoder_model.bin",
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_detokenizer.bin",
        "config.json",
        "generation_config.json",
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"Warning: Missing model files: {', '.join(missing_files)}")
        print("This may indicate an incomplete model export.")
        print("Try re-exporting the model with --disable-stateful flag:")
        print(
            f"uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo --trust-remote-code --weight-format fp32 --disable-stateful {model_path}"
        )
        return False

    print("OK All required model files found")
    return True


def split_audio_into_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_size_sec: float = 30.0,
    overlap_sec: float = 0.0,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Split audio into chunks with optional overlap.

    Returns:
        List of tuples (chunk_audio, start_time, end_time)
    """
    chunk_size_samples = int(chunk_size_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    stride_samples = chunk_size_samples - overlap_samples

    chunks = []
    total_samples = len(audio)

    for start_idx in range(0, total_samples, stride_samples):
        end_idx = min(start_idx + chunk_size_samples, total_samples)
        chunk = audio[start_idx:end_idx]

        start_time = start_idx / sample_rate
        end_time = end_idx / sample_rate

        chunks.append((chunk, start_time, end_time))

        if end_idx >= total_samples:
            break

    return chunks


def save_yaml_results(results, output_file):
    """Save benchmark results to YAML file with descriptive comments"""
    from datetime import datetime

    # Format features list for YAML
    features = results["system_info"]["cpu"]["features"]
    if features:
        features_yaml = "\n" + "\n".join(f"      - {feature}" for feature in features)
    else:
        features_yaml = " []"

    # Format detailed times list for YAML
    times = results["times"]
    if times:
        times_yaml = "\n" + "\n".join(f"  - {time:.3f}" for time in times)
    else:
        times_yaml = " []"

    # Create YAML content with comments
    yaml_content = f"""# Whisper OpenVINO ベンチマーク結果
# 生成日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# このファイルは音声認識モデルの性能測定結果を記録しています

# システム情報 - 実行環境の詳細
system_info:
  platform: "{results["system_info"]["platform"]}"  # オペレーティングシステム
  python: "{results["system_info"]["python"]}"      # Python バージョン
  cpu:
    name: "{results["system_info"]["cpu"]["name"]}"       # CPU名
    cores: {results["system_info"]["cpu"]["cores"]}       # 物理コア数
    threads: {results["system_info"]["cpu"]["threads"]}   # 論理プロセッサ数（ハイパースレッディング含む）
    detection_methods: "{results["system_info"]["cpu"].get("detection_methods", "Unknown")}"  # 検出手法
    features:{features_yaml}  # CPU拡張命令セット（OpenVINO性能に影響するもの）
  memory_gb: {results["system_info"]["memory_gb"]:.1f}    # システムメモリ容量（GB）

# ベンチマーク設定 - 測定条件
model_path: "{results["model_path"]}"          # 使用したWhisperモデルのパス
audio_duration: {results["audio_duration"]:.2f}           # 音声ファイルの長さ（秒）
num_beams: {results["num_beams"]}              # ビームサーチの幅（1=貪欲デコーディング）
device: "{results["device"]}"                  # 使用デバイス（CPU/GPU）
iterations: {results["iterations"]}            # ベンチマーク実行回数

# 性能結果 - 推論時間とメモリ使用量
performance:
  avg_time: {results["avg_time"]:.3f}          # 平均実行時間（秒）
  min_time: {results["min_time"]:.3f}          # 最短実行時間（秒）
  max_time: {results["max_time"]:.3f}          # 最長実行時間（秒）
  std_time: {results["std_time"]:.3f}          # 実行時間の標準偏差（秒）
  rtf: {results["rtf"]:.3f}                    # RTF（リアルタイムファクター）※1.0未満なら実時間より高速
  model_memory_gb: {results["model_memory_gb"]:.2f}     # モデルロード時のメモリ使用量（GB）
  inference_memory_mb: {results["inference_memory_mb"]:.1f} # 推論時の平均メモリ使用量（MB）

# 詳細データ - 各実行の生データ
detailed_times:{times_yaml}  # 各イテレーションの実行時間（秒）

# 認識結果 - 音声からテキストへの変換結果
transcription: "{results["transcription"]}"   # 音声認識で得られたテキスト
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def run_benchmark(model_path, audio_file, num_beams=1, device="CPU", iterations=5):
    """Run the benchmark with specified parameters"""
    # Validate model files
    if not validate_model_files(model_path):
        print("Warning: Model validation failed, but attempting to continue...")

    # Load audio
    print(f"Loading audio file: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_duration = len(audio) / sr
    print(f"Audio duration: {audio_duration:.2f} seconds")

    # Initialize pipeline with memory measurement
    print(f"Initializing WhisperPipeline with model: {model_path}")
    print(f"Device: {device}, Num beams: {num_beams}")

    # Measure memory before model loading
    process = psutil.Process()
    mem_before_model = process.memory_info().rss / 1024 / 1024  # MB

    try:
        pipe = WhisperPipeline(model_path, device=device)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the model was exported with --disable-stateful flag")
        print("2. Check that all required model files exist")
        print("3. Verify OpenVINO GenAI version compatibility")
        raise

    # Measure memory after model loading
    mem_after_model = process.memory_info().rss / 1024 / 1024  # MB
    model_memory_usage = (mem_after_model - mem_before_model) / 1024  # GB
    print(f"Model loaded. Memory usage: {model_memory_usage:.2f} GB")

    # Warm-up run
    print("Performing warm-up run...")
    try:
        _ = pipe.generate(audio, num_beams=num_beams)
    except Exception as e:
        print(f"Error during warm-up: {e}")
        raise

    # Benchmark runs
    print(f"\nPerforming {iterations} benchmark iterations...")
    times = []
    inference_memory_usage = []

    for i in range(iterations):
        # Get memory before inference
        mem_before_inference = process.memory_info().rss / 1024 / 1024  # MB

        # Time the inference
        start_time = time.time()
        result = pipe.generate(audio, num_beams=num_beams)
        end_time = time.time()

        # Get memory after inference
        mem_after_inference = process.memory_info().rss / 1024 / 1024  # MB

        inference_time = end_time - start_time
        times.append(inference_time)
        inference_memory_usage.append(mem_after_inference - mem_before_inference)

        print(
            f"Iteration {i + 1}/{iterations}: {inference_time:.3f}s (RTF: {inference_time / audio_duration:.3f})"
        )

    # Calculate statistics
    avg_time = mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = stdev(times) if len(times) > 1 else 0
    avg_inference_memory = mean(inference_memory_usage)
    rtf = avg_time / audio_duration

    return {
        "model_path": model_path,
        "audio_duration": audio_duration,
        "num_beams": num_beams,
        "device": device,
        "iterations": iterations,
        "times": times,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "rtf": rtf,
        "model_memory_gb": model_memory_usage,
        "inference_memory_mb": avg_inference_memory,
        "transcription": result,
    }


def run_streaming_benchmark(
    model_path: str,
    audio_file: str,
    chunk_size_sec: float = 30.0,
    overlap_sec: float = 0.0,
    num_beams: int = 1,
    device: str = "CPU",
    simulate_realtime: bool = False,
    verbose: bool = True,
) -> StreamingBenchmarkResults:
    """
    Run streaming benchmark with chunk-based processing.
    """
    # Load audio
    if verbose:
        print(f"Loading audio file: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=16000)
    total_audio_duration = len(audio) / sr

    if verbose:
        print(f"Audio duration: {total_audio_duration:.2f} seconds")
        print(f"Chunk size: {chunk_size_sec} seconds")
        if overlap_sec > 0:
            print(f"Overlap: {overlap_sec} seconds")

    # Split audio into chunks
    chunks = split_audio_into_chunks(audio, sr, chunk_size_sec, overlap_sec)
    num_chunks = len(chunks)

    if verbose:
        print(f"Number of chunks: {num_chunks}")

    # Initialize pipeline with memory measurement
    if verbose:
        print(f"\nInitializing WhisperPipeline with model: {model_path}")
        print(f"Device: {device}, Num beams: {num_beams}")

    process = psutil.Process()
    mem_before_model = process.memory_info().rss / 1024 / 1024  # MB

    try:
        pipe = WhisperPipeline(model_path, device=device)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        raise

    mem_after_model = process.memory_info().rss / 1024 / 1024  # MB
    model_memory_usage = (mem_after_model - mem_before_model) / 1024  # GB

    if verbose:
        print(f"Model loaded. Memory usage: {model_memory_usage:.2f} GB")

    # Warm-up with first chunk
    if verbose:
        print("\nPerforming warm-up...")
    _ = pipe.generate(chunks[0][0], num_beams=num_beams)

    # Process chunks
    if verbose:
        print(f"\nProcessing {num_chunks} chunks...")

    chunk_metrics = []
    inference_memory_usage = []
    cumulative_time = 0.0
    cumulative_audio_duration = 0.0
    transcriptions = []

    for i, (chunk_audio, start_time, end_time) in enumerate(chunks):
        chunk_duration = end_time - start_time

        # Simulate real-time if requested
        if simulate_realtime and i > 0:
            # Wait for the chunk duration before processing next chunk
            time.sleep(chunk_duration)

        # Measure memory before inference
        mem_before_inference = process.memory_info().rss / 1024 / 1024  # MB

        # Process chunk
        chunk_start_time = time.time()
        transcription = pipe.generate(chunk_audio, num_beams=num_beams)
        chunk_end_time = time.time()

        # Measure memory after inference
        mem_after_inference = process.memory_info().rss / 1024 / 1024  # MB
        inference_memory_usage.append(mem_after_inference - mem_before_inference)

        # Calculate metrics
        processing_time = chunk_end_time - chunk_start_time
        rtf = processing_time / chunk_duration
        cumulative_time += processing_time
        cumulative_audio_duration += chunk_duration
        cumulative_rtf = cumulative_time / cumulative_audio_duration

        # Store metrics
        metrics = ChunkMetrics(
            chunk_index=i,
            start_time=start_time,
            end_time=end_time,
            processing_time=processing_time,
            audio_duration=chunk_duration,
            rtf=rtf,
            transcription=str(transcription),
            cumulative_time=cumulative_time,
            cumulative_rtf=cumulative_rtf,
        )
        chunk_metrics.append(metrics)
        transcriptions.append(str(transcription))

        if verbose:
            print(
                f"Chunk {i + 1}/{num_chunks}: {processing_time:.3f}s "
                f"(RTF: {rtf:.3f}, Cumulative RTF: {cumulative_rtf:.3f})"
            )

    # Calculate statistics
    processing_times = [m.processing_time for m in chunk_metrics]
    first_chunk_latency = chunk_metrics[0].processing_time if chunk_metrics else 0

    # Calculate inter-chunk latencies
    inter_chunk_latencies = []
    for i in range(1, len(chunk_metrics)):
        # Time between end of previous chunk and start of current chunk processing
        inter_chunk_latencies.append(
            chunk_metrics[i].processing_time - chunk_metrics[i - 1].processing_time
        )

    # Compile results
    results = StreamingBenchmarkResults(
        chunk_metrics=chunk_metrics,
        total_audio_duration=total_audio_duration,
        total_processing_time=cumulative_time,
        overall_rtf=cumulative_time / total_audio_duration,
        first_chunk_latency=first_chunk_latency,
        avg_chunk_processing_time=mean(processing_times),
        min_chunk_processing_time=min(processing_times),
        max_chunk_processing_time=max(processing_times),
        std_chunk_processing_time=(
            stdev(processing_times) if len(processing_times) > 1 else 0
        ),
        throughput_audio_per_sec=total_audio_duration / cumulative_time,
        inter_chunk_latencies=inter_chunk_latencies,
        avg_inter_chunk_latency=(
            mean(inter_chunk_latencies) if inter_chunk_latencies else 0
        ),
        model_memory_gb=model_memory_usage,
        avg_inference_memory_mb=mean(inference_memory_usage),
        full_transcription=" ".join(transcriptions),
    )

    return results


def print_streaming_results(results: StreamingBenchmarkResults):
    """Print streaming benchmark results in a formatted table."""

    print("\n=== Streaming Benchmark Results ===")

    # Overall metrics table
    overall_data = [
        ["Metric", "Value"],
        ["Total Audio Duration", f"{results.total_audio_duration:.2f} seconds"],
        ["Total Processing Time", f"{results.total_processing_time:.3f} seconds"],
        ["Overall RTF", f"{results.overall_rtf:.3f}"],
        ["Throughput", f"{results.throughput_audio_per_sec:.2f} audio sec/sec"],
        ["Number of Chunks", len(results.chunk_metrics)],
    ]
    print("\n" + tabulate(overall_data, headers="firstrow", tablefmt="grid"))

    # Latency metrics table
    latency_data = [
        ["Latency Metric", "Value"],
        ["First Chunk Latency", f"{results.first_chunk_latency:.3f} seconds"],
        ["Avg Chunk Processing", f"{results.avg_chunk_processing_time:.3f} seconds"],
        ["Min Chunk Processing", f"{results.min_chunk_processing_time:.3f} seconds"],
        ["Max Chunk Processing", f"{results.max_chunk_processing_time:.3f} seconds"],
        ["Std Dev Processing", f"{results.std_chunk_processing_time:.3f} seconds"],
    ]

    if results.inter_chunk_latencies:
        latency_data.append(
            [
                "Avg Inter-chunk Latency",
                f"{results.avg_inter_chunk_latency:.3f} seconds",
            ]
        )

    print("\n" + tabulate(latency_data, headers="firstrow", tablefmt="grid"))

    # Memory metrics
    memory_data = [
        ["Memory Metric", "Value"],
        ["Model Memory Usage", f"{results.model_memory_gb:.2f} GB"],
        ["Avg Inference Memory", f"{results.avg_inference_memory_mb:.1f} MB"],
    ]
    print("\n" + tabulate(memory_data, headers="firstrow", tablefmt="grid"))

    # Per-chunk details
    print("\n=== Per-Chunk Details ===")
    chunk_data = [["Chunk", "Duration (s)", "Processing (s)", "RTF", "Cumulative RTF"]]
    for m in results.chunk_metrics:
        chunk_data.append(
            [
                f"{m.chunk_index + 1}",
                f"{m.audio_duration:.2f}",
                f"{m.processing_time:.3f}",
                f"{m.rtf:.3f}",
                f"{m.cumulative_rtf:.3f}",
            ]
        )
    print(tabulate(chunk_data, headers="firstrow", tablefmt="grid"))

    print(f"\nFull Transcription: {results.full_transcription}")


def save_streaming_yaml_results(
    results: StreamingBenchmarkResults,
    system_info: dict,
    model_path: str,
    chunk_size: float,
    overlap: float,
    num_beams: int,
    device: str,
    output_file: str,
):
    """Save streaming benchmark results to YAML file."""
    from datetime import datetime

    # Format features list for YAML
    features = system_info["cpu"]["features"]
    if features:
        features_yaml = "\n" + "\n".join(f"      - {feature}" for feature in features)
    else:
        features_yaml = " []"

    # Format chunk metrics for YAML
    chunks_yaml = ""
    for m in results.chunk_metrics:
        chunks_yaml += f"""
  - chunk_index: {m.chunk_index}
    start_time: {m.start_time:.2f}
    end_time: {m.end_time:.2f}
    audio_duration: {m.audio_duration:.2f}
    processing_time: {m.processing_time:.3f}
    rtf: {m.rtf:.3f}
    cumulative_rtf: {m.cumulative_rtf:.3f}"""

    # Format inter-chunk latencies
    if results.inter_chunk_latencies:
        latencies_yaml = "\n" + "\n".join(
            f"  - {lat:.3f}" for lat in results.inter_chunk_latencies
        )
    else:
        latencies_yaml = " []"

    # Create YAML content
    yaml_content = f"""# Whisper OpenVINO Streaming Benchmark Results
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Streaming mode with chunk-based processing

# System Information
system_info:
  platform: "{system_info["platform"]}"
  python: "{system_info["python"]}"
  cpu:
    name: "{system_info["cpu"]["name"]}"
    cores: {system_info["cpu"]["cores"]}
    threads: {system_info["cpu"]["threads"]}
    detection_methods: "{system_info["cpu"].get("detection_methods", "Unknown")}"
    features:{features_yaml}
  memory_gb: {system_info["memory_gb"]:.1f}

# Benchmark Configuration
configuration:
  model_path: "{model_path}"
  chunk_size_sec: {chunk_size}
  overlap_sec: {overlap}
  num_beams: {num_beams}
  device: "{device}"
  num_chunks: {len(results.chunk_metrics)}

# Overall Performance Metrics
performance:
  total_audio_duration: {results.total_audio_duration:.2f}
  total_processing_time: {results.total_processing_time:.3f}
  overall_rtf: {results.overall_rtf:.3f}
  throughput_audio_per_sec: {results.throughput_audio_per_sec:.2f}
  model_memory_gb: {results.model_memory_gb:.2f}
  avg_inference_memory_mb: {results.avg_inference_memory_mb:.1f}

# Latency Metrics
latency:
  first_chunk_latency: {results.first_chunk_latency:.3f}
  avg_chunk_processing_time: {results.avg_chunk_processing_time:.3f}
  min_chunk_processing_time: {results.min_chunk_processing_time:.3f}
  max_chunk_processing_time: {results.max_chunk_processing_time:.3f}
  std_chunk_processing_time: {results.std_chunk_processing_time:.3f}
  avg_inter_chunk_latency: {results.avg_inter_chunk_latency:.3f}

# Inter-chunk Latencies
inter_chunk_latencies:{latencies_yaml}

# Per-Chunk Metrics
chunk_metrics:{chunks_yaml}

# Transcription Result
transcription: "{results.full_transcription}"
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def main():
    parser = argparse.ArgumentParser(description="WhisperPipeline Benchmark Tool")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the Whisper model directory",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for decoding (default: 1)",
    )
    parser.add_argument(
        "--device", type=str, default="CPU", help="Device to use (default: CPU)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file for benchmarking (if not specified, downloads default file)",
    )
    parser.add_argument(
        "--output-yaml",
        type=str,
        default="benchmark_results.yaml",
        help="Output results to YAML file (default: benchmark_results.yaml)",
    )
    # Streaming mode arguments
    parser.add_argument(
        "--streaming-mode",
        action="store_true",
        help="Enable streaming mode with chunk-based processing",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=30.0,
        help="Chunk size in seconds for streaming mode (default: 30.0)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap between chunks in seconds for streaming mode (default: 0.0)",
    )
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        help="Simulate real-time processing by waiting between chunks (streaming mode only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output during processing (streaming mode only)",
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Print system information
    print("=== System Information ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")

    cpu_info = get_cpu_info()
    print(f"CPU: {cpu_info['name']}")
    print(f"Cores: {cpu_info['cores']}, Threads: {cpu_info['threads']}")
    if cpu_info["features"]:
        print(f"Features: {', '.join(cpu_info['features'])}")

    print(f"Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print()

    # Get audio file (specified or default)
    try:
        audio_file = get_audio_file(args.audio_file)
    except Exception as e:
        print(f"Failed to get audio file: {e}")
        sys.exit(1)

    # System info for both modes
    system_info = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "cpu": cpu_info,
        "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
    }

    # Run appropriate benchmark based on mode
    if args.streaming_mode:
        # Run streaming benchmark
        try:
            results = run_streaming_benchmark(
                model_path=args.model_path,
                audio_file=audio_file,
                chunk_size_sec=args.chunk_size,
                overlap_sec=args.overlap,
                num_beams=args.num_beams,
                device=args.device,
                simulate_realtime=args.simulate_realtime,
                verbose=not args.quiet,
            )
        except Exception as e:
            print(f"Streaming benchmark failed: {e}")
            sys.exit(1)

        # Print streaming results
        print_streaming_results(results)

        # Save streaming results to YAML
        output_path = args.output_yaml
        if not output_path.endswith("_streaming.yaml"):
            # Modify output filename for streaming mode
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_streaming{ext}"

        output_dir = os.environ.get("OUTPUT_DIR")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(output_path))

        save_streaming_yaml_results(
            results=results,
            system_info=system_info,
            model_path=args.model_path,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            num_beams=args.num_beams,
            device=args.device,
            output_file=output_path,
        )
        print(f"\nResults saved to: {output_path}")

    else:
        # Run standard benchmark
        try:
            results = run_benchmark(
                args.model_path,
                audio_file,
                num_beams=args.num_beams,
                device=args.device,
                iterations=args.iterations,
            )
        except Exception as e:
            print(f"Benchmark failed: {e}")
            sys.exit(1)

        # Add system info to results
        results["system_info"] = system_info

        # Print results table
        print("\n=== Benchmark Results ===")
        table_data = [
            ["Metric", "Value"],
            ["Model Path", args.model_path],
            ["Audio Duration", f"{results['audio_duration']:.2f} seconds"],
            ["Iterations", results["iterations"]],
            ["Average Time", f"{results['avg_time']:.3f} seconds"],
            ["Min Time", f"{results['min_time']:.3f} seconds"],
            ["Max Time", f"{results['max_time']:.3f} seconds"],
            ["Std Dev", f"{results['std_time']:.3f} seconds"],
            ["Real-time Factor", f"{results['rtf']:.3f}"],
            ["Model Memory Usage", f"{results['model_memory_gb']:.2f} GB"],
            ["Inference Memory Usage", f"{results['inference_memory_mb']:.1f} MB"],
        ]
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

        print(f"\nTranscription: {results['transcription']}")

        # Save to YAML (default enabled)
        # 環境変数OUTPUT_DIRが設定されている場合は、そのディレクトリに保存
        output_path = args.output_yaml
        output_dir = os.environ.get("OUTPUT_DIR")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(args.output_yaml))

        save_yaml_results(results, output_path)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
