import argparse
import os
import platform
import subprocess
import sys
import time
from statistics import mean, stdev

import cpuinfo
import librosa
import psutil
import requests
from openvino_genai import WhisperPipeline
from tabulate import tabulate


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
  avg_memory_mb: {results["avg_memory_mb"]:.1f} # 平均メモリ使用量（MB）

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

    # Initialize pipeline
    print(f"Initializing WhisperPipeline with model: {model_path}")
    print(f"Device: {device}, Num beams: {num_beams}")

    try:
        pipe = WhisperPipeline(model_path, device=device)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the model was exported with --disable-stateful flag")
        print("2. Check that all required model files exist")
        print("3. Verify OpenVINO GenAI version compatibility")
        raise

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
    memory_usage = []

    for i in range(iterations):
        # Get memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time the inference
        start_time = time.time()
        result = pipe.generate(audio, num_beams=num_beams)
        end_time = time.time()

        # Get memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        inference_time = end_time - start_time
        times.append(inference_time)
        memory_usage.append(mem_after - mem_before)

        print(
            f"Iteration {i + 1}/{iterations}: {inference_time:.3f}s (RTF: {inference_time / audio_duration:.3f})"
        )

    # Calculate statistics
    avg_time = mean(times)
    min_time = min(times)
    max_time = max(times)
    std_time = stdev(times) if len(times) > 1 else 0
    avg_memory = mean(memory_usage)
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
        "avg_memory_mb": avg_memory,
        "transcription": result,
    }


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

    # Run benchmark
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
    results["system_info"] = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "cpu": cpu_info,
        "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
    }

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
        ["Avg Memory Usage", f"{results['avg_memory_mb']:.1f} MB"],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

    print(f"\nTranscription: {results['transcription']}")

    # Save to YAML (default enabled)
    save_yaml_results(results, args.output_yaml)
    print(f"\nResults saved to: {args.output_yaml}")


if __name__ == "__main__":
    main()
