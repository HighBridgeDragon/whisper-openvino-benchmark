import argparse
import json
import os
import platform
import subprocess
import sys
import time
from statistics import mean, stdev

import librosa
import psutil
import requests
from openvino_genai import WhisperPipeline
from tabulate import tabulate


def get_cpu_info():
    """Get CPU information using Windows WMI"""
    try:
        # Get CPU name
        result = subprocess.run(
            ["wmic", "cpu", "get", "name", "/value"],
            capture_output=True,
            text=True,
            check=True,
        )
        cpu_name = ""
        for line in result.stdout.strip().split("\n"):
            if line.startswith("Name="):
                cpu_name = line.split("=", 1)[1].strip()
                break

        # Get number of cores and threads
        result = subprocess.run(
            ["wmic", "cpu", "get", "NumberOfCores,NumberOfLogicalProcessors", "/value"],
            capture_output=True,
            text=True,
            check=True,
        )

        cores = 0
        threads = 0
        for line in result.stdout.strip().split("\n"):
            if line.startswith("NumberOfCores="):
                cores = int(line.split("=", 1)[1].strip())
            elif line.startswith("NumberOfLogicalProcessors="):
                threads = int(line.split("=", 1)[1].strip())

        # Detect CPU features
        features = []
        try:
            # Check for AVX2
            result = subprocess.run(
                ["wmic", "cpu", "get", "Description", "/value"],
                capture_output=True,
                text=True,
                check=True,
            )
            description = result.stdout.lower()

            # Basic feature detection based on CPU generation
            if "avx2" in description or any(
                gen in cpu_name.lower()
                for gen in [
                    "core i7-4",
                    "core i5-4",
                    "core i3-4",
                    "xeon e3-12",
                    "xeon e5-26",
                ]
            ):
                features.append("AVX2")
            if "avx512" in description or any(
                gen in cpu_name.lower() for gen in ["core i9", "xeon", "core x"]
            ):
                features.append("AVX512")
        except:
            pass

        return {
            "name": cpu_name,
            "cores": cores,
            "threads": threads,
            "features": features,
        }
    except Exception as e:
        print(f"Warning: Could not get CPU info: {e}")
        return {
            "name": "Unknown",
            "cores": psutil.cpu_count(logical=False) or 1,
            "threads": psutil.cpu_count(logical=True) or 1,
            "features": [],
        }


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

    print("✓ All required model files found")
    return True


def run_benchmark(
    model_path, audio_file, num_beams=1, language="<|en|>", device="CPU", iterations=5
):
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
    print(f"Device: {device}, Num beams: {num_beams}, Language: {language}")

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
        _ = pipe.generate(audio, language=language, num_beams=num_beams)
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
        result = pipe.generate(audio, language=language, num_beams=num_beams)
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
        "language": language,
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
        "--language", type=str, default="<|en|>", help="Language code (default: <|en|>)"
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
    parser.add_argument("--output-json", type=str, help="Output results to JSON file")

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

    # Download audio file
    audio_url = "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav"
    try:
        audio_file = download_audio_file(audio_url)
    except Exception as e:
        print(f"Failed to download audio file: {e}")
        sys.exit(1)

    # Run benchmark
    try:
        results = run_benchmark(
            args.model_path,
            audio_file,
            num_beams=args.num_beams,
            language=args.language,
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

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
