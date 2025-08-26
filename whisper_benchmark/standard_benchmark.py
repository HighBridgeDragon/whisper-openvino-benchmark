"""
標準ベンチマーク機能

従来の反復ベンチマーク測定機能を提供
"""

import time
from statistics import mean, stdev

import librosa
import psutil
from openvino_genai import WhisperPipeline


def run_benchmark(
    model_path, audio_file, num_beams=1, device="CPU", iterations=5, language=None
):
    """
    指定されたパラメータでベンチマークを実行

    Args:
        model_path: Whisperモデルのパス
        audio_file: 音声ファイルのパス
        num_beams: ビーム数（デフォルト: 1）
        device: 使用デバイス（デフォルト: "CPU"）
        iterations: イテレーション数（デフォルト: 5）
        language: 言語トークン（例: "<|ja|>", "<|en|>"）。Noneの場合は自動検出

    Returns:
        ベンチマーク結果の辞書
    """
    # 音声読み込み
    print(f"Loading audio file: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_duration = len(audio) / sr
    print(f"Audio duration: {audio_duration:.2f} seconds")

    # パイプライン初期化とメモリ測定
    print(f"Initializing WhisperPipeline with model: {model_path}")
    print(f"Device: {device}, Num beams: {num_beams}")

    # モデル読み込み前のメモリ測定
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

    # モデル読み込み後のメモリ測定
    mem_after_model = process.memory_info().rss / 1024 / 1024  # MB
    model_memory_usage = (mem_after_model - mem_before_model) / 1024  # GB
    print(f"Model loaded. Memory usage: {model_memory_usage:.2f} GB")

    # ウォームアップ実行
    print("Performing warm-up run...")
    if language:
        print(f"Language specified: {language}")
    try:
        generate_kwargs = {"num_beams": num_beams}
        if language:
            generate_kwargs["language"] = language
        _ = pipe.generate(audio, **generate_kwargs)
    except Exception as e:
        print(f"Error during warm-up: {e}")
        raise

    # ベンチマーク実行
    print(f"\nPerforming {iterations} benchmark iterations...")
    times = []
    inference_memory_usage = []

    for i in range(iterations):
        # 推論前のメモリ取得
        mem_before_inference = process.memory_info().rss / 1024 / 1024  # MB

        # 推論時間を測定
        start_time = time.time()
        generate_kwargs = {"num_beams": num_beams}
        if language:
            generate_kwargs["language"] = language
        result = pipe.generate(audio, **generate_kwargs)
        end_time = time.time()

        # 推論後のメモリ取得
        mem_after_inference = process.memory_info().rss / 1024 / 1024  # MB

        inference_time = end_time - start_time
        times.append(inference_time)
        inference_memory_usage.append(mem_after_inference - mem_before_inference)

        print(
            f"Iteration {i + 1}/{iterations}: {inference_time:.3f}s (RTF: {inference_time / audio_duration:.3f})"
        )

    # 統計計算
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
