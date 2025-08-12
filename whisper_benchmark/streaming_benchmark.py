"""
ストリーミングベンチマーク機能

チャンク単位での音声処理とパフォーマンス測定
"""

import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import List, Tuple

import librosa
import numpy as np
import psutil
from openvino_genai import WhisperPipeline


@dataclass
class ChunkMetrics:
    """単一の音声チャンクのメトリクス"""

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
    """ストリーミングベンチマークの結果"""

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


def split_audio_into_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_size_sec: float = 30.0,
    overlap_sec: float = 0.0,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    音声をチャンクに分割（オプションでオーバーラップ付き）

    Args:
        audio: 音声データ
        sample_rate: サンプリングレート
        chunk_size_sec: チャンクサイズ（秒）
        overlap_sec: オーバーラップ時間（秒）

    Returns:
        チャンクのリスト [(chunk_audio, start_time, end_time), ...]
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
    ストリーミングベンチマークをチャンクベース処理で実行

    Args:
        model_path: Whisperモデルのパス
        audio_file: 音声ファイルのパス
        chunk_size_sec: チャンクサイズ（秒）
        overlap_sec: オーバーラップ時間（秒）
        num_beams: ビーム数
        device: 使用デバイス
        simulate_realtime: リアルタイム処理をシミュレート
        verbose: 詳細出力

    Returns:
        ベンチマーク結果
    """
    # 音声読み込み
    if verbose:
        print(f"Loading audio file: {audio_file}")
    audio, sr = librosa.load(audio_file, sr=16000)
    total_audio_duration = len(audio) / sr

    if verbose:
        print(f"Audio duration: {total_audio_duration:.2f} seconds")
        print(f"Chunk size: {chunk_size_sec} seconds")
        if overlap_sec > 0:
            print(f"Overlap: {overlap_sec} seconds")

    # 音声をチャンクに分割
    chunks = split_audio_into_chunks(audio, sr, chunk_size_sec, overlap_sec)
    num_chunks = len(chunks)

    if verbose:
        print(f"Number of chunks: {num_chunks}")

    # パイプライン初期化とメモリ測定
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

    # 最初のチャンクでウォームアップ
    if verbose:
        print("\nPerforming warm-up...")
    _ = pipe.generate(chunks[0][0], num_beams=num_beams)

    # チャンク処理
    if verbose:
        print(f"\nProcessing {num_chunks} chunks...")

    chunk_metrics = []
    inference_memory_usage = []
    cumulative_time = 0.0
    cumulative_audio_duration = 0.0
    transcriptions = []

    for i, (chunk_audio, start_time, end_time) in enumerate(chunks):
        chunk_duration = end_time - start_time

        # リアルタイムシミュレート（必要な場合）
        if simulate_realtime and i > 0:
            time.sleep(chunk_duration)

        # 推論前のメモリ測定
        mem_before_inference = process.memory_info().rss / 1024 / 1024  # MB

        # チャンク処理
        chunk_start_time = time.time()
        transcription = pipe.generate(chunk_audio, num_beams=num_beams)
        chunk_end_time = time.time()

        # 推論後のメモリ測定
        mem_after_inference = process.memory_info().rss / 1024 / 1024  # MB
        inference_memory_usage.append(mem_after_inference - mem_before_inference)

        # メトリクス計算
        processing_time = chunk_end_time - chunk_start_time
        rtf = processing_time / chunk_duration
        cumulative_time += processing_time
        cumulative_audio_duration += chunk_duration
        cumulative_rtf = cumulative_time / cumulative_audio_duration

        # メトリクス保存
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

    # 統計計算
    processing_times = [m.processing_time for m in chunk_metrics]
    first_chunk_latency = chunk_metrics[0].processing_time if chunk_metrics else 0

    # チャンク間レイテンシ計算
    inter_chunk_latencies = []
    for i in range(1, len(chunk_metrics)):
        inter_chunk_latencies.append(
            chunk_metrics[i].processing_time - chunk_metrics[i - 1].processing_time
        )

    # 結果をコンパイル
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
