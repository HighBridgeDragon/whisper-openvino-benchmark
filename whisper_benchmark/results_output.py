"""
結果出力とYAML保存機能

ベンチマーク結果の表示とファイル保存機能を提供
"""

from datetime import datetime
from tabulate import tabulate
from .streaming_metrics import StreamingMetrics


def save_yaml_results(results, output_file):
    """
    ベンチマーク結果をYAMLファイルに詳細コメント付きで保存

    Args:
        results: ベンチマーク結果辞書
        output_file: 出力ファイルパス
    """
    # 機能リストをYAML用にフォーマット
    features = results["system_info"]["cpu"]["features"]
    if features:
        features_yaml = "\n" + "\n".join(f"      - {feature}" for feature in features)
    else:
        features_yaml = " []"

    # 詳細時間リストをYAML用にフォーマット
    times = results["times"]
    if times:
        times_yaml = "\n" + "\n".join(f"  - {time:.3f}" for time in times)
    else:
        times_yaml = " []"

    # コメント付きYAMLコンテンツを作成
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


def print_streaming_results(results: StreamingMetrics):
    """ストリーミングベンチマーク結果をフォーマット済みテーブルで表示"""

    # メトリクスを計算
    results.calculate_aggregate_metrics()

    print("\n=== Streaming Benchmark Results ===")

    # 全体メトリクステーブル
    overall_data = [
        ["Metric", "Value"],
        ["Total Audio Duration", f"{results.total_audio_duration:.2f} seconds"],
        ["Total Processing Time", f"{results.total_processing_time:.3f} seconds"],
        ["Overall RTF", f"{results.overall_rtf:.3f}"],
        ["Throughput", f"{results.throughput_audio_per_sec:.2f} audio sec/sec"],
        ["Number of Chunks", results.total_chunks],
        ["Real-time Capable", "Yes" if results.is_realtime_capable() else "No"],
    ]
    print("\n" + tabulate(overall_data, headers="firstrow", tablefmt="grid"))

    # レイテンシメトリクステーブル
    latency_data = [
        ["Latency Metric", "Value"],
        ["First Chunk Latency", f"{results.first_chunk_latency:.3f} seconds"],
        ["First Token Latency", f"{results.first_token_latency:.3f} seconds"],
        ["Avg Latency", f"{results.avg_latency:.3f} seconds"],
        ["Min Latency", f"{results.min_latency:.3f} seconds"],
        ["Max Latency", f"{results.max_latency:.3f} seconds"],
    ]
    print("\n" + tabulate(latency_data, headers="firstrow", tablefmt="grid"))

    # 処理時間メトリクステーブル
    processing_data = [
        ["Processing Metric", "Value"],
        ["Avg Processing Time", f"{results.avg_processing_time:.3f} seconds"],
        ["Min Processing Time", f"{results.min_processing_time:.3f} seconds"],
        ["Max Processing Time", f"{results.max_processing_time:.3f} seconds"],
        ["Std Dev Processing", f"{results.std_processing_time:.3f} seconds"],
        ["Median Processing Time", f"{results.median_processing_time:.3f} seconds"],
    ]
    print("\n" + tabulate(processing_data, headers="firstrow", tablefmt="grid"))

    # ストリーミング品質メトリクス
    quality_data = [
        ["Streaming Quality", "Value"],
        ["Buffer Underruns", results.buffer_underruns],
        ["Buffer Overruns", results.buffer_overruns],
        ["Avg Buffer Size", f"{results.avg_buffer_size:.1f}"],
        ["Max Buffer Size", results.max_buffer_size],
        ["Throughput Stability (CV)", f"{results.throughput_stability:.3f}"],
        ["Cumulative Drift", f"{results.cumulative_drift:.3f} seconds"],
        ["Max Drift", f"{results.max_drift:.3f} seconds"],
    ]
    print("\n" + tabulate(quality_data, headers="firstrow", tablefmt="grid"))

    # メモリメトリクス
    memory_data = [
        ["Memory Metric", "Value"],
        ["Model Memory Usage", f"{results.model_memory_gb:.2f} GB"],
        ["Avg Inference Memory", f"{results.avg_inference_memory_mb:.1f} MB"],
        ["Peak Memory Usage", f"{results.peak_memory_mb:.1f} MB"],
    ]
    print("\n" + tabulate(memory_data, headers="firstrow", tablefmt="grid"))

    # RTFサマリー
    rtf_data = [
        ["RTF Metric", "Value"],
        ["Overall RTF", f"{results.overall_rtf:.3f}"],
        ["Avg Chunk RTF", f"{results.avg_chunk_rtf:.3f}"],
        ["Min Chunk RTF", f"{results.min_chunk_rtf:.3f}"],
        ["Max Chunk RTF", f"{results.max_chunk_rtf:.3f}"],
        ["RTF Consistency (StdDev)", f"{results.rtf_consistency:.3f}"],
    ]
    print("\n" + tabulate(rtf_data, headers="firstrow", tablefmt="grid"))

    # チャンク別詳細（最初の10チャンクのみ表示）
    if results.chunk_metrics:
        print("\n=== Per-Chunk Details (first 10 chunks) ===")
        chunk_data = [
            ["Chunk", "Audio Duration", "Processing", "RTF", "Latency", "Buffer"]
        ]
        for m in results.chunk_metrics[:10]:
            chunk_data.append(
                [
                    f"{m.chunk_index}",
                    f"{m.audio_duration:.2f}s",
                    f"{m.processing_duration:.3f}s",
                    f"{m.rtf:.3f}",
                    f"{m.latency:.3f}s",
                    m.buffer_size_at_arrival,
                ]
            )
        print(tabulate(chunk_data, headers="firstrow", tablefmt="grid"))

        if len(results.chunk_metrics) > 10:
            print(f"... and {len(results.chunk_metrics) - 10} more chunks")

    print(
        f"\nFull Transcription: {results.full_transcription[:200]}..."
        if len(results.full_transcription) > 200
        else f"\nFull Transcription: {results.full_transcription}"
    )


def save_streaming_yaml_results(
    results: StreamingMetrics,
    system_info: dict,
    model_path: str,
    chunk_size: float,
    overlap: float,
    num_beams: int,
    device: str,
    output_file: str,
):
    """ストリーミングベンチマーク結果をYAMLファイルに保存"""

    # 機能リストをYAML用にフォーマット
    features = system_info["cpu"]["features"]
    if features:
        features_yaml = "\n" + "\n".join(f"      - {feature}" for feature in features)
    else:
        features_yaml = " []"

    # チャンクメトリクスをYAML用にフォーマット（最初の20個のみ）
    chunks_yaml = ""
    for m in results.chunk_metrics[:20]:
        chunks_yaml += f"""
  - chunk_index: {m.chunk_index}
    start_time: {m.start_time:.2f}
    end_time: {m.end_time:.2f}
    audio_duration: {m.audio_duration:.2f}
    processing_duration: {m.processing_duration:.3f}
    rtf: {m.rtf:.3f}
    latency: {m.latency:.3f}
    buffer_size: {m.buffer_size_at_arrival}"""

    if len(results.chunk_metrics) > 20:
        chunks_yaml += f"\n  # ... and {len(results.chunk_metrics) - 20} more chunks"

    # サマリー情報を作成
    summary = results.get_summary()

    # YAMLコンテンツを作成
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
  total_audio_duration: {summary["overview"]["total_audio_duration"]:.2f}
  total_processing_time: {summary["overview"]["total_processing_time"]:.3f}
  overall_rtf: {summary["overview"]["overall_rtf"]:.3f}
  throughput_audio_per_sec: {summary["overview"]["throughput_audio_per_sec"]:.2f}
  total_chunks: {summary["overview"]["total_chunks"]}
  is_realtime_capable: {results.is_realtime_capable()}

# Latency Metrics
latency:
  first_chunk_latency: {summary["latency"]["first_chunk_latency"]:.3f}
  first_token_latency: {summary["latency"]["first_token_latency"]:.3f}
  avg_latency: {summary["latency"]["avg_latency"]:.3f}
  min_latency: {summary["latency"]["min_latency"]:.3f}
  max_latency: {summary["latency"]["max_latency"]:.3f}

# Processing Time Metrics
processing:
  avg_processing_time: {summary["processing"]["avg_processing_time"]:.3f}
  std_processing_time: {summary["processing"]["std_processing_time"]:.3f}
  min_processing_time: {summary["processing"]["min_processing_time"]:.3f}
  max_processing_time: {summary["processing"]["max_processing_time"]:.3f}
  median_processing_time: {summary["processing"]["median_processing_time"]:.3f}

# RTF Metrics
rtf_metrics:
  overall_rtf: {summary["rtf"]["overall_rtf"]:.3f}
  avg_chunk_rtf: {summary["rtf"]["avg_chunk_rtf"]:.3f}
  min_chunk_rtf: {summary["rtf"]["min_chunk_rtf"]:.3f}
  max_chunk_rtf: {summary["rtf"]["max_chunk_rtf"]:.3f}
  rtf_consistency: {summary["rtf"]["rtf_consistency"]:.3f}

# Streaming Quality
streaming_quality:
  buffer_underruns: {summary["streaming_quality"]["buffer_underruns"]}
  buffer_overruns: {summary["streaming_quality"]["buffer_overruns"]}
  avg_buffer_size: {summary["streaming_quality"]["avg_buffer_size"]:.1f}
  max_buffer_size: {summary["streaming_quality"]["max_buffer_size"]}
  throughput_stability: {summary["streaming_quality"]["throughput_stability"]:.3f}

# Drift Metrics
drift:
  cumulative_drift: {summary["drift"]["cumulative_drift"]:.3f}
  max_drift: {summary["drift"]["max_drift"]:.3f}
  drift_recovery_points: {summary["drift"]["drift_recovery_points"]}

# Memory Usage
memory:
  model_memory_gb: {summary["memory"]["model_memory_gb"]:.2f}
  avg_inference_memory_mb: {summary["memory"]["avg_inference_memory_mb"]:.1f}
  peak_memory_mb: {summary["memory"]["peak_memory_mb"]:.1f}

# Per-Chunk Metrics
chunk_metrics:{chunks_yaml}

# Transcription Result
transcription: "{results.full_transcription}"
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)
