"""
結果出力とYAML保存機能

ベンチマーク結果の表示とファイル保存機能を提供
"""

from datetime import datetime
from tabulate import tabulate
from .streaming_benchmark import StreamingBenchmarkResults


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


def print_streaming_results(results: StreamingBenchmarkResults):
    """ストリーミングベンチマーク結果をフォーマット済みテーブルで表示"""

    print("\n=== Streaming Benchmark Results ===")

    # 全体メトリクステーブル
    overall_data = [
        ["Metric", "Value"],
        ["Total Audio Duration", f"{results.total_audio_duration:.2f} seconds"],
        ["Total Processing Time", f"{results.total_processing_time:.3f} seconds"],
        ["Overall RTF", f"{results.overall_rtf:.3f}"],
        ["Throughput", f"{results.throughput_audio_per_sec:.2f} audio sec/sec"],
        ["Number of Chunks", len(results.chunk_metrics)],
    ]
    print("\n" + tabulate(overall_data, headers="firstrow", tablefmt="grid"))

    # レイテンシメトリクステーブル
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

    # メモリメトリクス
    memory_data = [
        ["Memory Metric", "Value"],
        ["Model Memory Usage", f"{results.model_memory_gb:.2f} GB"],
        ["Avg Inference Memory", f"{results.avg_inference_memory_mb:.1f} MB"],
    ]
    print("\n" + tabulate(memory_data, headers="firstrow", tablefmt="grid"))

    # チャンク別詳細
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
    """ストリーミングベンチマーク結果をYAMLファイルに保存"""

    # 機能リストをYAML用にフォーマット
    features = system_info["cpu"]["features"]
    if features:
        features_yaml = "\n" + "\n".join(f"      - {feature}" for feature in features)
    else:
        features_yaml = " []"

    # チャンクメトリクスをYAML用にフォーマット
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

    # チャンク間レイテンシをフォーマット
    if results.inter_chunk_latencies:
        latencies_yaml = "\n" + "\n".join(
            f"  - {lat:.3f}" for lat in results.inter_chunk_latencies
        )
    else:
        latencies_yaml = " []"

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
