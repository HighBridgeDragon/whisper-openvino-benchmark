"""
Whisper OpenVINO Benchmark Tool

OpenVINO GenAIのWhisperPipelineを使用したベンチマークツール
音声認識モデルのパフォーマンスを測定し、推論時間、メモリ使用量、
リアルタイムファクター（RTF）などの指標を収集します。
"""

import argparse
import os
import sys

from tabulate import tabulate

# whisper_benchmarkパッケージからモジュールをインポート
from whisper_benchmark import (
    get_audio_file,
    print_streaming_results,
    run_benchmark,
    run_streaming_benchmark,
    save_streaming_yaml_results,
    save_yaml_results,
    validate_model_files,
)
from whisper_benchmark.system_info import get_system_info, print_system_info


def main():
    """メイン関数 - コマンドライン引数を処理してベンチマークを実行"""
    parser = argparse.ArgumentParser(description="WhisperPipeline Benchmark Tool")

    # 基本引数
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

    # ストリーミングモード引数
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

    # モデルパス検証
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)

    # システム情報表示
    print_system_info()

    # 音声ファイル取得（指定またはデフォルト）
    try:
        audio_file = get_audio_file(args.audio_file)
    except Exception as e:
        print(f"Failed to get audio file: {e}")
        sys.exit(1)

    # モデルファイル検証
    if not validate_model_files(args.model_path):
        print("Warning: Model validation failed, but attempting to continue...")

    # システム情報取得（両モード共通）
    system_info = get_system_info()

    # 適切なベンチマークをモードに応じて実行
    if args.streaming_mode:
        # ストリーミングベンチマーク実行
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

        # ストリーミング結果表示
        print_streaming_results(results)

        # ストリーミング結果をYAMLに保存
        output_path = args.output_yaml
        if not output_path.endswith("_streaming.yaml"):
            # ストリーミングモード用にファイル名を修正
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
        # 標準ベンチマーク実行
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

        # システム情報を結果に追加
        results["system_info"] = system_info

        # 結果テーブル表示
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

        # YAMLに保存（デフォルト有効）
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
