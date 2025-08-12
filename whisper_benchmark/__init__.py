"""
Whisper OpenVINO Benchmark Package

音声認識モデルの性能測定を行うためのベンチマークツール
"""

__version__ = "1.0.0"
__author__ = "Claude Assistant"

from .system_info import get_cpu_info
from .streaming_benchmark import run_streaming_benchmark, StreamingBenchmarkResults
from .standard_benchmark import run_benchmark
from .results_output import (
    save_yaml_results,
    save_streaming_yaml_results,
    print_streaming_results,
)
from .audio_utils import get_audio_file, download_audio_file
from .model_utils import validate_model_files

__all__ = [
    "get_cpu_info",
    "run_streaming_benchmark",
    "run_benchmark",
    "StreamingBenchmarkResults",
    "save_yaml_results",
    "save_streaming_yaml_results",
    "print_streaming_results",
    "get_audio_file",
    "download_audio_file",
    "validate_model_files",
]
