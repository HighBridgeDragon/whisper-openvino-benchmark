"""
真のストリーミングベンチマーク実装

マルチスレッドを使用したプロデューサー/コンシューマーパターンで
リアルタイム音声ストリーミングをシミュレート
"""

import time
import threading
import queue
from typing import Any
from dataclasses import dataclass
import psutil
import numpy as np
from openvino_genai import WhisperPipeline

from .audio_stream import AudioStream, AudioBufferStream
from .streaming_metrics import StreamingMetrics, ChunkMetrics


@dataclass
class StreamingConfig:
    """ストリーミング設定"""

    chunk_duration: float = 1.0  # チャンクサイズ（秒）
    overlap: float = 0.0  # オーバーラップ（秒）
    buffer_size: int = 5  # バッファサイズ（チャンク数）
    realtime: bool = True  # リアルタイムシミュレーション
    num_beams: int = 1  # ビーム数
    device: str = "CPU"  # デバイス
    verbose: bool = True  # 詳細出力
    max_chunk_duration: float = 30.0  # WhisperPipelineの制限
    language: str = None  # 言語トークン（例：'<|ja|>', '<|en|>'）


class StreamingProcessor:
    """
    ストリーミング処理プロセッサ

    プロデューサースレッドが音声チャンクを生成し、
    コンシューマースレッドがWhisperPipelineで処理する
    """

    def __init__(self, model_path: str, config: StreamingConfig):
        """
        Args:
            model_path: Whisperモデルパス
            config: ストリーミング設定
        """
        self.model_path = model_path
        self.config = config

        # メモリ計測（パイプライン初期化前に設定）
        self.process = psutil.Process()
        self.model_memory_gb = 0.0
        self.peak_memory_mb = 0.0

        # パイプライン初期化
        self.pipeline = None
        self._init_pipeline()

        # キューとスレッド管理
        self.chunk_queue = queue.Queue(maxsize=config.buffer_size)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        # メトリクス
        self.metrics = StreamingMetrics()
        self.start_time = None

        # スレッド
        self.producer_thread = None
        self.consumer_thread = None

    def _init_pipeline(self):
        """WhisperPipelineを初期化"""
        if self.config.verbose:
            print(f"Initializing WhisperPipeline: {self.model_path}")
            print(f"Device: {self.config.device}, Num beams: {self.config.num_beams}")

        # メモリ測定
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB

        try:
            self.pipeline = WhisperPipeline(self.model_path, device=self.config.device)
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            raise

        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        self.model_memory_gb = (mem_after - mem_before) / 1024  # GB

        if self.config.verbose:
            print(f"Model loaded. Memory usage: {self.model_memory_gb:.2f} GB")

    def _producer_worker(self, audio_stream: AudioStream):
        """
        プロデューサースレッド: 音声チャンクを生成してキューに追加
        """
        if self.config.verbose:
            print("Producer thread started")

        for chunk_data in audio_stream.stream_chunks():
            if self.stop_event.is_set():
                break

            chunk_audio, start_time, end_time, chunk_index = chunk_data

            # チャンクが長すぎる場合は分割
            chunk_duration = end_time - start_time
            if chunk_duration > self.config.max_chunk_duration:
                # サブチャンクに分割
                sub_chunks = self._split_long_chunk(
                    chunk_audio, start_time, end_time, chunk_index
                )
                for sub_chunk in sub_chunks:
                    try:
                        arrival_time = time.perf_counter() - self.start_time
                        buffer_size = self.chunk_queue.qsize()

                        self.chunk_queue.put(
                            (sub_chunk, arrival_time, buffer_size), timeout=1.0
                        )

                        if self.config.verbose:
                            print(
                                f"Producer: Added sub-chunk to queue (buffer: {buffer_size}/{self.config.buffer_size})"
                            )

                    except queue.Full:
                        self.metrics.buffer_overruns += 1
                        if self.config.verbose:
                            print("Producer: Buffer overflow, dropping chunk")
            else:
                # 通常のチャンク処理
                try:
                    arrival_time = time.perf_counter() - self.start_time
                    buffer_size = self.chunk_queue.qsize()

                    self.chunk_queue.put(
                        (
                            (chunk_audio, start_time, end_time, chunk_index),
                            arrival_time,
                            buffer_size,
                        ),
                        timeout=1.0,
                    )

                    if self.config.verbose:
                        print(
                            f"Producer: Added chunk {chunk_index} to queue (buffer: {buffer_size}/{self.config.buffer_size})"
                        )

                except queue.Full:
                    self.metrics.buffer_overruns += 1
                    if self.config.verbose:
                        print("Producer: Buffer overflow, dropping chunk")

        if self.config.verbose:
            print("Producer thread finished")

    def _split_long_chunk(self, chunk_audio, start_time, end_time, chunk_index):
        """長いチャンクを30秒以下のサブチャンクに分割"""
        max_samples = int(self.config.max_chunk_duration * 16000)  # 16kHz

        sub_chunks = []
        audio_len = len(chunk_audio)
        sub_index = 0

        for i in range(0, audio_len, max_samples):
            sub_audio = chunk_audio[i : min(i + max_samples, audio_len)]
            sub_start = start_time + (i / 16000)
            sub_end = start_time + (min(i + max_samples, audio_len) / 16000)

            sub_chunks.append(
                (sub_audio, sub_start, sub_end, f"{chunk_index}.{sub_index}")
            )
            sub_index += 1

        return sub_chunks

    def _consumer_worker(self):
        """
        コンシューマースレッド: キューからチャンクを取得して処理
        """
        if self.config.verbose:
            print("Consumer thread started")

        inference_memory_usage = []

        while not self.stop_event.is_set() or not self.chunk_queue.empty():
            try:
                # キューからチャンクを取得
                chunk_data, arrival_time, buffer_size = self.chunk_queue.get(
                    timeout=0.1
                )
                chunk_audio, start_time, end_time, chunk_index = chunk_data

                # メトリクス作成
                chunk_metrics = ChunkMetrics(
                    chunk_index=chunk_index,
                    start_time=start_time,
                    end_time=end_time,
                    audio_duration=end_time - start_time,
                    arrival_time=arrival_time,
                    processing_start=time.perf_counter() - self.start_time,
                    processing_end=0,
                    buffer_size_at_arrival=buffer_size,
                )

                # メモリ測定
                mem_before = self.process.memory_info().rss / 1024 / 1024  # MB

                # WhisperPipeline処理
                if self.config.verbose:
                    print(f"\nConsumer: Processing chunk {chunk_index}")
                    print(f"  Time: {start_time:.1f}s - {end_time:.1f}s")
                    print("  Transcription: ", end="", flush=True)

                # ストリーミングコールバック
                transcribed_text = ""
                first_token_time = None
                token_count = 0

                def streamer_callback(text_chunk):
                    nonlocal transcribed_text, first_token_time, token_count
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - self.start_time
                    if self.config.verbose:
                        print(text_chunk, end="", flush=True)
                    transcribed_text += text_chunk
                    token_count += 1
                    return False  # 続行

                # 処理実行
                try:
                    # 30秒以下のチャンクはストリーミング可能
                    if chunk_metrics.audio_duration <= 30.0:
                        generate_kwargs = {
                            "num_beams": self.config.num_beams,
                            "streamer": streamer_callback,
                            "return_timestamps": False,
                        }
                        if self.config.language:
                            generate_kwargs["language"] = self.config.language

                        result = self.pipeline.generate(
                            (
                                chunk_audio.tolist()
                                if isinstance(chunk_audio, np.ndarray)
                                else chunk_audio
                            ),
                            **generate_kwargs,
                        )
                        transcription = str(result) if result else transcribed_text
                    else:
                        # 30秒超は通常処理
                        generate_kwargs = {"num_beams": self.config.num_beams}
                        if self.config.language:
                            generate_kwargs["language"] = self.config.language

                        result = self.pipeline.generate(chunk_audio, **generate_kwargs)
                        transcription = str(result)
                        if self.config.verbose:
                            print(transcription, end="")

                except Exception as e:
                    print(f"\nError processing chunk {chunk_index}: {e}")
                    transcription = ""

                if self.config.verbose:
                    print()  # 改行

                # メトリクス更新
                chunk_metrics.processing_end = time.perf_counter() - self.start_time
                chunk_metrics.transcription = transcription
                chunk_metrics.first_token_time = first_token_time
                chunk_metrics.token_count = token_count

                # メモリ測定
                mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
                inference_memory = mem_after - mem_before
                inference_memory_usage.append(inference_memory)
                self.peak_memory_mb = max(self.peak_memory_mb, mem_after)

                # メトリクスを追加
                self.metrics.add_chunk_metrics(chunk_metrics)

                # キューが空の場合はアンダーラン
                if self.chunk_queue.empty() and not self.stop_event.is_set():
                    self.metrics.buffer_underruns += 1
                    if self.config.verbose:
                        print("Consumer: Buffer underrun detected")

            except queue.Empty:
                continue

        # 平均推論メモリ使用量を計算
        if inference_memory_usage:
            self.metrics.avg_inference_memory_mb = sum(inference_memory_usage) / len(
                inference_memory_usage
            )

        if self.config.verbose:
            print("Consumer thread finished")

    def process_stream(self, audio_file: str) -> StreamingMetrics:
        """
        音声ファイルをストリーミング処理

        Args:
            audio_file: 音声ファイルパス

        Returns:
            ストリーミングメトリクス
        """
        # ストリームを作成
        if self.config.buffer_size > 0:
            audio_stream = AudioBufferStream(
                audio_file,
                chunk_duration=self.config.chunk_duration,
                sample_rate=16000,
                realtime=self.config.realtime,
                overlap=self.config.overlap,
                buffer_size=self.config.buffer_size,
            )
        else:
            audio_stream = AudioStream(
                audio_file,
                chunk_duration=self.config.chunk_duration,
                sample_rate=16000,
                realtime=self.config.realtime,
                overlap=self.config.overlap,
            )

        # ストリーム情報を表示
        stream_info = audio_stream.get_info()
        if self.config.verbose:
            print("\n=== Stream Information ===")
            for key, value in stream_info.items():
                print(f"  {key}: {value}")
            print()

        # ウォームアップ（最初のチャンクで）
        if self.config.verbose:
            print("Performing warm-up...")
        warmup_chunk = audio_stream.read_chunk()
        if warmup_chunk:
            chunk_audio, _, _, _ = warmup_chunk
            try:
                generate_kwargs = {"num_beams": self.config.num_beams}
                if self.config.language:
                    generate_kwargs["language"] = self.config.language
                _ = self.pipeline.generate(chunk_audio, **generate_kwargs)
            except Exception as e:
                print(f"Warm-up failed: {e}")
        audio_stream.reset()

        # 処理開始
        self.start_time = time.perf_counter()

        if self.config.verbose:
            print("\n=== Starting Streaming Processing ===\n")

        # スレッドを起動
        self.producer_thread = threading.Thread(
            target=self._producer_worker, args=(audio_stream,)
        )
        self.consumer_thread = threading.Thread(target=self._consumer_worker)

        self.producer_thread.start()
        self.consumer_thread.start()

        # スレッドの完了を待つ
        self.producer_thread.join()
        self.stop_event.set()  # プロデューサー完了後にストップシグナル
        self.consumer_thread.join()

        # メトリクスを計算
        self.metrics.model_memory_gb = self.model_memory_gb
        self.metrics.peak_memory_mb = self.peak_memory_mb
        self.metrics.calculate_aggregate_metrics()

        if self.config.verbose:
            print("\n=== Streaming Processing Complete ===")

        return self.metrics


def run_streaming_benchmark(
    model_path: str,
    audio_file: str,
    chunk_size_sec: float = 1.0,
    overlap_sec: float = 0.0,
    buffer_size: int = 5,
    num_beams: int = 1,
    device: str = "CPU",
    simulate_realtime: bool = True,
    verbose: bool = True,
    language: str = None,
) -> StreamingMetrics:
    """
    ストリーミングベンチマークを実行

    Args:
        model_path: Whisperモデルパス
        audio_file: 音声ファイルパス
        chunk_size_sec: チャンクサイズ（秒）
        overlap_sec: オーバーラップ（秒）
        buffer_size: バッファサイズ（チャンク数）
        num_beams: ビーム数
        device: デバイス
        simulate_realtime: リアルタイムシミュレーション
        verbose: 詳細出力
        language: 言語トークン（例：'<|ja|>', '<|en|>'）。Noneの場合は自動検出

    Returns:
        StreamingMetrics: ベンチマーク結果
    """
    config = StreamingConfig(
        chunk_duration=chunk_size_sec,
        overlap=overlap_sec,
        buffer_size=buffer_size,
        realtime=simulate_realtime,
        num_beams=num_beams,
        device=device,
        verbose=verbose,
        language=language,
    )

    processor = StreamingProcessor(model_path, config)
    return processor.process_stream(audio_file)
