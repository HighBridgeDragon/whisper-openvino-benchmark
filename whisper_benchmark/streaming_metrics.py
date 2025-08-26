"""
ストリーミングベンチマーク用メトリクス

真のストリーミング処理に特化した性能指標の計算と管理
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from statistics import mean, stdev, median


@dataclass
class ChunkMetrics:
    """単一チャンクの処理メトリクス"""

    chunk_index: int
    start_time: float  # チャンクの音声開始時刻
    end_time: float  # チャンクの音声終了時刻
    audio_duration: float  # チャンクの音声長さ

    # タイミング関連
    arrival_time: float  # チャンクがシステムに到着した時刻（相対）
    processing_start: float  # 処理開始時刻（相対）
    processing_end: float  # 処理終了時刻（相対）
    processing_duration: float = 0.0  # 処理にかかった時間

    # 出力関連
    first_token_time: Optional[float] = None  # 最初のトークン出力時刻
    transcription: str = ""
    token_count: int = 0

    # パフォーマンス指標
    rtf: float = 0.0  # リアルタイムファクター（処理時間/音声時間）
    latency: float = 0.0  # レイテンシ（到着から処理開始まで）

    # バッファ状態
    buffer_size_at_arrival: int = 0  # 到着時のバッファサイズ

    def calculate_metrics(self):
        """メトリクスを計算"""
        self.processing_duration = self.processing_end - self.processing_start
        self.rtf = (
            self.processing_duration / self.audio_duration
            if self.audio_duration > 0
            else 0
        )
        self.latency = self.processing_start - self.arrival_time


@dataclass
class StreamingMetrics:
    """ストリーミングベンチマーク全体のメトリクス"""

    # 基本情報
    total_audio_duration: float = 0.0
    total_processing_time: float = 0.0
    total_chunks: int = 0

    # チャンクメトリクス
    chunk_metrics: List[ChunkMetrics] = field(default_factory=list)

    # レイテンシ関連
    first_chunk_latency: float = 0.0  # 最初のチャンクの処理開始レイテンシ
    first_token_latency: float = 0.0  # 最初のトークン出力までの時間
    avg_latency: float = 0.0
    max_latency: float = 0.0
    min_latency: float = float("inf")

    # 処理時間関連
    avg_processing_time: float = 0.0
    std_processing_time: float = 0.0
    min_processing_time: float = float("inf")
    max_processing_time: float = 0.0
    median_processing_time: float = 0.0

    # RTF（リアルタイムファクター）
    overall_rtf: float = 0.0
    avg_chunk_rtf: float = 0.0
    max_chunk_rtf: float = 0.0
    min_chunk_rtf: float = float("inf")
    rtf_consistency: float = 0.0  # RTFの標準偏差

    # スループット
    throughput_audio_per_sec: float = 0.0
    throughput_stability: float = 0.0  # スループットの一貫性（CV）

    # ストリーミング品質
    buffer_underruns: int = 0  # バッファアンダーラン回数
    buffer_overruns: int = 0  # バッファオーバーラン回数
    max_buffer_size: int = 0  # 最大バッファサイズ
    avg_buffer_size: float = 0.0

    # ドリフト（リアルタイムからのずれ）
    cumulative_drift: float = 0.0  # 累積ドリフト
    max_drift: float = 0.0
    drift_recovery_points: int = 0  # ドリフトが回復した回数

    # メモリ使用量
    model_memory_gb: float = 0.0
    avg_inference_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # 転写結果
    full_transcription: str = ""
    total_tokens: int = 0

    def calculate_aggregate_metrics(self):
        """集約メトリクスを計算"""
        if not self.chunk_metrics:
            return

        # 基本統計
        self.total_chunks = len(self.chunk_metrics)
        self.total_audio_duration = sum(cm.audio_duration for cm in self.chunk_metrics)
        self.total_processing_time = sum(
            cm.processing_duration for cm in self.chunk_metrics
        )

        # レイテンシ統計
        latencies = [cm.latency for cm in self.chunk_metrics if cm.latency > 0]
        if latencies:
            self.avg_latency = mean(latencies)
            self.min_latency = min(latencies)
            self.max_latency = max(latencies)
            self.first_chunk_latency = (
                self.chunk_metrics[0].latency if self.chunk_metrics else 0
            )

        # 最初のトークンレイテンシ
        for cm in self.chunk_metrics:
            if cm.first_token_time is not None:
                self.first_token_latency = cm.first_token_time - cm.arrival_time
                break

        # 処理時間統計
        processing_times = [cm.processing_duration for cm in self.chunk_metrics]
        if processing_times:
            self.avg_processing_time = mean(processing_times)
            self.min_processing_time = min(processing_times)
            self.max_processing_time = max(processing_times)
            self.median_processing_time = median(processing_times)
            if len(processing_times) > 1:
                self.std_processing_time = stdev(processing_times)

        # RTF統計
        rtfs = [cm.rtf for cm in self.chunk_metrics if cm.rtf > 0]
        if rtfs:
            self.avg_chunk_rtf = mean(rtfs)
            self.min_chunk_rtf = min(rtfs)
            self.max_chunk_rtf = max(rtfs)
            if len(rtfs) > 1:
                self.rtf_consistency = stdev(rtfs)

        # 全体RTF
        if self.total_audio_duration > 0:
            self.overall_rtf = self.total_processing_time / self.total_audio_duration
            self.throughput_audio_per_sec = (
                self.total_audio_duration / self.total_processing_time
            )

        # スループット安定性（変動係数）
        if self.avg_processing_time > 0:
            self.throughput_stability = (
                self.std_processing_time / self.avg_processing_time
            )

        # バッファ統計
        buffer_sizes = [cm.buffer_size_at_arrival for cm in self.chunk_metrics]
        if buffer_sizes:
            self.avg_buffer_size = mean(buffer_sizes)
            self.max_buffer_size = max(buffer_sizes)

        # ドリフト計算
        self._calculate_drift()

        # 転写結果の結合
        self.full_transcription = " ".join(
            cm.transcription for cm in self.chunk_metrics if cm.transcription
        )
        self.total_tokens = sum(cm.token_count for cm in self.chunk_metrics)

    def _calculate_drift(self):
        """リアルタイムからのドリフトを計算"""
        if not self.chunk_metrics:
            return

        cumulative_drift = 0.0
        max_drift = 0.0
        recovery_points = 0
        was_drifting = False

        for i, cm in enumerate(self.chunk_metrics):
            # 期待される処理完了時刻
            expected_completion = cm.arrival_time + cm.audio_duration
            # 実際の処理完了時刻
            actual_completion = cm.processing_end

            # ドリフト（遅延が正、先行が負）
            drift = actual_completion - expected_completion
            cumulative_drift += drift

            max_drift = max(max_drift, abs(cumulative_drift))

            # ドリフト回復の検出
            if was_drifting and abs(cumulative_drift) < 0.1:  # 0.1秒以内に回復
                recovery_points += 1
                was_drifting = False
            elif abs(cumulative_drift) > 1.0:  # 1秒以上のドリフト
                was_drifting = True

        self.cumulative_drift = cumulative_drift
        self.max_drift = max_drift
        self.drift_recovery_points = recovery_points

    def add_chunk_metrics(self, chunk_metrics: ChunkMetrics):
        """チャンクメトリクスを追加"""
        chunk_metrics.calculate_metrics()
        self.chunk_metrics.append(chunk_metrics)

    def get_summary(self) -> Dict[str, Any]:
        """サマリー情報を辞書形式で取得"""
        self.calculate_aggregate_metrics()

        return {
            "overview": {
                "total_audio_duration": self.total_audio_duration,
                "total_processing_time": self.total_processing_time,
                "total_chunks": self.total_chunks,
                "overall_rtf": self.overall_rtf,
                "throughput_audio_per_sec": self.throughput_audio_per_sec,
            },
            "latency": {
                "first_chunk_latency": self.first_chunk_latency,
                "first_token_latency": self.first_token_latency,
                "avg_latency": self.avg_latency,
                "min_latency": self.min_latency,
                "max_latency": self.max_latency,
            },
            "processing": {
                "avg_processing_time": self.avg_processing_time,
                "std_processing_time": self.std_processing_time,
                "min_processing_time": self.min_processing_time,
                "max_processing_time": self.max_processing_time,
                "median_processing_time": self.median_processing_time,
            },
            "rtf": {
                "overall_rtf": self.overall_rtf,
                "avg_chunk_rtf": self.avg_chunk_rtf,
                "min_chunk_rtf": self.min_chunk_rtf,
                "max_chunk_rtf": self.max_chunk_rtf,
                "rtf_consistency": self.rtf_consistency,
            },
            "streaming_quality": {
                "buffer_underruns": self.buffer_underruns,
                "buffer_overruns": self.buffer_overruns,
                "avg_buffer_size": self.avg_buffer_size,
                "max_buffer_size": self.max_buffer_size,
                "throughput_stability": self.throughput_stability,
            },
            "drift": {
                "cumulative_drift": self.cumulative_drift,
                "max_drift": self.max_drift,
                "drift_recovery_points": self.drift_recovery_points,
            },
            "memory": {
                "model_memory_gb": self.model_memory_gb,
                "avg_inference_memory_mb": self.avg_inference_memory_mb,
                "peak_memory_mb": self.peak_memory_mb,
            },
            "transcription": {
                "full_transcription": self.full_transcription,
                "total_tokens": self.total_tokens,
            },
        }

    def is_realtime_capable(self) -> bool:
        """リアルタイム処理が可能かどうかを判定"""
        return self.overall_rtf <= 1.0 and self.max_chunk_rtf <= 1.5
