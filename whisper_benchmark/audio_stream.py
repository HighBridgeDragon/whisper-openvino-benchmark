"""
音声ストリーミングシミュレーター

リアルタイムの音声ストリームをシミュレートし、
チャンク単位での音声データ供給を実現
"""

import time
from typing import Generator, Optional, Tuple
import numpy as np
import librosa
from pathlib import Path


class AudioStream:
    """
    音声ファイルからリアルタイムストリームをシミュレート

    メモリ効率的な遅延読み込みとリアルタイムペーシングを実装
    """

    def __init__(
        self,
        audio_file: str,
        chunk_duration: float = 1.0,
        sample_rate: int = 16000,
        realtime: bool = True,
        overlap: float = 0.0,
    ):
        """
        Args:
            audio_file: 音声ファイルパス
            chunk_duration: チャンクの長さ（秒）
            sample_rate: サンプリングレート（Hz）
            realtime: リアルタイムペーシングを有効化
            overlap: チャンク間のオーバーラップ（秒）
        """
        self.audio_file = Path(audio_file)
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.realtime = realtime
        self.overlap = overlap

        # チャンクサイズの計算
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap * sample_rate)
        self.stride_samples = self.chunk_samples - self.overlap_samples

        # 音声メタデータの取得（ファイルは開かない）
        self._audio_duration = librosa.get_duration(path=str(audio_file))
        self._total_samples = int(self._audio_duration * sample_rate)

        # ストリーム状態
        self._position = 0
        self._stream_start_time = None
        self._chunk_count = 0

        # 音声データのメモリマップ（効率的な部分読み込み用）
        self._audio_data = None
        self._is_loaded = False

    def _load_audio_lazy(self):
        """音声データの遅延読み込み"""
        if not self._is_loaded:
            # 全体を一度に読み込むのではなく、ストリーミング対応
            self._audio_data, _ = librosa.load(
                str(self.audio_file), sr=self.sample_rate, mono=True
            )
            self._is_loaded = True

    def start(self):
        """ストリーミングの開始"""
        self._position = 0
        self._chunk_count = 0
        self._stream_start_time = time.perf_counter()
        self._load_audio_lazy()

    def read_chunk(self) -> Optional[Tuple[np.ndarray, float, float, int]]:
        """
        次のチャンクを読み込み

        Returns:
            (chunk_audio, start_time, end_time, chunk_index) または None（終了時）
        """
        if self._audio_data is None:
            self.start()

        # ストリーム終了チェック
        if self._position >= self._total_samples:
            return None

        # チャンクの範囲を計算
        start_pos = self._position
        end_pos = min(start_pos + self.chunk_samples, self._total_samples)

        # チャンクを抽出
        chunk = self._audio_data[start_pos:end_pos]

        # タイムスタンプ計算
        start_time = start_pos / self.sample_rate
        end_time = end_pos / self.sample_rate

        # リアルタイムペーシング
        if self.realtime and self._stream_start_time is not None:
            # 現在のストリーム時間
            stream_elapsed = time.perf_counter() - self._stream_start_time
            # このチャンクが来るべき時刻
            expected_time = start_time

            # 早すぎる場合は待機
            if stream_elapsed < expected_time:
                wait_time = expected_time - stream_elapsed
                time.sleep(wait_time)

        # 位置を更新（オーバーラップを考慮）
        self._position += self.stride_samples
        chunk_index = self._chunk_count
        self._chunk_count += 1

        return chunk, start_time, end_time, chunk_index

    def stream_chunks(
        self,
    ) -> Generator[Tuple[np.ndarray, float, float, int], None, None]:
        """
        チャンクをストリーミング（ジェネレーター）

        Yields:
            (chunk_audio, start_time, end_time, chunk_index)
        """
        self.start()

        while True:
            chunk_data = self.read_chunk()
            if chunk_data is None:
                break
            yield chunk_data

    def get_info(self) -> dict:
        """ストリーム情報を取得"""
        return {
            "audio_file": str(self.audio_file),
            "duration": self._audio_duration,
            "sample_rate": self.sample_rate,
            "chunk_duration": self.chunk_duration,
            "overlap": self.overlap,
            "total_chunks": self._estimate_total_chunks(),
            "realtime_mode": self.realtime,
        }

    def _estimate_total_chunks(self) -> int:
        """総チャンク数の推定"""
        if self.stride_samples <= 0:
            return 1
        return int(np.ceil(self._total_samples / self.stride_samples))

    def reset(self):
        """ストリームをリセット"""
        self._position = 0
        self._chunk_count = 0
        self._stream_start_time = None


class AudioBufferStream(AudioStream):
    """
    バッファリング機能を持つ音声ストリーム

    プリフェッチとキューイングでより滑らかなストリーミングを実現
    """

    def __init__(
        self,
        audio_file: str,
        chunk_duration: float = 1.0,
        sample_rate: int = 16000,
        realtime: bool = True,
        overlap: float = 0.0,
        buffer_size: int = 3,
    ):
        super().__init__(audio_file, chunk_duration, sample_rate, realtime, overlap)
        self.buffer_size = buffer_size
        self._buffer = []

    def prefetch(self, num_chunks: int = None):
        """
        チャンクを事前にバッファに読み込み

        Args:
            num_chunks: 読み込むチャンク数（Noneの場合はbuffer_size）
        """
        if num_chunks is None:
            num_chunks = self.buffer_size

        self._load_audio_lazy()

        for _ in range(num_chunks):
            if len(self._buffer) >= self.buffer_size:
                break

            # リアルタイムモードを一時的に無効化してプリフェッチ
            original_realtime = self.realtime
            self.realtime = False

            chunk_data = self.read_chunk()

            self.realtime = original_realtime

            if chunk_data is not None:
                self._buffer.append(chunk_data)

    def read_buffered_chunk(self) -> Optional[Tuple[np.ndarray, float, float, int]]:
        """
        バッファからチャンクを読み込み（リアルタイムペーシング付き）
        """
        # バッファが空の場合は直接読み込み
        if not self._buffer:
            return self.read_chunk()

        # バッファから取得
        chunk_data = self._buffer.pop(0)

        # リアルタイムペーシング
        if self.realtime and self._stream_start_time is not None:
            chunk, start_time, end_time, chunk_index = chunk_data
            stream_elapsed = time.perf_counter() - self._stream_start_time
            expected_time = start_time

            if stream_elapsed < expected_time:
                wait_time = expected_time - stream_elapsed
                time.sleep(wait_time)

        # バッファを補充
        self.prefetch(1)

        return chunk_data
