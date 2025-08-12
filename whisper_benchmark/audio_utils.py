"""
音声ファイル処理ユーティリティ

音声ファイルの取得、ダウンロード、キャッシュ機能を提供
"""

import os
import requests


def get_audio_file(audio_file_path=None, cache_dir="cache"):
    """
    指定されたパスまたはデフォルトから音声ファイルを取得

    Args:
        audio_file_path: 音声ファイルのパス（オプション）
        cache_dir: キャッシュディレクトリ

    Returns:
        音声ファイルのパス
    """
    # 音声ファイルパスが指定されている場合の処理
    if audio_file_path:
        if os.path.exists(audio_file_path):
            print(f"Using specified audio file: {audio_file_path}")
            return audio_file_path
        else:
            print(f"Warning: Specified audio file not found: {audio_file_path}")
            print("Falling back to default audio file...")

    # デフォルト音声ファイルをダウンロード
    return download_audio_file(
        "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav",
        cache_dir,
    )


def download_audio_file(url, cache_dir="cache"):
    """
    キャッシュ機能付きで音声ファイルをダウンロード

    Args:
        url: ダウンロードURL
        cache_dir: キャッシュディレクトリ

    Returns:
        ダウンロードされた音声ファイルのパス
    """
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
