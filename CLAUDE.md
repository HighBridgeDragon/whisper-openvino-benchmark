# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このプロジェクトは、OpenVINO GenAIのWhisperPipelineを使用したベンチマークツールです。音声認識モデルのパフォーマンスを測定し、推論時間、メモリ使用量、リアルタイムファクター（RTF）などの指標を収集します。

## ビルド・実行コマンド

### 開発環境のセットアップ
```bash
# 依存関係のインストール（UVを使用）
uv pip sync

# 開発用依存関係のインストール
uv pip install --group dev
```

### Whisperモデルのダウンロード

#### 標準エクスポート（推奨）
```bash
# OpenVINO形式でWhisperモデルをエクスポート（statelessモード）
uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo --trust-remote-code --weight-format fp32 --disable-stateful .\models\openai\whisper-large-v3-turbo-stateless\
```

#### 旧形式（非推奨、OpenVINO GenAI 2025.2でトークナイザー問題あり）
```bash
# 従来のエクスポート方法（問題が発生する可能性があります）
uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo --trust-remote-code --weight-format fp32 .\models\openai\whisper-large-v3-turbo\
```

### 実行可能ファイルのビルド

#### 標準ビルド（複数ファイル版）
```bash
# Windowsバッチファイルを実行
build.bat

# または手動でPyInstallerを実行
uv run pyinstaller benchmark.spec --clean
```

#### 単一実行ファイル版（サイズは大きいが配布が簡単）
```bash
# 単一ファイル版のビルドスクリプト
build_onefile.bat

# または手動で実行
uv run pyinstaller main.py --onefile --name whisper-benchmark-onefile --clean
```

#### 最小構成版（最適化されたビルド）
```bash
# 最小構成のspecファイルを使用
uv run pyinstaller benchmark_minimal.spec --clean
```

### コードの実行
```bash
# 直接実行
python main.py --model-path "path/to/whisper/model" --iterations 5

# テストスクリプト実行
python test.py
```

### リンター実行
```bash
# Ruffを使用してコードをチェック
ruff check .

# 自動修正を適用
ruff check --fix .
```

## アーキテクチャ概要

### 主要ファイルの構成

- **main.py**: ベンチマークツールのメインエントリーポイント
  - `get_cpu_info()`: Windows WMIを使用してCPU情報を取得
  - `download_audio_file()`: テスト用音声ファイルをキャッシュ付きでダウンロード
  - `run_benchmark()`: WhisperPipelineを使用してベンチマークを実行
  - コマンドライン引数の処理と結果の表示

- **test.py**: 簡単なテストスクリプト
  - WhisperPipelineの基本的な動作確認用
  - num_beams=2以上でエラーが発生する問題のテスト用

- **build.bat**: PyInstallerを使用した実行可能ファイルビルド用スクリプト
  - UV（パッケージマネージャー）の存在確認
  - 依存関係のインストール
  - PyInstallerによるビルド実行

- **benchmark.spec**: PyInstallerの設定ファイル
  - OpenVINOとOpenVINO GenAIのDLLとプラグインを含める設定
  - librosaやscipy関連の隠れた依存関係の指定

### 主要な依存関係

- **openvino-genai**: WhisperPipelineを提供するOpenVINOの音声認識ライブラリ
- **librosa**: 音声ファイルの読み込みと処理
- **psutil**: CPU情報とメモリ使用量の取得
- **tabulate**: ベンチマーク結果の表形式表示

### 技術的な注意点

1. Windows環境に特化した実装（WMIを使用したCPU情報取得）
2. 音声ファイルは16kHzにリサンプリングされる
3. デフォルトでは英語（`<|en|>`）の認識を行う
4. CPUデバイスでの実行がデフォルト
5. キャッシュディレクトリに音声ファイルを保存して再利用

## トラブルシューティング

### OpenVINO GenAI 2025.2でのトークナイザーエラー

**症状**: `Unable to read the model: openvino_tokenizer.xml` エラーが発生し、"Available frontends: jax pytorch" と表示される。

**原因**: OpenVINO GenAI 2025.2ではstatefulモードでエクスポートされたモデルのトークナイザー読み込みに問題がある。

**解決方法**:
1. `--disable-stateful` フラグを使用してモデルを再エクスポートする
2. statelessモードでエクスポートされたモデルを使用する

```bash
# 問題のあるモデルを再エクスポート
uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo --trust-remote-code --weight-format fp32 --disable-stateful .\models\openai\whisper-large-v3-turbo-stateless\
```

### 依存関係の問題

**症状**: `ModuleNotFoundError: No module named 'librosa'` などの依存関係エラー

**解決方法**:
```bash
# UV環境で実行する
uv run python main.py --model-path "path/to/model" --iterations 5

# または依存関係を再インストール
uv pip sync
```

### モデルファイルの検証

モデルディレクトリに以下のファイルが存在することを確認してください：
- `openvino_encoder_model.xml` / `.bin`
- `openvino_decoder_model.xml` / `.bin`
- `openvino_tokenizer.xml` / `.bin`
- `openvino_detokenizer.xml` / `.bin`
- `config.json`
- `generation_config.json`

### パフォーマンス最適化

1. **量子化モデルの使用**: INT8量子化により推論速度を向上
2. **CPUコア数の調整**: OpenVINOの並列処理設定
3. **メモリ最適化**: 大きなモデルでのメモリ使用量管理