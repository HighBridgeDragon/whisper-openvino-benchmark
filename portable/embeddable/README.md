# Whisper Benchmark - Python Embeddable版

このディレクトリには、Python Embeddable + uv を使用したポータブルなWhisperベンチマーク環境の構築スクリプトが含まれています。

## 特徴

- **完全ポータブル**: Python環境の事前インストール不要
- **管理者権限不要**: レジストリやシステムファイルを変更しない
- **軽量**: 約150MBの配布パッケージ
- **高速**: uv使用による高速パッケージ管理
- **USBメモリ対応**: USBドライブから直接実行可能

## ファイル構成

```
embeddable/
├── setup_portable.bat      # ポータブル環境セットアップスクリプト
├── run_benchmark.bat       # ベンチマーク実行スクリプト
├── test_setup.bat         # 環境テストスクリプト
├── create_distribution.bat # 配布パッケージ作成スクリプト
├── cleanup.bat            # 仮想環境クリーンアップスクリプト
├── debug_setup.bat        # デバッグ用セットアップスクリプト
├── README.md              # このファイル
└── QUICKSTART.md          # クイックスタートガイド
```

## 使用方法

### 1. ポータブル環境の構築

```batch
setup_portable.bat
```

このスクリプトは以下を実行します：
- Python 3.13.2 Embeddable版をダウンロード
- uv パッケージマネージャーをインストール
- プロジェクトファイル（pyproject.toml、uv.lock）をコピー
- uv syncで.venv仮想環境を作成
- 必要な依存関係を仮想環境にインストール

### 2. ベンチマークの実行

```batch
REM 基本実行
run_benchmark.bat

REM オプション指定
run_benchmark.bat --model-path models\whisper-large-v3-turbo-stateless --iterations 10

REM ヘルプ表示
run_benchmark.bat --help
```

### 3. 配布パッケージの作成

```batch
create_distribution.bat
```

これにより `dist/whisper-benchmark-portable-YYYYMMDD.zip` が作成されます。

## 配布方法

1. `create_distribution.bat` を実行してzipファイルを作成
2. zipファイルを配布先に送付
3. 受信者は任意のディレクトリに展開
4. `run_benchmark.bat` を実行してベンチマーク開始

## システム要件

- Windows 10/11 x64
- 最低4GB RAM
- モデル用に2GB以上の空き容量
- インターネット接続（初回セットアップ時のみ）

## 利点

### PyInstaller onefileと比較して
- ✅ 複雑な依存関係問題が発生しにくい
- ✅ OpenVINO DLL依存関係の問題を回避
- ✅ ファイルサイズが小さい（150MB vs 500MB+）
- ✅ 起動時間が速い（展開不要）
- ✅ デバッグが容易
- ✅ 仮想環境による隔離されたパッケージ管理

### 従来のPython環境と比較して
- ✅ 環境の競合なし
- ✅ 管理者権限不要
- ✅ レジストリ変更なし
- ✅ アンインストールが簡単（フォルダ削除のみ）

## 注意事項

- 初回実行時はインターネット接続が必要
- モデルファイルは別途用意する必要がある
- Windows Defender等のセキュリティソフトが実行を阻止する場合がある

## トラブルシューティング

### 環境のテスト
環境が正しくセットアップされているか確認：
```batch
test_setup.bat
```

### 仮想環境の再作成
問題が発生した場合、仮想環境をクリーンアップして再作成：
```batch
cleanup.bat
setup_portable.bat
```

### デバッグモード
詳細なデバッグ情報を表示：
```batch
debug_setup.bat
```

### PowerShell実行ポリシーエラー
```batch
powershell -ExecutionPolicy Bypass -File setup_portable.bat
```

### ダウンロードエラー
- インターネット接続を確認
- プロキシ設定を確認
- 手動でファイルをダウンロードして`downloads/`ディレクトリに配置

### 依存関係インストールエラー
- `.venv`ディレクトリを削除して再実行
- uvのバージョンを変更（setup_portable.bat内のUV_VERSION）

### "... の使い方が誤っています"エラー
- Windowsバッチファイルでechoコマンドに`...`を含めるとエラーになる場合がある
- 最新版では修正済み

## モデルの準備

Whisperモデルを使用するには、開発環境で以下を実行：

```bash
# statelessモード（推奨）
uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo \
    --trust-remote-code --weight-format fp32 --disable-stateful \
    models\whisper-large-v3-turbo-stateless

# エクスポート後、modelsディレクトリをポータブル環境にコピー
```

## パフォーマンス最適化

- CPUコア数に応じて`--iterations`を調整
- 大きなモデルでは`--num-beams 1`を推奨
- メモリ不足の場合は小さなモデルを使用

## 技術詳細

### 仮想環境について
- uv syncにより`.venv`ディレクトリに仮想環境が作成される
- すべての依存関係は`.venv`内に隔離される
- embeddable Python本体は`python/`ディレクトリに保持
- uvはembeddable Python側で管理、パッケージは仮想環境側で管理

### ディレクトリ構造
```
embeddable/
├── python/                 # Python Embeddable環境
│   ├── python.exe
│   ├── Scripts/
│   │   └── uv.exe         # uvパッケージマネージャー
│   └── ...
├── .venv/                  # 仮想環境（uv syncで作成）
│   ├── Scripts/
│   │   └── python.exe     # 仮想環境のPython
│   └── Lib/site-packages/ # インストールされたパッケージ
├── models/                 # Whisperモデル配置場所
└── *.bat                   # 各種スクリプト
```