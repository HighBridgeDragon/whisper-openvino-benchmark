# クイックスタートガイド - Embeddable版

## 最短で始める方法

### 1. 環境セットアップ（5-10分）
```batch
# 環境を自動構築（Python Embeddable + uv + .venv仮想環境）
setup_portable.bat

# 環境テスト（推奨）
test_setup.bat
```

### 2. モデル準備
開発環境で以下を実行してモデルをエクスポート：
```bash
uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo \
    --trust-remote-code --weight-format fp32 --disable-stateful \
    models\whisper-large-v3-turbo-stateless
```

### 3. ベンチマーク実行
```batch
# 基本実行（モデルを自動検出）
run_benchmark.bat

# モデル指定実行
run_benchmark.bat --model-path models\whisper-large-v3-turbo-stateless
```

### 4. 配布パッケージ作成
```batch
# 配布用zipファイル作成
create_distribution.bat
```

## ファイル構成（セットアップ後）

```
embeddable/
├── python/                    # Python Embeddable環境
│   ├── python.exe            # Python実行ファイル
│   ├── Scripts/
│   │   └── uv.exe           # uvパッケージマネージャー
│   └── Lib/site-packages/    # uv自体のパッケージ
├── .venv/                     # 仮想環境（uv syncで作成）
│   ├── Scripts/
│   │   └── python.exe       # 仮想環境のPython
│   └── Lib/site-packages/    # インストール済みパッケージ
├── models/                   # Whisperモデル（要準備）
├── cache/                    # 音声ファイルキャッシュ
├── downloads/                # ダウンロードファイル
├── dist/                     # 配布パッケージ
├── main.py                   # ベンチマークスクリプト
├── pyproject.toml            # プロジェクト設定
├── uv.lock                   # 依存関係ロックファイル
├── run_benchmark.bat         # ベンチマーク実行
├── setup_portable.bat        # 環境セットアップ
├── test_setup.bat           # 環境テスト
├── cleanup.bat              # 仮想環境クリーンアップ
├── debug_setup.bat          # デバッグ用スクリプト
└── create_distribution.bat   # 配布パッケージ作成
```

## よくある問題と解決方法

### Q: setup_portable.batでエラーが発生
**A:** 
1. 「... の使い方が誤っています」エラーの場合は、最新版に更新してください
2. インターネット接続を確認
3. `cleanup.bat`で環境をクリーンアップして再実行

### Q: モデルが見つからない
**A:** `models`ディレクトリにエクスポートしたモデルを配置してください。

### Q: OpenVINOでエラー
**A:** モデルが`--disable-stateful`フラグでエクスポートされているか確認してください。

### Q: 配布先でうまく動かない
**A:** 
1. 配布先でも`run_benchmark.bat`を使用（Pythonを直接実行しない）
2. Windows Defenderの例外設定を確認
3. 配布先で`test_setup.bat`を実行して環境を確認

## パフォーマンス最適化

- **CPU使用率**: `--iterations`を調整
- **メモリ使用量**: 小さなモデル（base, small）を使用  
- **実行速度**: `--num-beams 1`を使用（デフォルト）

## サポート

問題が発生した場合：
1. `test_setup.bat`で環境をテスト
2. ログファイルを確認
3. GitHubのissueを参照

ポータブル環境なので、問題がある場合は：
1. `cleanup.bat`で仮想環境をクリーンアップ
2. `python`フォルダを削除して完全に再セットアップ
3. `debug_setup.bat`で問題を特定

## 仮想環境について

このポータブル版は、Python Embeddableとuvを組み合わせ、`.venv`仮想環境を作成して依存関係を管理します。これにより、システムのPython環境に影響を与えずに、完全に隔離された環境でベンチマークを実行できます。