# クイックスタートガイド - Embeddable版

## 最短で始める方法

### 1. 環境セットアップ（5-10分）
```batch
# 環境を自動構築
setup_portable.bat

# テスト実行（オプション）
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
├── python/                    # ポータブルPython環境
│   ├── python.exe            # Python実行ファイル
│   ├── main.py               # ベンチマークスクリプト
│   └── Lib/site-packages/    # インストール済みパッケージ
├── models/                   # Whisperモデル（要準備）
├── cache/                    # 音声ファイルキャッシュ
├── downloads/                # ダウンロードファイル
├── dist/                     # 配布パッケージ
├── run_benchmark.bat         # ベンチマーク実行
├── setup_portable.bat        # 環境セットアップ
├── create_distribution.bat   # 配布パッケージ作成
└── test_setup.bat           # 環境テスト
```

## よくある問題と解決方法

### Q: setup_portable.batでエラーが発生
**A:** インターネット接続を確認し、管理者権限で実行してください。

### Q: モデルが見つからない
**A:** `models`ディレクトリにエクスポートしたモデルを配置してください。

### Q: OpenVINOでエラー
**A:** モデルが`--disable-stateful`フラグでエクスポートされているか確認してください。

### Q: 配布先でうまく動かない
**A:** 配布先でも`run_benchmark.bat`を使用し、Pythonを直接実行しないでください。

## パフォーマンス最適化

- **CPU使用率**: `--iterations`を調整
- **メモリ使用量**: 小さなモデル（base, small）を使用  
- **実行速度**: `--num-beams 1`を使用（デフォルト）

## サポート

問題が発生した場合：
1. `test_setup.bat`で環境をテスト
2. ログファイルを確認
3. GitHubのissueを参照

ポータブル環境なので、問題がある場合は`python`フォルダを削除して再セットアップも可能です。