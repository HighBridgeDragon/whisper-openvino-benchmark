#!/bin/bash
# 完全な配布パッケージ作成スクリプト
# Docker イメージと必要なファイルを含む配布用アーカイブを作成

echo "=========================================================="
echo "Whisper Benchmark Distribution Package Creator"
echo "=========================================================="
echo

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# プロジェクトルートを決定（配布先対応）
if [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    # 開発環境: portable/dockerディレクトリを使用
    PROJECT_ROOT="$SCRIPT_DIR"
else
    # 配布先: 同じディレクトリを使用
    PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT"

# 設定
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="whisper-benchmark-docker"
DIST_DIR="./distribution"
PACKAGE_DIR="$DIST_DIR/${PACKAGE_NAME}-${TIMESTAMP}"
ARCHIVE_NAME="${PACKAGE_NAME}-${TIMESTAMP}.tar.gz"

echo "Creating distribution package: $ARCHIVE_NAME"
echo

# Docker イメージの存在確認
if ! docker images | grep -q "whisper-benchmark.*latest"; then
    echo "Error: Docker image 'whisper-benchmark:latest' not found."
    echo "Please run ./build_docker.sh first."
    exit 1
fi

# 配布ディレクトリの作成
echo "Setting up distribution directory..."
rm -rf "$DIST_DIR"
mkdir -p "$PACKAGE_DIR"

# 必要なディレクトリの作成
mkdir -p "$PACKAGE_DIR"/{models,cache,output,audio}

# Docker イメージのエクスポート
echo "Exporting Docker image..."
if ! docker save whisper-benchmark:latest -o "$PACKAGE_DIR/whisper-benchmark-image.tar"; then
    echo "Error: Failed to export Docker image."
    exit 1
fi

# イメージファイルの圧縮
echo "Compressing Docker image..."
gzip "$PACKAGE_DIR/whisper-benchmark-image.tar"

# 実行スクリプトのコピー
echo "Copying execution scripts..."
cp "$SCRIPT_DIR/run_docker.sh" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/run_interactive.sh" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/image_import.sh" "$PACKAGE_DIR/"

# Docker Compose ファイルのコピー（上級ユーザー用）
cp "$SCRIPT_DIR/docker-compose.yml" "$PACKAGE_DIR/"

# 配布用 README の作成
echo "Creating distribution README..."
cat > "$PACKAGE_DIR/README.md" << 'EOF'
# Whisper Benchmark Docker 配布パッケージ

OpenVINO GenAI を使用した Whisper 音声認識ベンチマークツールの Docker 版です。

## 前提条件

- Docker がインストールされていること
- 十分なディスク容量（約3GB以上）

## セットアップ手順

### 1. Docker イメージのインポート

```bash
./image_import.sh
```

このスクリプトが以下を自動実行します：
- Docker イメージのインポート
- 必要なディレクトリの作成
- 環境の検証

### 2. モデルの配置

Whisper モデルを `models/` ディレクトリに配置してください。

#### モデルのエクスポート方法（開発環境）

```bash
# 例：Whisper Large V3 Turbo のエクスポート
pip install optimum[openvino]
optimum-cli export openvino \
  -m openai/whisper-large-v3-turbo \
  --trust-remote-code \
  --weight-format fp32 \
  --disable-stateful \
  models/whisper-large-v3-turbo-stateless
```

### 3. ベンチマークの実行

#### 基本実行
```bash
./run_docker.sh
```

#### モデル指定実行
```bash
./run_docker.sh --model-path /app/models/whisper-large-v3-turbo-stateless
```

#### カスタム音声ファイル使用
```bash
# audio/ ディレクトリに音声ファイルを配置
./run_docker.sh --audio-file /app/audio/my_audio.wav
```

## ディレクトリ構造

```
whisper-benchmark-docker/
├── whisper-benchmark-image.tar.gz  # Docker イメージファイル
├── image_import.sh                 # イメージインポートスクリプト
├── run_docker.sh                   # ベンチマーク実行スクリプト
├── run_interactive.sh              # デバッグ用対話モード
├── docker-compose.yml              # 上級ユーザー用設定
├── README.md                       # このファイル
├── models/                         # Whisper モデル配置（要配置）
├── cache/                          # 音声ファイルキャッシュ
├── output/                         # ベンチマーク結果出力
└── audio/                          # カスタム音声ファイル
```

## 結果の確認

ベンチマーク結果は `output/benchmark_results.yaml` に保存されます。

## トラブルシューティング

### Docker が動作しない場合
```bash
# Docker デーモンの確認
docker ps

# Docker サービスの開始（Linux）
sudo systemctl start docker
```

### イメージのインポートが失敗する場合
```bash
# ディスク容量の確認
df -h

# Docker のクリーンアップ
docker system prune -a
```

### モデルが見つからない場合
```bash
# モデルディレクトリの確認
ls -la models/

# 対話モードでのデバッグ
./run_interactive.sh
```

## 詳細設定

上級ユーザーは `docker-compose.yml` を編集して：
- CPU スレッド数の調整
- メモリ制限の変更
- 環境変数の設定

などが可能です。

## サポート

問題が発生した場合：
1. `./run_interactive.sh` でコンテナ内の状況を確認
2. `docker logs <container_id>` でログを確認
3. `docker system prune` でシステムをクリーンアップ

---

Generated: $(date)
Package: whisper-benchmark-docker
Version: Docker Portable Edition
EOF

# ライセンスやその他のドキュメントのコピー（存在する場合）
if [ -f "LICENSE" ]; then
    cp "LICENSE" "$PACKAGE_DIR/"
fi

# バージョン情報ファイルの作成
cat > "$PACKAGE_DIR/VERSION" << EOF
Package: Whisper Benchmark Docker Distribution
Version: ${TIMESTAMP}
Build Date: $(date)
Docker Image: whisper-benchmark:latest
Python Version: 3.13
OpenVINO GenAI Version: 2025.2+
EOF

# 配布パッケージのアーカイブ作成
echo "Creating distribution archive..."
cd "$DIST_DIR"
tar -czf "$ARCHIVE_NAME" "${PACKAGE_NAME}-${TIMESTAMP}"

# ファイルサイズの確認
ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
echo

echo "=========================================================="
echo "Distribution package created successfully!"
echo "=========================================================="
echo
echo "Package: $DIST_DIR/$ARCHIVE_NAME"
echo "Size: $ARCHIVE_SIZE"
echo
echo "Contents:"
echo "  - Docker image (compressed)"
echo "  - Execution scripts"
echo "  - Configuration files"
echo "  - Documentation"
echo "  - Directory structure"
echo
echo "To distribute:"
echo "1. Copy $ARCHIVE_NAME to target system"
echo "2. Extract: tar -xzf $ARCHIVE_NAME"
echo "3. cd ${PACKAGE_NAME}-${TIMESTAMP}"
echo "4. Run: ./image_import.sh"
echo
echo "Distribution package is ready for deployment!"
echo