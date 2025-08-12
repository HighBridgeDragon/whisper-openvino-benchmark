#!/bin/bash
# Docker イメージエクスポートスクリプト
# 配布用の単一ファイルとしてDocker イメージを保存

echo "=========================================================="
echo "Whisper Benchmark Docker Image Export Tool"
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
IMAGE_NAME="whisper-benchmark"
IMAGE_TAG="latest"
EXPORT_DIR="./docker-export"
IMAGE_FILE="$EXPORT_DIR/whisper-benchmark-image.tar"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Docker イメージの存在確認
echo "Checking Docker image..."
if ! docker images | grep -q "${IMAGE_NAME}.*${IMAGE_TAG}"; then
    echo "Error: Docker image '${IMAGE_NAME}:${IMAGE_TAG}' not found."
    echo "Please run ./build_docker.sh first."
    exit 1
fi

# イメージサイズの確認
IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${IMAGE_TAG} --format "table {{.Size}}" | tail -n 1)
echo "Image size: $IMAGE_SIZE"

# エクスポートディレクトリの作成
echo "Creating export directory..."
mkdir -p "$EXPORT_DIR"

# 既存のエクスポートファイルの確認
if [ -f "$IMAGE_FILE" ]; then
    echo "Warning: Export file already exists: $IMAGE_FILE"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Export cancelled."
        exit 0
    fi
    rm -f "$IMAGE_FILE"
fi

# Docker イメージのエクスポート
echo "Exporting Docker image..."
echo "This may take several minutes depending on image size..."
echo

if docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "$IMAGE_FILE"; then
    echo "✓ Docker image exported successfully!"
else
    echo "✗ Failed to export Docker image!"
    exit 1
fi

# エクスポートファイルのサイズ確認
EXPORT_SIZE=$(du -h "$IMAGE_FILE" | cut -f1)
echo "Export file size: $EXPORT_SIZE"

# 必須の圧縮処理
echo ""
echo "Compressing image file..."
gzip "$IMAGE_FILE"
IMAGE_FILE="${IMAGE_FILE}.gz"
COMPRESSED_SIZE=$(du -h "$IMAGE_FILE" | cut -f1)
echo "✓ Compressed size: $COMPRESSED_SIZE"

echo ""
echo "Note: For complete distribution package including scripts and documentation,"
echo "use ./create_distribution.sh instead."

echo
echo "=========================================================="
echo "Docker image export completed!"
echo "=========================================================="
echo
echo "Export file: $IMAGE_FILE"
echo "File size: $(du -h "$IMAGE_FILE" | cut -f1)"
echo
echo "To distribute this image:"
echo "1. Copy the export file to the target system"
echo "2. Run: docker load -i $(basename "$IMAGE_FILE")"
echo "3. Verify: docker images | grep whisper-benchmark"
echo
echo "For automated import, use the image_import.sh script."
echo
echo "To create a complete distribution package:"
echo "  ./create_distribution.sh"
echo