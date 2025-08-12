#!/bin/bash
# Docker イメージインポートスクリプト
# 配布されたDocker イメージファイルをインポート

echo "=========================================================="
echo "Whisper Benchmark Docker Image Import Tool"
echo "=========================================================="
echo

# Docker デーモンの確認
if ! docker ps &> /dev/null; then
    echo "Error: Docker daemon is not running or not accessible."
    echo "Please start Docker and try again."
    exit 1
fi

# 現在のディレクトリでイメージファイルを検索
IMAGE_FILES=$(ls whisper-benchmark-image.tar* 2>/dev/null | head -1)

if [ -z "$IMAGE_FILES" ]; then
    echo "No image file found in current directory."
    echo
    echo "Looking for files matching:"
    echo "  - whisper-benchmark-image.tar"
    echo "  - whisper-benchmark-image.tar.gz"
    echo
    echo "Please ensure the Docker image file is in this directory."
    echo "You can export an image using: ./image_export.sh"
    exit 1
fi

IMAGE_FILE="$IMAGE_FILES"
echo "Found image file: $IMAGE_FILE"

# ファイルサイズの表示
FILE_SIZE=$(du -h "$IMAGE_FILE" | cut -f1)
echo "File size: $FILE_SIZE"

# 既存イメージの確認
if docker images | grep -q "whisper-benchmark.*latest"; then
    echo
    echo "Warning: whisper-benchmark:latest already exists."
    docker images | grep "whisper-benchmark"
    echo
    read -p "Replace existing image? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Import cancelled."
        exit 0
    fi
fi

echo
echo "Importing Docker image..."
echo "This may take a few minutes..."

# 圧縮ファイルの場合は展開しながらインポート
if [[ "$IMAGE_FILE" == *.gz ]]; then
    echo "Decompressing and importing..."
    if gunzip -c "$IMAGE_FILE" | docker load; then
        echo "✓ Docker image imported successfully!"
    else
        echo "✗ Failed to import Docker image!"
        exit 1
    fi
else
    if docker load -i "$IMAGE_FILE"; then
        echo "✓ Docker image imported successfully!"
    else
        echo "✗ Failed to import Docker image!"
        exit 1
    fi
fi

echo
echo "Verifying imported image..."
if docker images | grep -q "whisper-benchmark.*latest"; then
    echo "✓ Image verification successful!"
    echo
    echo "Available images:"
    docker images | grep whisper-benchmark
else
    echo "✗ Image verification failed!"
    exit 1
fi

# 必要なディレクトリの作成
echo
echo "Setting up working directories..."
mkdir -p models cache output audio
echo "✓ Directories created: models/, cache/, output/, audio/"

echo
echo "=========================================================="
echo "Docker image import completed!"
echo "=========================================================="
echo
echo "Next steps:"
echo "1. Place your Whisper model in the models/ directory"
echo "2. Run benchmark: ./run_docker.sh"
echo "3. For help: ./run_docker.sh --help"
echo
echo "Example model directory structure:"
echo "  models/whisper-large-v3-turbo-stateless/"
echo "    ├── config.json"
echo "    ├── openvino_encoder_model.xml"
echo "    ├── openvino_decoder_model.xml"
echo "    └── ... (other model files)"
echo
echo "To test the installation:"
echo "  ./run_interactive.sh"
echo