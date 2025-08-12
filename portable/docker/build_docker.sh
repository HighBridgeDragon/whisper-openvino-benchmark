#!/bin/bash
# Docker イメージビルドスクリプト

echo "=========================================================="
echo "Whisper Benchmark Docker Image Builder"
echo "=========================================================="
echo

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# プロジェクトルートを決定（配布先対応）
if [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    # 開発環境: プロジェクトルートに移動
    PROJECT_ROOT="$SCRIPT_DIR/../.."
    DOCKERFILE_PATH="portable/docker/Dockerfile"
else
    # 配布先: 同じディレクトリを使用
    PROJECT_ROOT="$SCRIPT_DIR"
    DOCKERFILE_PATH="Dockerfile"
fi

cd "$PROJECT_ROOT"

# タイムスタンプの生成
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
IMAGE_NAME="whisper-benchmark"
IMAGE_TAG="latest"

echo "Building Docker image..."
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo

# Docker イメージのビルド
docker build \
    -f ${DOCKERFILE_PATH} \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -t ${IMAGE_NAME}:${TIMESTAMP} \
    --progress=plain \
    .

if [ $? -eq 0 ]; then
    echo
    echo "=========================================================="
    echo "Docker image built successfully!"
    echo "=========================================================="
    echo
    echo "Image tags:"
    echo "  - ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  - ${IMAGE_NAME}:${TIMESTAMP}"
    echo
    echo "To run the benchmark:"
    echo "  cd portable/docker"
    echo "  ./run_docker.sh [options]"
    echo
else
    echo
    echo "=========================================================="
    echo "Docker image build failed!"
    echo "=========================================================="
    echo
    echo "Troubleshooting tips:"
    echo "1. Check Docker daemon is running: docker ps"
    echo "2. Ensure you have sufficient disk space"
    echo "3. Check Docker build permissions"
    echo "4. Review error messages above"
    exit 1
fi