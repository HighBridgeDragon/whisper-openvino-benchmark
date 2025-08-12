#!/bin/bash
# Docker インタラクティブモード実行スクリプト

echo "=========================================================="
echo "Whisper Benchmark Docker Interactive Mode"
echo "=========================================================="
echo

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# プロジェクトルートを決定（配布先対応）
if [ -f "$SCRIPT_DIR/../../pyproject.toml" ]; then
    # 開発環境: プロジェクトルートに移動
    PROJECT_ROOT="$SCRIPT_DIR/../.."
else
    # 配布先: 同じディレクトリを使用
    PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT"

# 必要なディレクトリの作成
echo "Checking required directories..."
mkdir -p models cache output audio

# Docker イメージの存在確認
if ! docker images | grep -q "whisper-benchmark.*latest"; then
    echo "Error: Docker image 'whisper-benchmark:latest' not found."
    echo "Please run ./build_docker.sh first."
    exit 1
fi

echo "Starting interactive shell..."
echo
echo "Volume mappings:"
echo "  ./models  -> /app/models   (read-only)"
echo "  ./cache   -> /app/cache"
echo "  ./output  -> /app/output"
echo "  ./audio   -> /app/audio    (read-only)"
echo
echo "To run benchmark inside container:"
echo "  uv run python main.py --model-path /app/models/your-model [options]"
echo
echo "To exit: type 'exit' or press Ctrl+D"
echo "=========================================================="
echo

# Docker コンテナをインタラクティブモードで実行
docker run \
    --rm \
    -it \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/cache:/app/cache" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/audio:/app/audio:ro" \
    --entrypoint /bin/bash \
    whisper-benchmark:latest

echo
echo "Exited interactive mode."