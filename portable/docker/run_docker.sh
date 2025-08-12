#!/bin/bash
# Docker コンテナ実行スクリプト

echo "=========================================================="
echo "Whisper Benchmark Docker Runner"
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

# 必要なディレクトリの作成
echo "Checking required directories..."
mkdir -p models cache output audio

# Docker イメージの存在確認
if ! docker images | grep -q "whisper-benchmark.*latest"; then
    echo "Error: Docker image 'whisper-benchmark:latest' not found."
    echo "Please run ./build_docker.sh first."
    exit 1
fi

# デフォルトパラメータ
DOCKER_ARGS=""
BENCHMARK_ARGS=""

# ヘルプメッセージ
show_help() {
    echo "Usage: ./run_docker.sh [options]"
    echo
    echo "Options:"
    echo "  --model-path PATH     Path to Whisper model directory (inside container: /app/models/...)"
    echo "  --iterations N        Number of benchmark iterations (default: 5)"
    echo "  --num-beams N         Number of beams for decoding (default: 1)"
    echo "  --device DEVICE       Device to use: CPU, GPU (default: CPU)"
    echo "  --audio-file PATH     Path to audio file (inside container: /app/audio/...)"
    echo "  --help                Show this help message"
    echo
    echo "Volume mappings:"
    echo "  ./models  -> /app/models   (read-only)"
    echo "  ./cache   -> /app/cache"
    echo "  ./output  -> /app/output"
    echo "  ./audio   -> /app/audio    (read-only)"
    echo
    echo "Examples:"
    echo "  ./run_docker.sh --model-path /app/models/whisper-large-v3-turbo-stateless"
    echo "  ./run_docker.sh --audio-file /app/audio/my_audio.wav --iterations 10"
    echo
    echo "Note: Model paths should use the container path (/app/models/...)"
    exit 0
}

# 引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            ;;
        *)
            # すべての引数をベンチマークに渡す
            BENCHMARK_ARGS="$BENCHMARK_ARGS $1"
            shift
            ;;
    esac
done

# モデルの自動検出（引数が指定されていない場合）
if [ -z "$BENCHMARK_ARGS" ]; then
    echo "Auto-detecting model..."
    CONFIG_FILE=$(find models -name "config.json" -type f 2>/dev/null | head -1)
    if [ -n "$CONFIG_FILE" ]; then
        # ホストパスをコンテナパスに変換
        MODEL_REL_PATH=$(echo "$CONFIG_FILE" | sed 's|^models/||' | sed 's|/config.json$||')
        BENCHMARK_ARGS="--model-path /app/models/$MODEL_REL_PATH"
        echo "Found model: $MODEL_REL_PATH"
    else
        echo "Warning: No model found in models/ directory."
        echo "Proceeding without model path (will use default or fail)."
    fi
fi

echo "Running benchmark with arguments: $BENCHMARK_ARGS"
echo "=========================================================="
echo

# Docker コンテナの実行（root権限で実行）
docker run \
    --rm \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/cache:/app/cache" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/audio:/app/audio:ro" \
    whisper-benchmark:latest \
    $BENCHMARK_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo
    echo "=========================================================="
    echo "Benchmark completed successfully!"
    echo "=========================================================="
    echo
    echo "Results saved to: ./output/benchmark_results.yaml"
    
    # 結果ファイルの確認
    if [ -f "./output/benchmark_results.yaml" ]; then
        echo
        echo "Note: Output files are owned by root. Use 'cat' or 'less' to view:"
        echo "  cat ./output/benchmark_results.yaml"
        echo
        echo "Quick preview of results:"
        head -n 20 ./output/benchmark_results.yaml
    fi
else
    echo
    echo "=========================================================="
    echo "Benchmark failed with exit code: $EXIT_CODE"
    echo "=========================================================="
    echo
    echo "Troubleshooting tips:"
    echo "1. Check if model exists in ./models/ directory"
    echo "2. Verify model was exported with --disable-stateful flag"
    echo "3. Check Docker container logs"
    echo "4. Ensure audio file exists if specified"
fi

exit $EXIT_CODE