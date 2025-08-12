FROM ghcr.io/astral-sh/uv:debian

# 作業ディレクトリの設定
WORKDIR /app

# システムライブラリのインストール
RUN apt-get update && apt-get install -y \
    # 音声処理に必要
    libsndfile1 \
    ffmpeg \
    # ビルドツール
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uvのインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

# uvのインストール確認
RUN which uv && uv --version

# プロジェクトファイルのコピー
COPY pyproject.toml uv.lock .python-version ./
COPY main.py ./

# 必要なディレクトリの作成
RUN mkdir -p /app/models /app/cache /app/output /app/audio

# 依存関係のインストール（CPU専用、CUDA無効）
ENV CUDA_VISIBLE_DEVICES=""
ENV UV_EXTRA_INDEX_URL=""
ENV UV_INDEX_URL="https://pypi.org/simple/"

# 依存関係を完全にインストール（開発依存も含めて確実にインストール）
RUN uv sync --frozen

# 依存関係の解決を完全に行い、キャッシュを準備
RUN uv pip list --format=json > /tmp/installed_packages.json

# 実行時の追加ダウンロードを防ぐため、依存関係を再確認
RUN uv pip check

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/app/output

# デフォルトコマンド
ENTRYPOINT ["uv", "run", "python", "main.py"]