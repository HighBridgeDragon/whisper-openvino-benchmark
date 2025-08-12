# Whisper Benchmark Docker版 使用ガイド

このディレクトリには、Docker を使用した Whisper Benchmark の実行環境が含まれています。

## 概要

Docker版の利点：
- **環境の一貫性**: どのLinux環境でも同じ動作を保証
- **簡単なセットアップ**: Docker があれば追加のインストール不要
- **隔離された実行環境**: ホストシステムへの影響なし
- **ポータブルな配布**: Docker イメージとして配布可能

## 前提条件

- Docker がインストールされていること
- 十分なディスク容量（イメージサイズ約2GB、作業領域含めて5GB以上推奨）

## ディレクトリ構造

```
whisper-openvino-benchmark/
├── portable/docker/         # Docker関連ファイル
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── build_docker.sh      # イメージビルド
│   ├── run_docker.sh        # ベンチマーク実行
│   ├── run_interactive.sh   # 対話モード
│   └── README_DOCKER.md     # このファイル
├── models/                  # Whisperモデル配置（コンテナから読み取り専用）
├── cache/                   # 音声ファイルキャッシュ
├── output/                  # ベンチマーク結果出力
└── audio/                   # カスタム音声ファイル（コンテナから読み取り専用）
```

## 使用方法

### 1. Docker イメージのビルド

```bash
cd portable/docker
./build_docker.sh
```

ビルドには数分かかります。成功すると `whisper-benchmark:latest` イメージが作成されます。

### 2. ベンチマークの実行

#### 基本実行（モデル自動検出）
```bash
./run_docker.sh
```

#### モデル指定実行
```bash
./run_docker.sh --model-path /app/models/whisper-large-v3-turbo-stateless
```

#### カスタム音声ファイル使用
```bash
# audioディレクトリに音声ファイルを配置してから
./run_docker.sh --audio-file /app/audio/my_audio.wav
```

#### その他のオプション
```bash
./run_docker.sh --model-path /app/models/whisper-base --iterations 10 --num-beams 5
```

### 3. 対話モード（デバッグ用）

```bash
./run_interactive.sh
```

コンテナ内でbashシェルが起動し、手動でコマンドを実行できます：

```bash
# コンテナ内で実行
uv run python main.py --model-path /app/models/your-model --help
```

## ボリュームマウント

| ホスト側 | コンテナ側 | モード | 用途 |
|----------|------------|--------|------|
| `./models` | `/app/models` | 読み取り専用 | Whisperモデル |
| `./cache` | `/app/cache` | 読み書き | 音声ファイルキャッシュ |
| `./output` | `/app/output` | 読み書き | ベンチマーク結果 |
| `./audio` | `/app/audio` | 読み取り専用 | カスタム音声ファイル |

## モデルの準備

### 1. モデルのエクスポート

```bash
# 開発環境で実行（Dockerコンテナ外）
uv run optimum-cli export openvino \
  -m openai/whisper-large-v3-turbo \
  --trust-remote-code \
  --weight-format fp32 \
  --disable-stateful \
  models/whisper-large-v3-turbo-stateless
```

### 2. 既存モデルの配置

既存のOpenVINO形式のWhisperモデルを `models/` ディレクトリに配置してください。

必要なファイル：
- `config.json`
- `generation_config.json`
- `openvino_encoder_model.xml/.bin`
- `openvino_decoder_model.xml/.bin`
- `openvino_tokenizer.xml/.bin`
- `openvino_detokenizer.xml/.bin`

## 結果の確認

ベンチマーク結果は `./output/benchmark_results.yaml` に保存されます：

```yaml
cpu_info: "Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz"
model_path: "/app/models/whisper-large-v3-turbo-stateless"
inference_times: [2.145, 2.089, 2.123, ...]
rtf: 0.892
transcription: "How are you doing today?"
...
```

## トラブルシューティング

### Docker イメージのビルドが失敗する場合

```bash
# Docker デーモンの確認
docker ps

# ディスク容量の確認
df -h

# 古いイメージの削除
docker system prune -a
```

### モデルが見つからない場合

```bash
# モデルディレクトリの確認
ls -la models/

# コンテナ内からのパス確認
./run_interactive.sh
# コンテナ内で: ls -la /app/models/
```

### メモリ不足エラー

`docker-compose.yml` でリソース制限を調整：

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # より大きなメモリ割り当て
```

### 権限エラー

出力ディレクトリの権限を確認：

```bash
chmod 755 output/
sudo chown -R $USER:$USER output/
```

## Docker Compose を使用した実行（オプション）

上級ユーザー向けに`docker-compose.yml`も提供されています：

```bash
# docker-compose で実行（配布先）
docker-compose run --rm whisper-benchmark --model-path /app/models/your-model

# または開発環境から
docker-compose -f portable/docker/docker-compose.yml run --rm whisper-benchmark --model-path /app/models/your-model
```

**注意**: 通常は `./run_docker.sh` の使用を推奨します。docker-compose は以下の場合に有用です：
- 環境変数を細かく調整したい場合
- 複数のサービスと連携する場合
- CI/CD環境での自動実行

## パフォーマンス最適化

### CPU スレッド数の調整

`docker-compose.yml` で環境変数を調整：

```yaml
environment:
  - OMP_NUM_THREADS=16  # CPUコア数に応じて調整
  - TBB_NUM_THREADS=16
```

### メモリ制限の調整

```yaml
deploy:
  resources:
    limits:
      memory: 8G    # 使用可能メモリに応じて調整
    reservations:
      memory: 4G
```

## セキュリティ

- コンテナは非rootユーザー（whisper:1000）で実行
- モデルと音声ファイルは読み取り専用でマウント
- 不要な権限は付与されていません

## 制限事項

- GPU サポートは含まれていません（CPUのみ）
- Windows コンテナには対応していません
- リアルタイム音声入力には対応していません

## サポート

問題が発生した場合：

1. Docker ログの確認: `docker logs <container_id>`
2. 対話モードでのデバッグ: `./run_interactive.sh`
3. イメージの再ビルド: `./build_docker.sh`
4. システムクリーンアップ: `docker system prune`