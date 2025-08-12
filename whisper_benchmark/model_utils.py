"""
モデルファイル検証ユーティリティ

Whisperモデルファイルの存在確認と検証機能を提供
"""

import os


def validate_model_files(model_path):
    """
    必要なモデルファイルがすべて存在するかを検証

    Args:
        model_path: モデルディレクトリのパス

    Returns:
        bool: すべてのファイルが存在する場合True
    """
    required_files = [
        "openvino_encoder_model.xml",
        "openvino_encoder_model.bin",
        "openvino_decoder_model.xml",
        "openvino_decoder_model.bin",
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_detokenizer.bin",
        "config.json",
        "generation_config.json",
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"Warning: Missing model files: {', '.join(missing_files)}")
        print("This may indicate an incomplete model export.")
        print("Try re-exporting the model with --disable-stateful flag:")
        print(
            f"uv run optimum-cli export openvino -m openai/whisper-large-v3-turbo --trust-remote-code --weight-format fp32 --disable-stateful {model_path}"
        )
        return False

    print("OK All required model files found")
    return True
