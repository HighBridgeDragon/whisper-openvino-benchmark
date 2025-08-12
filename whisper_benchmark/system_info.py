"""
システム情報取得モジュール

CPU情報やシステム詳細を取得する機能を提供
"""

import platform
import subprocess
import cpuinfo
import psutil


def get_cpu_info():
    """CPU情報をpy-cpuinfoを使用して取得"""
    try:
        cpu_info_data = cpuinfo.get_cpu_info()

        cpu_name = cpu_info_data.get("brand_raw", "Unknown")
        cores = psutil.cpu_count(logical=False) or 1
        threads = psutil.cpu_count(logical=True) or 1

        # CPU フラグを取得
        detected_flags = set(cpu_info_data.get("flags", []))

        # フラグを機能名にマッピング
        features = []

        flag_mapping = {
            # ベクトル命令（OpenVINO性能に重要）
            "avx": "AVX",
            "avx2": "AVX2",
            "fma": "FMA",
            # AVX512ファミリー（高性能推論）
            "avx512f": "AVX512F",
            "avx512dq": "AVX512DQ",
            "avx512cd": "AVX512CD",
            "avx512bw": "AVX512BW",
            "avx512vl": "AVX512VL",
            "avx512vnni": "AVX512VNNI",
            "avx512_bf16": "AVX512-BF16",
            # AI最適化命令（ニューラルネットワーク加速）
            "vnni": "VNNI",
            "amx_tile": "AMX-TILE",
            "amx_int8": "AMX-INT8",
            "amx_bf16": "AMX-BF16",
        }

        for flag, feature_name in flag_mapping.items():
            if flag in detected_flags:
                features.append(feature_name)

        # Windows WMI フォールバック（必要な場合）
        if cpu_name == "Unknown" and platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name", "/value"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Name="):
                        cpu_name = line.split("=", 1)[1].strip()
                        break
            except Exception:
                pass

        return {
            "name": cpu_name,
            "cores": cores,
            "threads": threads,
            "features": features,
            "detection_methods": "py-cpuinfo",
        }

    except Exception as e:
        print(f"Warning: Could not get CPU info: {e}")
        return {
            "name": "Unknown",
            "cores": psutil.cpu_count(logical=False) or 1,
            "threads": psutil.cpu_count(logical=True) or 1,
            "features": [],
            "detection_methods": "Failed",
        }


def get_system_info():
    """システム情報の完全な詳細を取得"""
    cpu_info = get_cpu_info()

    return {
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": cpu_info,
        "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
    }


def print_system_info():
    """システム情報をコンソールに表示"""
    print("=== System Information ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")

    cpu_info = get_cpu_info()
    print(f"CPU: {cpu_info['name']}")
    print(f"Cores: {cpu_info['cores']}, Threads: {cpu_info['threads']}")
    if cpu_info["features"]:
        print(f"Features: {', '.join(cpu_info['features'])}")

    print(f"Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print()
