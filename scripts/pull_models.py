#!/usr/bin/env python3
"""
Pre-download specified HF models (tokenizers + model weights) into the HuggingFace cache.

Usage:
  export HUGGINGFACE_HUB_TOKEN="<your-token>"
  python scripts/pull_models.py

Optional env vars:
  HF_MODEL_CACHE_DIR - path to use as `cache_dir` for transformers downloads.
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    # "meta-llama/Meta-Llama-3-8B-Instruct",
    "ZJU-AI4H/Hulu-Med-7B",
]

def main():
    cache_dir = os.environ.get("HF_MODEL_CACHE_DIR")
    # 提取环境变量中的 Token
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") 
    
    print("HF cache dir:", cache_dir or "(default)")
    if not hf_token:
        print("⚠️ Warning: HUGGINGFACE_HUB_TOKEN is not set. Downloading gated models will fail.")

    for m in MODELS:
        print(f"\nDownloading model: {m}")
        try:
            # 将 token=hf_token 传入
            AutoTokenizer.from_pretrained(
                m, 
                use_fast=False, 
                cache_dir=cache_dir,
                token=hf_token 
            )
            # 将 token=hf_token 传入
            AutoModelForCausalLM.from_pretrained(
                m, 
                low_cpu_mem_usage=True, 
                cache_dir=cache_dir, 
                trust_remote_code=True,
                token=hf_token
            )
            print(f"  ✓ {m} downloaded")
        except Exception as e:
            print(f"  ✗ Failed to download {m}: {e}")

    print("\nAll done.")

if __name__ == '__main__':
    main()