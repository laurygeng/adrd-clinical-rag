# core/local_google_shim.py
import os
import gc
import torch
from types import SimpleNamespace
from transformers import pipeline, BitsAndBytesConfig

_hulumed_pipeline = None

def _get_hulumed_pipeline():
    global _hulumed_pipeline
    if _hulumed_pipeline is None:
        model_id = os.environ.get("LOCAL_AGENT_B_MODEL", "ZJU-AI4H/Hulu-Med-7B")
        cache_dir = os.environ.get("HF_HOME", "/users/minjieg/code/scripts/models")
        
        device_map = os.environ.get("HF_DEVICE_MAP", "auto")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model_kwargs = {"torch_dtype": dtype}
        
        # 真正激活 4-bit 压缩配置
        if os.environ.get("HF_USE_4BIT") == "1":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        print(f"\n[Shim] Loading local model for Google GenAI API: {model_id} (4-bit: {os.environ.get('HF_USE_4BIT') == '1'})...")

        try:
            _hulumed_pipeline = pipeline(
                "text-generation",
                model=model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
                device_map=device_map,
                local_files_only=True,
            )
        except TypeError as e:
            print(f"[Shim Warning] Caught pipeline init error ({e}), retrying with minimal kwargs...")
            if "model_kwargs" in str(e) or "not used" in str(e):
                _hulumed_pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    device_map=device_map,
                    local_files_only=True,
                )
            else:
                raise e
                
    return _hulumed_pipeline

class _Models:
    def generate_content(self, model, contents, config=None):
        pipe = _get_hulumed_pipeline()
        prompt = contents[0] if isinstance(contents, list) else str(contents or "")
        
        temp = 0.2
        max_tokens = 256
        if config:
            try:
                temp = float(getattr(config, "temperature", config.get("temperature", temp)))
            except Exception:
                pass
            try:
                max_tokens = int(getattr(config, "max_output_tokens", config.get("max_output_tokens", max_tokens)))
            except Exception:
                pass

        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temp > 0.0}
        if temp > 0.0:
            gen_kwargs["temperature"] = float(temp)

        try:
            outputs = pipe(prompt, **gen_kwargs, return_full_text=False)
        except Exception:
            outputs = pipe(prompt, max_new_tokens=max_tokens, return_full_text=False)
        finally:
            # 强制清空 GPU KV Cache 垃圾
            torch.cuda.empty_cache()
            gc.collect()

        generated_text = outputs[0].get("generated_text") if isinstance(outputs[0], dict) else str(outputs[0])
        if isinstance(generated_text, str):
            generated_text = generated_text.strip()

        return SimpleNamespace(text=generated_text or "")

class Client:
    def __init__(self, api_key=None, **kwargs):
        self.models = _Models()