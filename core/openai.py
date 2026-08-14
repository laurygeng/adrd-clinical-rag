# core/openai.py
import os
import torch
from types import SimpleNamespace
from transformers import pipeline

_llama_pipeline = None

def _get_pipeline():
    global _llama_pipeline
    if _llama_pipeline is None:
        model_id = os.environ.get("LOCAL_AGENT_A_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        print(f"\n[Shim] Loading local model for OpenAI API: {model_id}...")

        device_map = os.environ.get("HF_DEVICE_MAP", "auto")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs = {"torch_dtype": dtype}

        # Completely remove explicit cache_dir and local_files_only to prevent keyword argument mismatch errors
        _llama_pipeline = pipeline(
            "text-generation",
            model=model_id,
            token=hf_token,
            model_kwargs=model_kwargs,
            device_map=device_map,
        )
    return _llama_pipeline

class _ChatCompletions:
    def create(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        pipe = _get_pipeline()
        
        try:
            prompt = pipe.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            prompt = str(messages)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(kwargs.get("top_p", 0.9))

        outputs = pipe(prompt, **gen_kwargs, return_full_text=False)

        generated_text = outputs[0].get("generated_text") if isinstance(outputs[0], dict) else str(outputs[0])
        if isinstance(generated_text, str):
            generated_text = generated_text.strip()

        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=generated_text or ""))])

class _Chat:
    def __init__(self):
        self.completions = _ChatCompletions()

class OpenAI:
    def __init__(self, api_key=None, **kwargs):
        self.chat = _Chat()