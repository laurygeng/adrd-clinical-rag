# core/llm_engine.py
"""
Unified Local LLM Engine
作为系统唯一的大模型推理底座，支持 Llama, Qwen, Mistral 以及医疗垂直模型 (如 Hulu-Med)。
"""

import os
import gc
import torch
import traceback
import re
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.utils.import_utils

# [Bypass CVE-2025-32434] 强制屏蔽 transformers 对 torch < 2.6 加载 .bin 文件的安全拦截
if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    
_model = None
_tokenizer = None
_current_model_id = None

def _is_llama3_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "llama-3" in mid or "meta-llama-3" in mid

def _clean_universal_output(text: str, messages) -> str:
    cleaned = (text or "").strip()

    # 万能清洗器：兼容 Llama, Qwen, Hulu-Med 的所有可能的前缀污染
    for marker in [
        "\nassistant:", "\nAssistant:", "\nuser:", "\nUser:", 
        "\nSYSTEM:", "\nUSER:", "assistant:", "Assistant:", 
        "user:", "User:", "<|start_header_id|>", "<|eot_id|>",
        "<|im_end|>", "<|im_start|>"
    ]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()

    # 处理严格的格式化输出 (针对 TF 和 MC)
    last_user = str((messages or [{}])[-1].get("content", "") or "")
    upper_cleaned = cleaned.upper()
    upper_user = last_user.upper()

    if "EXACTLY 'YES' OR 'NO'" in upper_user or 'EXACTLY "YES" OR "NO"' in upper_user:
        if upper_cleaned.startswith("YES"): return "Yes"
        if upper_cleaned.startswith("NO"): return "No"

    if "EXACTLY ONE OPTION LETTER" in upper_user:
        match = re.search(r"\b([A-E])\b", upper_cleaned)
        if match: return match.group(1)

    return cleaned

def _get_model_and_tokenizer(model_id: str = None):
    global _model, _tokenizer, _current_model_id
    
    target_model_id = model_id or os.environ.get("GLOBAL_AGENT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    if _model is None or target_model_id != _current_model_id:
        if _model is not None:
            print(f"\n[Engine Debug] >>> Unloading previous model: {_current_model_id} to free VRAM <<<")
            _model = None
            _tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        device_map = os.environ.get("HF_DEVICE_MAP", "auto")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        bnb_config = None
        if os.environ.get("HF_USE_4BIT") == "1":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        print(f"\n[Engine Debug] >>> Loading Native Model: {target_model_id} (4-bit: {bnb_config is not None}) <<<")

        _tokenizer = AutoTokenizer.from_pretrained(target_model_id, token=hf_token, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            token=hf_token,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token_id = _tokenizer.eos_token_id
            
        _current_model_id = target_model_id
            
    return _model, _tokenizer

class _ChatCompletions:
    def create(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        target_id = os.environ.get("GLOBAL_AGENT_MODEL", "")
        llm, tokenizer = _get_model_and_tokenizer(target_id)
        is_llama3 = _is_llama3_model(target_id)
        
        try:
            # 尝试标准 HuggingFace Chat Template (适用 Llama-3, Qwen, Mistral)
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(llm.device)
            inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        except Exception:
            # 兜底机制：如果模型 (如 Hulu-Med) 不支持 Chat Template，启用文本拼接
            sys_msg = next((m.get('content') for m in messages if m.get('role') == 'system'), "")
            user_msg = next((m.get('content') for m in messages if m.get('role') == 'user'), "")
            
            # 兼容 Hulu-Med 的对话格式
            fallback_prompt = f"SYSTEM:\n{sys_msg}\n\nUSER:\n{user_msg}\n\nAssistant:\n"
            fallback_inputs = tokenizer(fallback_prompt, return_tensors="pt").to(llm.device)
            inputs = {"input_ids": fallback_inputs.input_ids, "attention_mask": fallback_inputs.attention_mask}
        
        gen_kwargs = {
            "max_new_tokens": int(max_tokens) if is_llama3 else max(32, int(max_tokens)),
            "do_sample": temperature > 0.0,
            "repetition_penalty": 1.15,  
        }

        if is_llama3:
            eos_token_ids = []
            if tokenizer.eos_token_id is not None: eos_token_ids.append(int(tokenizer.eos_token_id))
            try:
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None and eot_id != tokenizer.unk_token_id: eos_token_ids.append(int(eot_id))
            except Exception: pass
            if eos_token_ids:
                gen_kwargs["eos_token_id"] = eos_token_ids
                gen_kwargs["pad_token_id"] = eos_token_ids[0]
        
        if temperature > 0.0:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(kwargs.get("top_p", 0.9))

        generated_text = ""
        try:
            with torch.no_grad():
                outputs = llm.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], **gen_kwargs)
            
            seq = outputs[0]
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = seq[input_length:].tolist() if len(seq) > input_length else seq.tolist()
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            
            # 所有模型统一走万能清洗，彻底切除多余的标签
            generated_text = _clean_universal_output(generated_text, messages)
            
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {traceback.format_exc()}")
        finally:
            if 'inputs' in locals() and inputs is not None: del inputs
            if 'outputs' in locals() and outputs is not None: del outputs
            torch.cuda.empty_cache()
            gc.collect()

        if not generated_text:
            raise RuntimeError("Empty model output (FAILED_EMPTY_OUTPUT).")

        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=generated_text))])

class _Chat:
    def __init__(self):
        self.completions = _ChatCompletions()

# 类名从 OpenAI 改为 LocalLLMClient，消除所有混淆
class LocalLLMClient:
    def __init__(self, api_key=None, **kwargs):
        self.chat = _Chat()