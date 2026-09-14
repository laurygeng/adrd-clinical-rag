# core/modelA.py
"""
Model A shim (renamed from openai.py). Exposes `OpenAI` with same interface as before.
Includes robust OOM protection, variable scoping fixes, and repetition penalties.
"""

import os
import gc
import sys
import torch
import traceback
import re
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_model = None
_tokenizer = None
_current_model_id = None  # 用于追踪当前加载的模型，防止显存泄漏

def _is_llama3_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "llama-3" in mid or "meta-llama-3" in mid

def _clean_llama3_output(text: str, messages) -> str:
    cleaned = (text or "").strip()

    for marker in ["\nassistant:", "\nuser:", "assistant:", "user:", "<|start_header_id|>", "<|eot_id|>"]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()

    last_user = str((messages or [{}])[-1].get("content", "") or "")
    upper_cleaned = cleaned.upper()
    upper_user = last_user.upper()

    if "EXACTLY 'YES' OR 'NO'" in upper_user or 'EXACTLY "YES" OR "NO"' in upper_user:
        if upper_cleaned.startswith("YES"):
            return "Yes"
        if upper_cleaned.startswith("NO"):
            return "No"

    if "EXACTLY ONE OPTION LETTER" in upper_user:
        match = re.search(r"\b([A-E])\b", upper_cleaned)
        if match:
            return match.group(1)

    return cleaned

def _get_model_and_tokenizer(model_id: str = None):
    global _model, _tokenizer, _current_model_id
    
    target_model_id = model_id or os.environ.get("LOCAL_AGENT_A_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    
    # 如果模型未加载，或者需要切换到新的模型，则执行加载/重载逻辑
    if _model is None or target_model_id != _current_model_id:
        
        # [修复 2] 显存 OOM 保护：安全释放旧模型，使用 = None 防止 NameError
        if _model is not None:
            print(f"\n[Shim Debug] >>> Unloading previous model: {_current_model_id} to free VRAM <<<")
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

        print(f"\n[Shim Debug] >>> Loading Native Model: {target_model_id} (4-bit: {bnb_config is not None}) <<<")

        _tokenizer = AutoTokenizer.from_pretrained(
            target_model_id,
            token=hf_token,
            trust_remote_code=True,
        )

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
        if model and str(model).lower().startswith("gpt"):
            target_id = os.environ.get("LOCAL_AGENT_A_MODEL", "")
            llm, tokenizer = _get_model_and_tokenizer(target_id)
            model_id = target_id
        else:
            llm, tokenizer = _get_model_and_tokenizer(model)
            model_id = model or os.environ.get("LOCAL_AGENT_A_MODEL", "")
            
        is_llama3 = _is_llama3_model(model_id)
        
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(llm.device)
            inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        except Exception:
            prompt = "".join([f"{m.get('role')}: {m.get('content')}\n" for m in messages]) + "assistant:\n"
            fallback_inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
            inputs = {"input_ids": fallback_inputs.input_ids, "attention_mask": fallback_inputs.attention_mask}
        
        # [修复 3] 加入 repetition_penalty 防止死循环输出乱码
        gen_kwargs = {
            "max_new_tokens": int(max_tokens) if is_llama3 else max(32, int(max_tokens)),
            "do_sample": temperature > 0.0,
            "repetition_penalty": 1.15,  
        }

        if is_llama3:
            eos_token_ids = []
            if tokenizer.eos_token_id is not None:
                eos_token_ids.append(int(tokenizer.eos_token_id))
            try:
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None and eot_id != tokenizer.unk_token_id:
                    eos_token_ids.append(int(eot_id))
            except Exception:
                pass
            if eos_token_ids:
                gen_kwargs["eos_token_id"] = eos_token_ids
                gen_kwargs["pad_token_id"] = eos_token_ids[0]
        
        if temperature > 0.0:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(kwargs.get("top_p", 0.9))

        generated_text = ""
        try:
            with torch.no_grad():
                outputs = llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
            
            seq = outputs[0]
            input_length = inputs["input_ids"].shape[1]
            
            if len(seq) > input_length:
                generated_tokens = seq[input_length:].tolist()
            else:
                generated_tokens = seq.tolist()
                
            # [修复 3] 启用 clean_up_tokenization_spaces 防止不完整字符解码报错
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            
            if is_llama3:
                generated_text = _clean_llama3_output(generated_text, messages)
            
        except Exception as e:
            err_txt = traceback.format_exc()
            msg_preview = " | ".join([f"{m.get('role')}:{str(m.get('content',''))[:200]}" for m in (messages or [])])
            raise RuntimeError(f"Model inference failed: {err_txt}\nModel preview: {msg_preview}")
            
        finally:
            # [修复 1] 安全检查，防止 UnboundLocalError 掩盖原始报错
            if 'inputs' in locals() and inputs is not None:
                del inputs
            if 'outputs' in locals() and outputs is not None:
                del outputs
            torch.cuda.empty_cache()
            gc.collect()

        if not generated_text:
            msg_preview = " | ".join([f"{m.get('role')}:{str(m.get('content',''))[:200]}" for m in (messages or [])])
            raise RuntimeError(f"Empty model output (FAILED_EMPTY_OUTPUT). Model_id={model_id}. Prompt preview: {msg_preview}")

        print(f"[Shim Trace] ✅ 成功生成答案: {repr(generated_text)}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=generated_text))])

class _Chat:
    def __init__(self):
        self.completions = _ChatCompletions()

class OpenAI:
    def __init__(self, api_key=None, **kwargs):
        self.chat = _Chat()