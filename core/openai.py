# core/openai.py
import os
import gc
import sys
import torch
import traceback
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_model = None
_tokenizer = None

def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        model_id = os.environ.get("LOCAL_AGENT_A_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
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

        print(f"\n[Shim Debug] >>> Loading Native Model: {model_id} (4-bit: {bnb_config is not None}) <<<")
        
        _tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=hf_token, 
            trust_remote_code=True
        )
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token_id = _tokenizer.eos_token_id
            
    return _model, _tokenizer

class _ChatCompletions:
    def create(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        llm, tokenizer = _get_model_and_tokenizer()
        
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
        
        gen_kwargs = {
            "max_new_tokens": max(32, max_tokens),
            "do_sample": temperature > 0.0,
        }
        
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
                
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
        except Exception as e:
            # 🚨 强制熔断机制 1：代码执行报错，立刻停机！
            print("\n" + "❌"*30)
            print("🚨 [CRITICAL ERROR] 模型推理发生底层崩溃！")
            print(f"具体报错信息:\n{traceback.format_exc()}")
            print("❌"*30 + "\n")
            os._exit(1) # 直接强杀进程
            
        finally:
            del inputs
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            gc.collect()

        if not generated_text:
            # 🚨 强制熔断机制 2：模型吐出空字符串，立刻停机！
            print("\n" + "⚠️"*30)
            print("🚨 [FATAL WARNING] 模型输出了空字符串 (FAILED_EMPTY_OUTPUT)！")
            print("为了防止浪费时间，已强制终止整个评测任务。请检查 Prompt 或模型状态。")
            print("⚠️"*30 + "\n")
            os._exit(1) # 直接强杀进程

        # 如果能顺利走到这里，说明生成绝对成功，打印一下生成的答案让你安心
        print(f"[Shim Trace] ✅ 成功生成答案: {repr(generated_text)}")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=generated_text))])

class _Chat:
    def __init__(self):
        self.completions = _ChatCompletions()

class OpenAI:
    def __init__(self, api_key=None, **kwargs):
        self.chat = _Chat()