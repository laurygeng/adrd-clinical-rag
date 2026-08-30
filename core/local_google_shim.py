# core/local_google_shim.py
import os
import gc
import torch
import traceback
import re
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_hulumed_model = None
_hulumed_tokenizer = None


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _build_pretrained_kwargs(cache_dir: str | None, hf_token: str | None, local_files_only: bool) -> dict:
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if hf_token:
        kwargs["token"] = hf_token
    return kwargs

def _clean_generated_text(text: str) -> str:
    cleaned = str(text or "").strip()
    for marker in [
        "\nassistant:",
        "\nAssistant:",
        "\nuser:",
        "\nUser:",
        "\nSYSTEM:",
        "\nUSER:",
        "assistant:",
        "Assistant:",
        "user:",
        "User:",
        "<|start_header_id|>",
        "<|eot_id|>",
    ]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _coerce_text_prompt(contents) -> str:
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(contents or "").strip()


def _build_messages(prompt: str):
    raw = str(prompt or "")
    if "SYSTEM:\n" in raw and "\n\nUSER:\n" in raw:
        system_part, user_part = raw.split("\n\nUSER:\n", 1)
        system_part = system_part.replace("SYSTEM:\n", "", 1).strip()
        user_part = user_part.strip()
        return [
            {"role": "system", "content": system_part},
            {"role": "user", "content": user_part},
        ]
    return [{"role": "user", "content": raw.strip()}]


def _get_hulumed_model_and_tokenizer():
    global _hulumed_model, _hulumed_tokenizer
    if _hulumed_model is None or _hulumed_tokenizer is None:
        model_id = os.environ.get("LOCAL_AGENT_B_MODEL", "ZJU-AI4H/Hulu-Med-7B")
        cache_dir = os.environ.get("HF_HOME") or None
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        local_files_only = _env_flag("HF_LOCAL_FILES_ONLY", "0")
        
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
            
        print(
            f"\n[Shim] Loading local model for Google GenAI API: {model_id} "
            f"(4-bit: {os.environ.get('HF_USE_4BIT') == '1'}, local_only: {local_files_only})..."
        )

        pretrained_kwargs = _build_pretrained_kwargs(cache_dir, hf_token, local_files_only)

        try:
            _hulumed_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                **pretrained_kwargs,
            )
            _hulumed_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                **pretrained_kwargs,
                **model_kwargs,
            )
        except OSError as e:
            if local_files_only:
                raise e
            print(f"[Shim Warning] Initial load failed ({e}), retrying without cache_dir...")
            retry_kwargs = _build_pretrained_kwargs(None, hf_token, False)
            _hulumed_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                **retry_kwargs,
            )
            _hulumed_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                **retry_kwargs,
                **model_kwargs,
            )
        except TypeError as e:
            print(f"[Shim Warning] Caught model init error ({e}), retrying with minimal kwargs...")
            if "model_kwargs" in str(e) or "not used" in str(e) or "quantization_config" in str(e):
                retry_kwargs = _build_pretrained_kwargs(cache_dir, hf_token, local_files_only)
                _hulumed_tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    **retry_kwargs,
                )
                _hulumed_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    **retry_kwargs,
                    torch_dtype=dtype,
                )
            else:
                raise e

        if _hulumed_tokenizer.pad_token_id is None:
            _hulumed_tokenizer.pad_token_id = _hulumed_tokenizer.eos_token_id

    return _hulumed_model, _hulumed_tokenizer

class _Models:
    def generate_content(self, model, contents, config=None):
        llm, tokenizer = _get_hulumed_model_and_tokenizer()
        prompt = _coerce_text_prompt(contents)
        messages = _build_messages(prompt)
        
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

        gen_kwargs = {"max_new_tokens": max(16, int(max_tokens)), "do_sample": temp > 0.0}
        if temp > 0.0:
            gen_kwargs["temperature"] = float(temp)
            gen_kwargs["top_p"] = 0.9

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

        inputs = None
        outputs = None
        generated_text = ""

        try:
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(llm.device)
                inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            except Exception:
                fallback_prompt = "".join([f"{m.get('role')}: {m.get('content')}\n" for m in messages]) + "assistant:\n"
                fallback_inputs = tokenizer(fallback_prompt, return_tensors="pt").to(llm.device)
                inputs = {"input_ids": fallback_inputs.input_ids, "attention_mask": fallback_inputs.attention_mask}

            with torch.no_grad():
                outputs = llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )

            seq = outputs[0]
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = seq[input_length:].tolist() if len(seq) > input_length else seq.tolist()
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            generated_text = _clean_generated_text(generated_text)

            if not generated_text:
                fallback_prompt = prompt if prompt else "\n".join(m.get("content", "") for m in messages)
                retry_inputs = tokenizer(fallback_prompt, return_tensors="pt").to(llm.device)
                with torch.no_grad():
                    retry_outputs = llm.generate(
                        input_ids=retry_inputs.input_ids,
                        attention_mask=retry_inputs.attention_mask,
                        **gen_kwargs,
                    )
                retry_seq = retry_outputs[0]
                retry_input_length = retry_inputs.input_ids.shape[1]
                retry_tokens = retry_seq[retry_input_length:].tolist() if len(retry_seq) > retry_input_length else retry_seq.tolist()
                generated_text = tokenizer.decode(retry_tokens, skip_special_tokens=True).strip()
                generated_text = _clean_generated_text(generated_text)
        except Exception:
            print("[Shim Error] Google GenAI local generation failed:\n" + traceback.format_exc())
        finally:
            if inputs is not None:
                del inputs
            if outputs is not None:
                del outputs
            torch.cuda.empty_cache()
            gc.collect()

        if not generated_text:
            print("[Shim Warning] Google GenAI local shim produced empty output.")
        else:
            print(f"[Shim Trace] Gemini-local output: {repr(generated_text)}")

        return SimpleNamespace(text=generated_text or "")

class Client:
    def __init__(self, api_key=None, **kwargs):
        self.models = _Models()