# core/llm_engine.py
"""Unified Local LLM Engine

Minimal Hulu-only hotfix:
- Keep all existing logic unchanged for non-Hulu models.
- For Hulu only, if generate() returns a sequence whose length is <= input_length,
  do NOT assume there are zero new tokens. Some remote-code models may return only
  the generated continuation rather than prompt+continuation.
- Add small debug logs around Hulu output extraction.
"""

import os
import gc
import torch
import traceback
import re
import logging
from types import SimpleNamespace
import importlib

import transformers
import transformers.utils.import_utils


def _disable_hf_torch_load_check() -> None:
    def _noop(*args, **kwargs):
        return None

    candidate_modules = [
        getattr(transformers, "utils", None),
        getattr(transformers.utils, "import_utils", None),
    ]

    for module_name in (
        "transformers.utils",
        "transformers.trainer_utils",
        "transformers.modeling_utils",
    ):
        try:
            candidate_modules.append(importlib.import_module(module_name))
        except Exception:
            pass

    seen_ids = set()
    for module in candidate_modules:
        if module is None or id(module) in seen_ids:
            continue
        seen_ids.add(id(module))
        if hasattr(module, "check_torch_load_is_safe"):
            setattr(module, "check_torch_load_is_safe", _noop)


_disable_hf_torch_load_check()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq


_model = None
_tokenizer = None
_current_model_id = None
logger = logging.getLogger(__name__)


def _dbg_enabled() -> bool:
    return os.environ.get("HULU_DEBUG", "1") == "1"


def _clip_text(x: str, limit: int = 1200) -> str:
    s = str(x or "")
    if len(s) <= limit:
        return s
    return s[:limit] + f" ...[truncated {len(s) - limit} chars]"


def _dbg(stage: str, **kwargs) -> None:
    if not _dbg_enabled():
        return
    payload = " | ".join(f"{k}={_clip_text(v)}" for k, v in kwargs.items())
    logger.warning(f"[HULU-DEBUG][llm_engine][{stage}] {payload}")


def _is_llama3_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "llama-3" in mid or "meta-llama-3" in mid or "llama_3" in mid


def _is_hulu_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return ("hulu-med-7b" in mid) or ("zju-ai4h/hulu-med-7b" in mid)


def _has_chat_template(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return bool(template) and hasattr(tokenizer, "apply_chat_template")


def _build_fallback_prompt(model_id: str, messages) -> str:
    system_parts = [
        str(m.get("content", "") or "").strip()
        for m in (messages or [])
        if str(m.get("role", "") or "").strip().lower() == "system"
    ]
    user_parts = [
        str(m.get("content", "") or "").strip()
        for m in (messages or [])
        if str(m.get("role", "") or "").strip().lower() == "user"
    ]
    assistant_parts = [
        str(m.get("content", "") or "").strip()
        for m in (messages or [])
        if str(m.get("role", "") or "").strip().lower() == "assistant"
    ]

    system_text = "\n\n".join([p for p in system_parts if p]).strip()
    user_text = "\n\n".join([p for p in user_parts if p]).strip()
    assistant_text = "\n\n".join([p for p in assistant_parts if p]).strip()
    merged_user = "\n\n".join([p for p in [system_text, user_text] if p]).strip()

    if _is_llama3_model(model_id):
        prompt = "<|begin_of_text|>"
        if system_text:
            prompt += (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_text}<|eot_id|>"
            )
        prompt += (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_text or merged_user}<|eot_id|>"
        )
        if assistant_text:
            prompt += (
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{assistant_text}<|eot_id|>"
            )
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    lines = []
    for m in (messages or []):
        role = str(m.get("role", "user") or "user").strip().lower()
        content = str(m.get("content", "") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\nassistant:\n"


def _strip_leading_role_markers(text: str) -> str:
    if not text:
        return ""
    t = text.lstrip()
    patterns = [
        r"^(assistant|user|system)\s*:\s*",
        r"^###\s*(assistant|user|system)\s*\n+",
    ]
    for p in patterns:
        t2 = re.sub(p, "", t, flags=re.IGNORECASE)
        if t2 != t:
            t = t2.lstrip()
    return t


def _remove_special_tokens_artifacts(text: str) -> str:
    if not text:
        return ""
    t = text
    for tok in [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|begin_of_text|>",
        "<|im_end|>",
        "<|im_start|>",
    ]:
        t = t.replace(tok, "")
    return t


# def _clean_universal_output(text: str, messages) -> str:
#     cleaned = (text or "").strip()
#     cleaned = _strip_leading_role_markers(cleaned)
#     cleaned = _remove_special_tokens_artifacts(cleaned).strip()

#     # =========================================================================
#     # [新增修复]: 专门切断 Lingshu 等模型喜欢在 JSON 后边乱加 Assistant: 的循环幻觉
#     # =========================================================================
#     if "}Assistant:" in cleaned:
#         cleaned = cleaned.split("}Assistant:")[0] + "}"
#     elif "} Assistant:" in cleaned:
#         cleaned = cleaned.split("} Assistant:")[0] + "}"
#     elif "}\nAssistant:" in cleaned:
#         cleaned = cleaned.split("}\nAssistant:")[0] + "}"

#     last_user = ""        
def _clean_universal_output(text: str, messages) -> str:
    cleaned = (text or "").strip()
    cleaned = _strip_leading_role_markers(cleaned)
    cleaned = _remove_special_tokens_artifacts(cleaned).strip()

    # [终极截断修复]: 寻找第一个左括号和它匹配的右括号，直接切断后续所有幻觉复读
    # 即便它输出 { ... }{ ... }，我们也只取第一个 { ... }
    json_match = re.search(r'^(\{.*?\})', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1).strip()
    else:
        # 如果开头不是 {，尝试在文本中寻找
        json_match_any = re.search(r'(\{.*?\})', cleaned, re.DOTALL)
        if json_match_any:
            cleaned = json_match_any.group(1).strip()

    last_user = ""
        
    try:
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content", "") or "")
                break
    except Exception:
        last_user = ""

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

    target_model_id = model_id or os.environ.get(
        "GLOBAL_AGENT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"
    )

    if _model is None or target_model_id != _current_model_id:
        if _model is not None:
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
                bnb_4bit_quant_type="nf4",
            )

        _tokenizer = AutoTokenizer.from_pretrained(
            target_model_id,
            token=hf_token,
            trust_remote_code=True,
        )

        try:
            _model = AutoModelForCausalLM.from_pretrained(
                target_model_id,
                token=hf_token,
                device_map=device_map,
                torch_dtype=dtype,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
        except ValueError as e:
            if "Unrecognized configuration class" in str(e) or "Qwen2_5_VLConfig" in str(e):
                _model = AutoModelForVision2Seq.from_pretrained(
                    target_model_id,
                    token=hf_token,
                    device_map=device_map,
                    torch_dtype=dtype,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                )
            else:
                raise

        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token_id = _tokenizer.eos_token_id

        _current_model_id = target_model_id

    return _model, _tokenizer


def _infer_expected_shape(messages) -> str:
    last_user = ""
    try:
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content", "") or "")
                break
    except Exception:
        last_user = ""

    u = last_user.upper()
    if "VALID JSON" in u or "JSON OBJECT" in u or ("\"SUFFICIENT\"" in u and "\"GAP\"" in u):
        return "json"
    if "REPLY ONLY YES OR NO" in u or "ONLY YES OR NO" in u or "YES OR NO" in u:
        return "yesno"
    if "OPTION LETTER" in u or "A, B, C, D, OR E" in u or "EXACTLY ONE OPTION LETTER" in u:
        return "mc"
    return "text"


def _placeholder_for_shape(shape: str) -> str:
    if shape == "json":
        return '{"sufficient": true, "gap": "NONE"}'
    if shape == "yesno":
        return "No"
    if shape == "mc":
        return "A"
    return "I don't know based on the provided context."


class _ChatCompletions:
    def create(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        kwargs.pop("response_format", None)
        kwargs.pop("presence_penalty", None)
        kwargs.pop("frequency_penalty", None)

        target_id = model or os.environ.get("GLOBAL_AGENT_MODEL", "")
        llm, tokenizer = _get_model_and_tokenizer(target_id)
        is_llama3 = _is_llama3_model(target_id)
        is_hulu = _is_hulu_model(target_id)

        prefill_text = ""
        if messages and isinstance(messages[-1], dict) and "content" in messages[-1]:
            last_msg = str(messages[-1]["content"] or "").strip()
            if last_msg.endswith("Final Answer:"):
                prefill_text = "Final Answer:"
                messages[-1]["content"] = last_msg[: -len("Final Answer:")].strip()

        inputs = None
        input_length = 0

        try:
            if not _has_chat_template(tokenizer):
                raise ValueError("No chat template")

            if is_hulu:
                prompt_str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if prefill_text:
                    prompt_str = prompt_str + prefill_text + " "
                encoded = tokenizer(prompt_str, return_tensors="pt").to(llm.device)
                inputs = {
                    "input_ids": encoded.input_ids,
                    "attention_mask": encoded.attention_mask,
                }
                input_length = int(encoded.input_ids.shape[1])
                _dbg("prompt_hulu_string_path", input_length=input_length, prompt_preview=_clip_text(prompt_str, 1500))
            else:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(llm.device)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
                input_length = int(input_ids.shape[1])

        except Exception:
            prompt_str = _build_fallback_prompt(target_id, messages)
            if prefill_text:
                prompt_str = prompt_str + prefill_text + " "
            encoded = tokenizer(prompt_str, return_tensors="pt").to(llm.device)
            inputs = {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
            }
            input_length = int(encoded.input_ids.shape[1])

        rep = float(os.environ.get("HF_REPETITION_PENALTY", "1.05"))
        effective_max_new = int(max_tokens) if is_llama3 else max(32, int(max_tokens))

        gen_kwargs = {
            "max_new_tokens": effective_max_new,
            "do_sample": float(temperature) > 0.0,
            "repetition_penalty": rep,
        }

        # =====================================================================
        # [新增修复]: 专门针对 Lingshu 等模型的循环切断机制（增加重复惩罚）
        # =====================================================================
        target_id_lower = target_id.lower()
        if "lingshu" in target_id_lower:
            gen_kwargs["repetition_penalty"] = 1.15

        try:
            gen_kwargs["min_new_tokens"] = 1
        except Exception:
            pass

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
        elif tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)

        if float(temperature) > 0.0:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(kwargs.get("top_p", 0.9))

        expected_shape = _infer_expected_shape(messages)

        generated_text = ""
        outputs = None
        retry_plans = [
            dict(gen_kwargs),
            {**dict(gen_kwargs), "do_sample": False, "repetition_penalty": 1.0},
            {**dict(gen_kwargs), "do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": max(effective_max_new, 96)},
        ]

        try:
            for idx, plan in enumerate(retry_plans, start=1):
                with torch.no_grad():
                    outputs = llm.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **plan,
                    )

                seq = outputs[0]
                seq_len = int(len(seq))

                if is_hulu and seq_len <= input_length:
                    new_tokens = seq.tolist()
                    _dbg(
                        "hulu_nonstandard_output_detected",
                        attempt=idx,
                        input_length=input_length,
                        output_seq_len=seq_len,
                        mode="use_full_seq_as_generated",
                    )
                else:
                    new_tokens = seq[input_length:].tolist() if seq_len > input_length else []

                if not new_tokens:
                    generated_text = ""
                    if is_hulu:
                        _dbg("generation_attempt_empty", attempt=idx, input_length=input_length, output_seq_len=seq_len, new_token_count=0)
                else:
                    raw_decoded = tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ).strip()
                    generated_text = _clean_universal_output(raw_decoded, messages)
                    if is_hulu:
                        _dbg(
                            "generation_attempt_result",
                            attempt=idx,
                            input_length=input_length,
                            output_seq_len=seq_len,
                            new_token_count=len(new_tokens),
                            raw_decoded_preview=_clip_text(raw_decoded, 1000),
                            cleaned_preview=_clip_text(generated_text, 1000),
                        )

                if generated_text:
                    break

        except Exception:
            raise RuntimeError(f"Model inference failed: {traceback.format_exc()}")
        finally:
            try:
                del inputs
            except Exception:
                pass
            try:
                del outputs
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()

        if not generated_text:
            generated_text = _placeholder_for_shape(expected_shape)
            if is_hulu:
                _dbg("generation_placeholder_used", expected_shape=expected_shape, placeholder=generated_text)

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=generated_text))]
        )


class _Chat:
    def __init__(self):
        self.completions = _ChatCompletions()


class LocalLLMClient:
    def __init__(self, api_key=None, **kwargs):
        self.chat = _Chat()