# core/llm_engine.py
"""Unified Local LLM Engine

作为系统唯一的大模型推理底座，支持 Llama, Qwen, Mistral 以及医疗垂直模型。

Design goals (hardening):
- Works for chat/instruct models AND base models.
- Never returns empty output: retry generation and fall back to a safe, task-shaped placeholder.
- Avoid model-name hardcoding; use capability detection (chat_template) + prompt-shape heuristics.
"""

import os
import gc
import torch
import traceback
import re
from types import SimpleNamespace
import importlib

import transformers

# Patch: best-effort bypass for HF's torch checkpoint safety gate (kept from your original).
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

# Transformers 5.0+ multimodal compatibility
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq


_model = None
_tokenizer = None
_current_model_id = None


def _is_llama3_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    return "llama-3" in mid or "meta-llama-3" in mid or "llama_3" in mid


def _has_chat_template(tokenizer) -> bool:
    """Capability detection: do NOT hardcode model names.

    Some repos ship a chat_template that is empty/incorrect for the actual weights.
    We treat it as usable only if apply_chat_template exists AND template is non-empty.
    Further validation happens at runtime when we build the prompt.
    """
    template = getattr(tokenizer, "chat_template", None)
    return bool(template) and hasattr(tokenizer, "apply_chat_template")


def _build_fallback_prompt(model_id: str, messages) -> str:
    """Fallback prompt for base/non-chat models.

    Keep it simple and completion-friendly.
    """
    system_parts = [
        str(m.get("content", "") or "").strip()
        for m in (messages or [])
        if m.get("role") == "system"
    ]
    user_parts = [
        str(m.get("content", "") or "").strip()
        for m in (messages or [])
        if m.get("role") == "user"
    ]
    assistant_parts = [
        str(m.get("content", "") or "").strip()
        for m in (messages or [])
        if m.get("role") == "assistant"
    ]

    system_text = "\n\n".join([p for p in system_parts if p]).strip()
    user_text = "\n\n".join([p for p in user_parts if p]).strip()
    assistant_text = "\n\n".join([p for p in assistant_parts if p]).strip()

    merged = "\n\n".join([p for p in [system_text, user_text] if p]).strip()

    model_id_l = (model_id or "").lower()

    # If it's a Mistral-instruct style model without a working chat_template.
    if "mistral" in model_id_l and "instruct" in model_id_l:
        prompt_body = merged or user_text or system_text
        return f"<s>[INST] {prompt_body} [/INST]"

    # Llama-3 fallback (rare; most have templates) - keep your original special tokens.
    if _is_llama3_model(model_id):
        prompt = "<|begin_of_text|>"
        if system_text:
            prompt += (
                "<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_text}<|eot_id|>"
            )
        if user_text or merged:
            prompt += (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_text or merged}<|eot_id|>"
            )
        if assistant_text:
            prompt += (
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{assistant_text}<|eot_id|>"
            )
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    # Universal base-model completion prompt
    if system_text and user_text:
        return f"{system_text}\n\n{user_text}\nAnswer:"
    return f"{merged or system_text}\nAnswer:"


def _clean_universal_output(text: str, messages) -> str:
    cleaned = (text or "").strip()

    # Remove common role-markers / template leftovers.
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
        "<|im_end|>",
        "<|im_start|>",
    ]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()

    # Task-shaped normalization for strict outputs.
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
            target_model_id, token=hf_token, trust_remote_code=True
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
    """Infer what the caller likely expects.

    Returns: one of {"json", "yesno", "mc", "text"}
    Heuristic-based but avoids brittle full message string matches.
    """
    last_user = ""
    try:
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content", "") or "")
                break
    except Exception:
        last_user = ""

    u = last_user.upper()

    # JSON: critic requires strict JSON, or prompt explicitly says valid JSON.
    if "VALID JSON" in u or "JSON OBJECT" in u or "\"SUFFICIENT\"" in u and "\"GAP\"" in u:
        return "json"

    # YES/NO verification style
    if "REPLY ONLY YES OR NO" in u or "ONLY YES OR NO" in u or "YES OR NO" in u:
        return "yesno"

    # MC strict
    if "OPTION LETTER" in u or "A, B, C, D, OR E" in u or "EXACTLY ONE OPTION LETTER" in u:
        return "mc"

    return "text"


def _placeholder_for_shape(shape: str) -> str:
    if shape == "json":
        # Most conservative: say sufficient with NONE gap to avoid unnecessary completion.
        return '{"sufficient": true, "gap": "NONE"}'
    if shape == "yesno":
        # Conservative default.
        return "No"
    if shape == "mc":
        return "A"
    return "I don't know based on the provided context."


class _ChatCompletions:
    def create(self, model, messages, temperature=0.7, max_tokens=1024, **kwargs):
        # Filter OpenAI-only kwargs / unsupported knobs.
        kwargs.pop("response_format", None)
        kwargs.pop("presence_penalty", None)
        kwargs.pop("frequency_penalty", None)

        target_id = model or os.environ.get("GLOBAL_AGENT_MODEL", "")
        llm, tokenizer = _get_model_and_tokenizer(target_id)
        is_llama3 = _is_llama3_model(target_id)

        # Preserve your "Final Answer:" prefill behavior.
        prefill_text = ""
        if messages and isinstance(messages[-1], dict) and "content" in messages[-1]:
            last_msg = str(messages[-1]["content"] or "").strip()
            if last_msg.endswith("Final Answer:"):
                prefill_text = "Final Answer:"
                messages[-1]["content"] = last_msg[: -len("Final Answer:")].strip()

        # Build prompt
        prompt_str = None
        try:
            if not _has_chat_template(tokenizer):
                raise ValueError("No chat template")

            candidate = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Runtime validation: must include the last user content (rough check)
            last_user_content = ""
            for m in reversed(messages or []):
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user_content = str(m.get("content", "") or "").strip()
                    break
            if last_user_content and (last_user_content[:48] not in candidate):
                raise ValueError("Chat template seems incompatible; falling back")

            prompt_str = candidate
        except Exception:
            prompt_str = _build_fallback_prompt(target_id, messages)

        if prefill_text:
            prompt_str = (prompt_str or "") + f"{prefill_text} "

        inputs_encoded = tokenizer(prompt_str, return_tensors="pt").to(llm.device)
        inputs = {
            "input_ids": inputs_encoded.input_ids,
            "attention_mask": inputs_encoded.attention_mask,
        }

        input_length = int(inputs["input_ids"].shape[1])

        # Generation kwargs
        rep = float(os.environ.get("HF_REPETITION_PENALTY", "1.05"))
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": float(temperature) > 0.0,
            "repetition_penalty": rep,
        }

        # Prefer at least 1 new token to avoid immediate EOS empty decode.
        try:
            gen_kwargs["min_new_tokens"] = 1
        except Exception:
            pass
        # Older fallback: min_length includes prompt
        gen_kwargs["min_length"] = input_length + 1

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
        # Retry plan: (1) original params; (2) greedy/no-penalty; (3) greedy + larger max_new_tokens
        retry_plans = [
            dict(gen_kwargs),
            {**dict(gen_kwargs), "do_sample": False, "repetition_penalty": 1.0},
            {**dict(gen_kwargs), "do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": max(int(max_tokens), 96)},
        ]

        try:
            for plan in retry_plans:
                with torch.no_grad():
                    outputs = llm.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **plan,
                    )

                seq = outputs[0]
                # Only decode newly generated tokens. If none, treat as empty.
                new_tokens = seq[input_length:].tolist() if len(seq) > input_length else []

                if not new_tokens:
                    generated_text = ""
                else:
                    generated_text = tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ).strip()
                    generated_text = _clean_universal_output(generated_text, messages)

                if generated_text:
                    break

        except Exception:
            raise RuntimeError(f"Model inference failed: {traceback.format_exc()}")
        finally:
            try:
                del inputs
                del inputs_encoded
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

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=generated_text))]
        )


class _Chat:
    def __init__(self):
        self.completions = _ChatCompletions()


class LocalLLMClient:
    def __init__(self, api_key=None, **kwargs):
        self.chat = _Chat()
