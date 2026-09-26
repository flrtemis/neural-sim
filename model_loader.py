"""
model_loader.py — Load HuggingFace model with 4-bit quantization + LoRA adapter.

Supports Qwen3-8B and Llama3.1-8B. Uses bitsandbytes NF4 quantization to fit
in 16GB VRAM with room for LoRA training gradients.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def load_model(config: dict) -> tuple:
    """Load quantized model with LoRA adapter.

    Returns (model, tokenizer, peft_config).
    """
    model_cfg = config["model"]
    model_path = model_cfg["path"]

    # --- Quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg["quantization"],
        bnb_4bit_compute_dtype=getattr(torch, model_cfg["compute_dtype"]),
        bnb_4bit_use_double_quant=True,
    )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Base model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=getattr(torch, model_cfg["compute_dtype"]),
        attn_implementation="eager",  # Need raw attention weights for hooks
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # --- LoRA adapter ---
    peft_config = LoraConfig(
        r=model_cfg["lora_rank"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        target_modules=model_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # --- Summary ---
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    print(f"[model_loader] Loaded: {model_path}")
    print(f"[model_loader] Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")
    print(f"[model_loader] VRAM used: {vram_mb:.0f} MB")

    return model, tokenizer, peft_config


def get_layer_names(model) -> list[str]:
    """Return the names of all transformer block layers for hook registration."""
    names = []
    for name, module in model.named_modules():
        # Match transformer block layers (works for both Qwen3 and Llama)
        if any(pat in name for pat in [".layers.", ".h."]):
            # Only top-level blocks, not sub-modules
            parts = name.split(".")
            # e.g. "model.model.layers.0" or "base_model.model.model.layers.0"
            if parts[-1].isdigit():
                names.append(name)
    return names


def get_embedding_layer(model):
    """Return the embedding weight tensor (for drift computation)."""
    for name, param in model.named_parameters():
        if "embed_tokens" in name and "weight" in name:
            return param.data.clone()
    return None
