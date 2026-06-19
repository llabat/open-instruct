"""Verify Jean Zay Llama-3.1-8B config and weight shapes match official specs."""

import json
import sys
from pathlib import Path

import safetensors

BASE = Path("/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B")

# Official Llama-3.1-8B architecture specs from Meta's release
EXPECTED_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
}

# A few key tensors with expected shapes
EXPECTED_SHAPES = {
    "model.embed_tokens.weight": (128256, 4096),
    "model.layers.0.self_attn.q_proj.weight": (4096, 4096),
    "model.layers.0.self_attn.k_proj.weight": (1024, 4096),
    "model.layers.0.self_attn.v_proj.weight": (1024, 4096),
    "model.layers.0.mlp.gate_proj.weight": (14336, 4096),
    "model.layers.31.self_attn.q_proj.weight": (4096, 4096),
    "lm_head.weight": (128256, 4096),
}


def check_config():
    print("=== config.json ===")
    cfg = json.loads((BASE / "config.json").read_text())
    all_ok = True
    for key, expected in EXPECTED_CONFIG.items():
        actual = cfg.get(key)
        ok = actual == expected
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(f"  {key:<30} expected={expected!r:<15} actual={actual!r}  [{status}]")
    return all_ok


def check_shapes():
    print("\n=== tensor shapes (from safetensors header) ===")
    # Build index: key → shard file
    index_path = BASE / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        # Single-file model
        index = {k: "model.safetensors" for k in EXPECTED_SHAPES}

    all_ok = True
    for key, expected_shape in EXPECTED_SHAPES.items():
        shard = BASE / index.get(key, "model.safetensors")
        with safetensors.safe_open(str(shard), framework="pt", device="cpu") as f:
            meta = f.get_slice(key)
            actual_shape = tuple(meta.get_shape())
        ok = actual_shape == expected_shape
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(f"  {key:<55} expected={str(expected_shape):<18} actual={actual_shape}  [{status}]")
    return all_ok


def check_tokenizer():
    print("\n=== tokenizer ===")
    tc = json.loads((BASE / "tokenizer_config.json").read_text())
    print(f"  tokenizer_class : {tc.get('tokenizer_class')}")
    print(f"  model_max_length: {tc.get('model_max_length')}")
    print(f"  has chat_template: {bool(tc.get('chat_template'))}")
    tok = json.loads((BASE / "tokenizer.json").read_text())
    vocab_size = len(tok.get("model", {}).get("vocab", {}))
    print(f"  vocab size (tokenizer.json): {vocab_size}")


def main():
    config_ok = check_config()
    shapes_ok = check_shapes()
    check_tokenizer()
    print(f"\n{'All checks passed' if config_ok and shapes_ok else 'MISMATCHES FOUND — base model differs from official spec'}")


if __name__ == "__main__":
    sys.exit(main())
