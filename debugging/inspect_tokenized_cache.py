"""Inspect the cached tokenized SFT corpus to verify chat template was applied correctly."""

import json
import sys
from pathlib import Path

import datasets
from transformers import AutoTokenizer

CACHE_DIR = Path("/lustre/fswork/projects/rech/oag/unz84ar/data/dataset_cache/563f1e8f05")
TOKENIZER_PATH = "/lustre/fsmisc/dataset/HuggingFace_Models/meta-llama/Llama-3.1-8B"
N_EXAMPLES = 5


def main():
    print("=== Cache config ===")
    with open(CACHE_DIR / "config.json") as f:
        config = json.load(f)
    print(json.dumps(config["tokenizer_config"], indent=2))

    print("\n=== Dataset statistics ===")
    with open(CACHE_DIR / "dataset_statistics.json") as f:
        stats = json.load(f)
    s = stats["per_dataset_stats"][0]
    print(f"  initial_instances : {s['initial_instances']:,}")
    print(f"  final_instances   : {s['final_instances']:,}")
    print(f"  instances_filtered: {s['instances_filtered']:,}")
    print(f"  total_tokens      : {s['total_tokens']:,}")
    print(f"  trainable_tokens  : {s['trainable_tokens']:,}")
    print(f"  avg_tokens/example: {s['avg_tokens_per_instance']:.1f}")

    print(f"\n=== Loading tokenizer from {TOKENIZER_PATH} ===")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
    print(f"  tokenizer class   : {type(tokenizer).__name__}")
    has_template = bool(getattr(tokenizer, "chat_template", None))
    print(f"  has chat_template : {has_template}")

    print(f"\n=== Reading {N_EXAMPLES} examples from cache ===")
    ds = datasets.load_from_disk(str(CACHE_DIR))

    indices = list(range(N_EXAMPLES))
    for i in indices:
        input_ids = ds[i]["input_ids"]
        labels = ds[i]["labels"]
        trainable = [l for l in labels if l != -100]
        first_trainable_idx = next((j for j, l in enumerate(labels) if l != -100), -1)

        decoded_start = tokenizer.decode(input_ids[:30])
        decoded_end = tokenizer.decode(input_ids[-20:])

        print(f"\n--- Example {i} ---")
        print(f"  length           : {len(input_ids)} tokens")
        print(f"  trainable tokens : {len(trainable)} / {len(labels)}")
        print(f"  first trainable  : index {first_trainable_idx}")
        print(f"  first token id   : {input_ids[0]} (BOS={tokenizer.bos_token_id})")
        print(f"  last token id    : {input_ids[-1]} (EOS={tokenizer.eos_token_id})")
        print(f"  start (30 tok)   : {repr(decoded_start)}")
        print(f"  end   (20 tok)   : {repr(decoded_end)}")

        # Check tulu-format markers are present
        full_text = tokenizer.decode(input_ids)
        has_user_tag = "<|user|>" in full_text
        has_asst_tag = "<|assistant|>" in full_text
        print(f"  has <|user|>     : {has_user_tag}")
        print(f"  has <|assistant|>: {has_asst_tag}")

    # Compare against reference: re-tokenize one raw example with the tulu template
    print("\n=== Reference: what the tulu template should produce ===")
    from open_instruct.dataset_transformation import CHAT_TEMPLATES
    tokenizer.chat_template = CHAT_TEMPLATES["tulu"]
    sample_messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    ref = tokenizer.apply_chat_template(sample_messages, tokenize=False)
    print(f"  {repr(ref)}")


if __name__ == "__main__":
    sys.exit(main())
