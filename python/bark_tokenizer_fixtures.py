#!/usr/bin/env python3
"""Generate compact Python Transformers tokenizer fixtures for Bark.

This script is developer tooling only. The Rust Bark backend consumes the JSON
fixtures in tests; it never imports or calls Python at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CASES = [
    ("short_english", "hello from puppygrad"),
    ("punctuation", "Hello, Bark! Are you ready?"),
    ("numbers", "Order 42 costs $19.95 on 2026-05-23."),
    ("mixed_case", "MiXeD Case Bark Tokens"),
    (
        "long_truncates",
        " ".join(["puppygrad"] * 320),
    ),
    ("unknown_non_ascii", "naive cafe Привет こんにちは"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def module_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def normalized_text(tokenizer: Any, text: str) -> str | None:
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    normalizer = getattr(backend_tokenizer, "normalizer", None)
    if normalizer is not None and hasattr(normalizer, "normalize_str"):
        return str(normalizer.normalize_str(text))
    return None


def main() -> None:
    args = parse_args()

    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir)
    generation_config = json.loads(
        (args.model_dir / "generation_config.json").read_text(encoding="utf-8")
    )
    semantic_config = generation_config["semantic_config"]
    tokenizer = processor.tokenizer
    max_input_len = int(semantic_config["max_input_semantic_length"])
    pad_token_id = int(tokenizer.pad_token_id)

    cases = []
    for label, text in DEFAULT_CASES:
        inputs = processor(text)
        input_ids = inputs["input_ids"][0].detach().cpu().numpy().astype(int).tolist()
        attention_mask = (
            inputs["attention_mask"][0].detach().cpu().numpy().astype(int).tolist()
        )
        if len(input_ids) > max_input_len:
            input_ids = input_ids[:max_input_len]
            attention_mask = attention_mask[:max_input_len]
        elif len(input_ids) < max_input_len:
            pad_len = max_input_len - len(input_ids)
            input_ids.extend([pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(input_ids)]
        semantic_input_ids = [
            int(token_id) + int(semantic_config["text_encoding_offset"])
            if int(mask) == 1
            else int(semantic_config["text_pad_token"])
            for token_id, mask in zip(input_ids, attention_mask)
        ]
        cases.append(
            {
                "label": label,
                "text": text,
                "normalized_text": normalized_text(tokenizer, text),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "tokens": tokens,
                "semantic_input_ids": semantic_input_ids,
            }
        )

    fixture = {
        "metadata": {
            "model_dir": str(args.model_dir),
            "transformers_version": module_version(transformers),
            "case_count": len(cases),
            "text_encoding_offset": int(semantic_config["text_encoding_offset"]),
            "text_pad_token": int(semantic_config["text_pad_token"]),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(cases)} tokenizer cases")


if __name__ == "__main__":
    main()
