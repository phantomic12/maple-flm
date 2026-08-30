#!/usr/bin/env python3
"""
Maple-Preview to FastFlowLM Converter
Converts deepgrove/maple-preview Hugging Face checkpoint to FastFlowLM format.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

def convert_config(hf_config: dict) -> dict:
    """Convert Hugging Face MapleConfig to FastFlowLM config format."""
    flm_config = dict(hf_config)
    flm_config["model_type"] = "maple"
    flm_config["family"] = "maple"
    flm_config["flm_version"] = "1.0.0"
    
    # Ensure all required dimensions are explicitly present
    flm_config["vocab_size"] = hf_config.get("vocab_size", 151936)
    flm_config["hidden_size"] = hf_config.get("hidden_size", 2048)
    flm_config["num_hidden_layers"] = hf_config.get("num_hidden_layers", 24)
    flm_config["num_attention_heads"] = hf_config.get("num_attention_heads", 16)
    flm_config["num_key_value_heads"] = hf_config.get("num_key_value_heads", 4)
    flm_config["head_dim"] = hf_config.get("head_dim", 128)
    flm_config["num_experts"] = hf_config.get("num_experts", 256)
    flm_config["num_experts_per_tok"] = hf_config.get("num_experts_per_tok", 8)
    flm_config["moe_intermediate_size"] = hf_config.get("moe_intermediate_size", 512)
    flm_config["sliding_window"] = hf_config.get("sliding_window", 512)
    flm_config["rms_norm_eps"] = hf_config.get("rms_norm_eps", 1e-6)
    flm_config["rope_theta"] = hf_config.get("rope_theta", 10000.0)
    flm_config["partial_rotary_factor"] = hf_config.get("partial_rotary_factor", 0.5)
    flm_config["nope_on_global_attention"] = hf_config.get("nope_on_global_attention", True)
    flm_config["default_context_length"] = 32768
    flm_config["max_position_embeddings"] = hf_config.get("max_position_embeddings", 131072)
    
    return flm_config

def convert_tokenizer_config(hf_tok_cfg: dict) -> dict:
    """Ensure tokenizer_config.json contains appropriate eos_token_id and template."""
    flm_tok_cfg = dict(hf_tok_cfg)
    
    if "eos_token_id" not in flm_tok_cfg or flm_tok_cfg["eos_token_id"] is None:
        flm_tok_cfg["eos_token_id"] = [151645]
    elif isinstance(flm_tok_cfg["eos_token_id"], int):
        flm_tok_cfg["eos_token_id"] = [flm_tok_cfg["eos_token_id"]]
        
    if "bos_token_id" not in flm_tok_cfg or flm_tok_cfg["bos_token_id"] is None:
        flm_tok_cfg["bos_token_id"] = 151643
        
    if "eos_token" not in flm_tok_cfg or flm_tok_cfg["eos_token"] is None:
        flm_tok_cfg["eos_token"] = "<|im_end|>"
        
    return flm_tok_cfg

def convert_model_directory(src_dir: Path, out_dir: Path):
    """Converts a local directory of deepgrove/maple-preview files to FastFlowLM format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Config
    src_config_file = src_dir / "config.json"
    if src_config_file.exists():
        with open(src_config_file, "r") as f:
            hf_config = json.load(f)
        flm_config = convert_config(hf_config)
        with open(out_dir / "config.json", "w") as f:
            json.dump(flm_config, f, indent=2)
        print(f"[OK] Converted config.json -> {out_dir / 'config.json'}")
    else:
        print(f"[WARN] config.json not found in {src_dir}")

    # 2. Tokenizer config
    src_tok_cfg_file = src_dir / "tokenizer_config.json"
    if src_tok_cfg_file.exists():
        with open(src_tok_cfg_file, "r") as f:
            hf_tok_cfg = json.load(f)
        flm_tok_cfg = convert_tokenizer_config(hf_tok_cfg)
        with open(out_dir / "tokenizer_config.json", "w") as f:
            json.dump(flm_tok_cfg, f, indent=2)
        print(f"[OK] Converted tokenizer_config.json -> {out_dir / 'tokenizer_config.json'}")

    # 3. Copy other tokenizer assets & chat template
    for filename in ["tokenizer.json", "chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]:
        src_file = src_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, out_dir / filename)
            print(f"[OK] Copied {filename} -> {out_dir / filename}")

    print("\nModel files conversion completed successfully.")
    print(f"Target directory: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert deepgrove/maple-preview to FastFlowLM format.")
    parser.add_argument("--src-dir", type=str, required=True, help="Source directory of deepgrove/maple-preview checkpoint")
    parser.add_argument("--out-dir", type=str, required=True, help="Destination directory for FastFlowLM model files")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    if not src_dir.exists():
        print(f"Error: source directory {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    convert_model_directory(src_dir, out_dir)

if __name__ == "__main__":
    main()
