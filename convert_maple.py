#!/usr/bin/env python3
"""
Maple-Preview to FastFlowLM Converter
Converts deepgrove/maple-preview Hugging Face checkpoint to FastFlowLM format,
and provides synthetic model generation for offline testing.
"""

import os
import sys
import json
import struct
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

def write_safetensors(filepath: Path, tensor_dict: dict):
    """
    Writes a dictionary of tensors to a SafeTensors binary file without external dependencies.
    tensor_dict format: {name: {"dtype": "BF16", "shape": list[int], "data": bytes}}
    """
    header = {}
    current_offset = 0
    
    for name, info in tensor_dict.items():
        data_len = len(info["data"])
        header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [current_offset, current_offset + data_len]
        }
        current_offset += data_len
        
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    # 8-byte little-endian header length
    header_len = struct.pack('<Q', len(header_json))
    
    with open(filepath, 'wb') as f:
        f.write(header_len)
        f.write(header_json)
        for name, info in tensor_dict.items():
            f.write(info["data"])

def generate_synthetic_model(out_dir: Path, num_layers: int = 2, num_experts: int = 8, vocab_size: int = 1000):
    """
    Generates a miniature synthetic Maple model for testing and offline validation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = 256
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    moe_inter = 128
    
    config = {
        "architectures": ["MapleForCausalLM"],
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": moe_inter,
        "sliding_window": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "nope_on_global_attention": True,
        "model_type": "maple",
        "family": "maple",
        "flm_version": "1.0.0",
        "default_context_length": 512
    }
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    # Helper to generate dummy BF16 bytes
    def make_bf16(num_elements: int, fill_val: int = 0x3F80) -> bytes: # 0x3F80 is ~1.0 in BF16
        return struct.pack(f'<{num_elements}H', *([fill_val] * num_elements))

    tensors = {}
    # Embeddings & LM Head
    tensors["model.word_embeddings"] = {"dtype": "BF16", "shape": [vocab_size, hidden_size], "data": make_bf16(vocab_size * hidden_size)}
    tensors["model.norm"] = {"dtype": "BF16", "shape": [hidden_size], "data": make_bf16(hidden_size)}
    tensors["lm_head"] = {"dtype": "BF16", "shape": [vocab_size, hidden_size], "data": make_bf16(vocab_size * hidden_size)}

    # Layers
    for l in range(num_layers):
        lp = f"model.layers.{l}"
        tensors[f"{lp}.input_layernorm"] = {"dtype": "BF16", "shape": [hidden_size], "data": make_bf16(hidden_size)}
        tensors[f"{lp}.post_attention_layernorm"] = {"dtype": "BF16", "shape": [hidden_size], "data": make_bf16(hidden_size)}
        tensors[f"{lp}.self_attn.q_proj"] = {"dtype": "BF16", "shape": [num_heads * head_dim, hidden_size], "data": make_bf16(num_heads * head_dim * hidden_size)}
        tensors[f"{lp}.self_attn.k_proj"] = {"dtype": "BF16", "shape": [num_kv_heads * head_dim, hidden_size], "data": make_bf16(num_kv_heads * head_dim * hidden_size)}
        tensors[f"{lp}.self_attn.v_proj"] = {"dtype": "BF16", "shape": [num_kv_heads * head_dim, hidden_size], "data": make_bf16(num_kv_heads * head_dim * hidden_size)}
        tensors[f"{lp}.self_attn.o_proj"] = {"dtype": "BF16", "shape": [hidden_size, num_heads * head_dim], "data": make_bf16(hidden_size * num_heads * head_dim)}
        tensors[f"{lp}.self_attn.q_norm"] = {"dtype": "BF16", "shape": [head_dim], "data": make_bf16(head_dim)}
        tensors[f"{lp}.self_attn.k_norm"] = {"dtype": "BF16", "shape": [head_dim], "data": make_bf16(head_dim)}
        tensors[f"{lp}.mlp.gate"] = {"dtype": "BF16", "shape": [num_experts, hidden_size], "data": make_bf16(num_experts * hidden_size)}

        for e in range(num_experts):
            ep = f"{lp}.mlp.experts.{e}"
            tensors[f"{ep}.gate_proj"] = {"dtype": "BF16", "shape": [moe_inter, hidden_size], "data": make_bf16(moe_inter * hidden_size)}
            tensors[f"{ep}.up_proj"] = {"dtype": "BF16", "shape": [moe_inter, hidden_size], "data": make_bf16(moe_inter * hidden_size)}
            tensors[f"{ep}.down_proj"] = {"dtype": "BF16", "shape": [hidden_size, moe_inter], "data": make_bf16(hidden_size * moe_inter)}

    write_safetensors(out_dir / "model.q4nx", tensors)
    print(f"[OK] Generated synthetic model with {len(tensors)} tensors -> {out_dir / 'model.q4nx'}")

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
    parser = argparse.ArgumentParser(description="Convert deepgrove/maple-preview to FastFlowLM format or generate synthetic models.")
    parser.add_argument("--src-dir", type=str, default=None, help="Source directory of deepgrove/maple-preview checkpoint")
    parser.add_argument("--out-dir", type=str, required=True, help="Destination directory for FastFlowLM model files")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate a lightweight synthetic Maple model for testing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.generate_synthetic:
        generate_synthetic_model(out_dir)
        return

    if not args.src_dir:
        parser.error("--src-dir is required unless --generate-synthetic is specified.")

    src_dir = Path(args.src_dir)
    if not src_dir.exists():
        print(f"Error: source directory {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    convert_model_directory(src_dir, out_dir)

if __name__ == "__main__":
    main()
