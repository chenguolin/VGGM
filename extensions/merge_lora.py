#!/usr/bin/env python3
"""
Merge LoRA weights into base model weights.

This script directly operates on state_dict files without loading the full model,
making it memory-efficient and fast.

LoRA merging formula:
    W_merged = W_base + scale * (lora_B @ lora_A)
    where scale = lora_alpha / lora_rank

Usage:
    python extensions/merge_lora.py \
        --base_weights /path/to/base/model_states.pth \
        --lora_weights /path/to/lora_weights.pth \
        --output_weights /path/to/merged_model_states.pth \
        --lora_alpha 64 \
        --lora_rank 64

    # If lora_alpha is not specified, it defaults to lora_rank (scale = 1.)
    python extensions/merge_lora.py \
        --base_weights base.pth \
        --lora_weights lora.pth \
        --output_weights merged.pth
"""

import argparse
import torch
from collections import defaultdict
from pathlib import Path


def merge_lora_weights(
    base_state_dict: dict,
    lora_state_dict: dict,
    lora_alpha: float = None,
    lora_rank: int = None,
) -> dict:
    """
    Merge LoRA weights into base model weights.

    Args:
        base_state_dict: Base model state dict
        lora_state_dict: LoRA weights state dict
        lora_alpha: LoRA alpha parameter (if None, defaults to lora_rank)
        lora_rank: LoRA rank parameter (if None, inferred from lora_A shape)

    Returns:
        Merged state dict with LoRA weights incorporated
    """
    merged_state_dict = base_state_dict.copy()

    # Group LoRA parameters by their base layer name
    # LoRA parameters are named like: "layer.weight.lora_A" and "layer.weight.lora_B"
    lora_pairs = defaultdict(dict)

    for name, param in lora_state_dict.items():
        if "lora_A" in name or "lora_B" in name:
            # Extract base layer name (remove .lora_A or .lora_B suffix)
            if ".lora_A" in name:
                base_name = name.replace(".lora_A", "")
                lora_pairs[base_name]["A"] = param
            elif ".lora_B" in name:
                base_name = name.replace(".lora_B", "")
                lora_pairs[base_name]["B"] = param

    print(f"Found {len(lora_pairs)} LoRA layer pairs to merge")

    # Infer lora_rank from the first lora_A if not provided
    if lora_rank is None and lora_pairs:
        first_pair = next(iter(lora_pairs.values()))
        if "A" in first_pair:
            lora_rank = first_pair["A"].shape[0]
            print(f"Inferred lora_rank: {lora_rank}")

    # Default lora_alpha to lora_rank if not specified (scale = 1.)
    if lora_alpha is None:
        lora_alpha = lora_rank if lora_rank else 1.
        print(f"Using lora_alpha: {lora_alpha}")

    # Calculate scaling factor
    scale = lora_alpha / lora_rank if lora_rank else 1.
    print(f"LoRA scaling factor: {scale:.4f}")

    # Merge each LoRA pair into the base weights
    merged_count = 0
    for base_name, lora_params in lora_pairs.items():
        if "A" not in lora_params or "B" not in lora_params:
            print(f"Warning: Incomplete LoRA pair for {base_name}, skipping")
            continue

        # Check if base weight exists
        if base_name not in merged_state_dict:
            print(f"Warning: Base weight {base_name} not found in base model, skipping")
            continue

        lora_A = lora_params["A"]  # shape: [rank, in_features]
        lora_B = lora_params["B"]  # shape: [out_features, rank]
        base_weight = merged_state_dict[base_name]

        # Compute LoRA delta: ΔW = scale * (B @ A)
        # Move to same device and dtype as base weight
        lora_A = lora_A.to(device=base_weight.device, dtype=base_weight.dtype)
        lora_B = lora_B.to(device=base_weight.device, dtype=base_weight.dtype)

        delta_weight = scale * (lora_B @ lora_A)

        # Verify shapes match
        if delta_weight.shape != base_weight.shape:
            print(f"Warning: Shape mismatch for {base_name}: "
                  f"base {base_weight.shape} vs delta {delta_weight.shape}, skipping")
            continue

        # Merge: W_merged = W_base + ΔW
        merged_state_dict[base_name] = base_weight + delta_weight
        merged_count += 1

    print(f"Successfully merged {merged_count} LoRA layers")

    return merged_state_dict


def load_state_dict(path: str) -> dict:
    """Load state dict from file."""
    print(f"Loading weights from: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    num_params = sum(p.numel() for p in state_dict.values())
    print(f"  Loaded {len(state_dict)} keys, {num_params / 1e9:.2f}B parameters")

    return state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights into base model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        required=True,
        help="Path to base model weights file (.pth or .pt)"
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA weights file (.pth or .pt)"
    )
    parser.add_argument(
        "--output_weights",
        type=str,
        required=True,
        help="Path to save merged weights"
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha parameter (default: same as lora_rank)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank parameter (default: inferred from lora_A shape)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["raw", "nested"],
        default="raw",
        help="Output format: 'raw' (just state_dict) or 'nested' (wrapped in 'model' key)"
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.base_weights).exists():
        raise FileNotFoundError(f"Base weights not found: {args.base_weights}")
    if not Path(args.lora_weights).exists():
        raise FileNotFoundError(f"LoRA weights not found: {args.lora_weights}")

    # Create output directory if needed
    output_dir = Path(args.output_weights).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("LoRA Weight Merging Tool")
    print("="*70)

    # Load weights
    base_state_dict = load_state_dict(args.base_weights)
    lora_state_dict = load_state_dict(args.lora_weights)

    # Merge
    print("\nMerging LoRA weights...")
    merged_state_dict = merge_lora_weights(
        base_state_dict=base_state_dict,
        lora_state_dict=lora_state_dict,
        lora_alpha=args.lora_alpha,
        lora_rank=args.lora_rank,
    )

    # Save merged weights
    print(f"\nSaving merged weights to: {args.output_weights}")
    if args.output_format == "nested":
        save_dict = {"model": merged_state_dict}
    else:
        save_dict = merged_state_dict

    torch.save(save_dict, args.output_weights)

    # Summary
    print("\n" + "="*70)
    print("✓ Merge completed successfully!")
    print(f"  Output: {args.output_weights}")
    print(f"  Size: {Path(args.output_weights).stat().st_size / 1e9:.2f} GB")
    print(f"  Parameters: {sum(p.numel() for p in merged_state_dict.values()) / 1e9:.2f}B")
    print("="*70)


if __name__ == "__main__":
    main()
