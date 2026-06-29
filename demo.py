"""
demo.py
-------
End-to-end demonstration of the Hierarchical Feedback RL-VLM framework.

Run
---
    python demo.py                        # uses a synthetic test image
    python demo.py --image path/to/img   # uses a real image
    python demo.py --dp_table            # prints the DP trade-off table
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import ModelConfig, PromptConfig
from hierarchical_framework import HierarchicalFeedbackRLVLM
from rag_module import RAGModule


# ---------------------------------------------------------------------------
# Synthetic test image  (cars / road scene)
# ---------------------------------------------------------------------------

def make_synthetic_image(width: int = 320, height: int = 240) -> Image.Image:
    """Draw a simple synthetic road-scene image for testing."""
    img = Image.new("RGB", (width, height), color=(180, 220, 180))
    draw = ImageDraw.Draw(img)

    # Sky
    draw.rectangle([0, 0, width, height // 2], fill=(100, 160, 230))
    # Road
    draw.rectangle([0, height // 2, width, height], fill=(80, 80, 80))
    # Lane marking
    for x in range(0, width, 40):
        draw.rectangle([x, height // 2 + height // 4 - 5,
                        x + 20, height // 2 + height // 4 + 5],
                       fill=(255, 255, 255))
    # Car body
    draw.rectangle([60, height // 2 - 30, 180, height // 2 + 20],
                   fill=(200, 50, 50))
    # Windows
    draw.rectangle([75, height // 2 - 25, 115, height // 2 - 5],
                   fill=(150, 220, 255))
    draw.rectangle([120, height // 2 - 25, 165, height // 2 - 5],
                   fill=(150, 220, 255))
    # Wheels
    draw.ellipse([65, height // 2 + 10, 95, height // 2 + 35], fill=(20, 20, 20))
    draw.ellipse([145, height // 2 + 10, 175, height // 2 + 35], fill=(20, 20, 20))
    return img


# ---------------------------------------------------------------------------
# Differential Privacy trade-off table (paper Table 3)
# ---------------------------------------------------------------------------

def print_dp_table():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Differential Privacy Trade-off (Table 3 replica)   ║")
    print("╠══════════╦══════════╦═════════════╦═══════════════════╣")
    print("║  σ (noise)║  ε (↓)  ║ BLEU (↓)   ║  Privacy Gain (%) ║")
    print("╠══════════╬══════════╬═════════════╬═══════════════════╣")
    for sigma in [0.0, 0.1, 0.2, 0.3]:
        r = RAGModule.privacy_gain(sigma)
        eps_str = "∞" if r["epsilon"] > 1e6 else f"{r['epsilon']:.2f}"
        print(f"║  {sigma:<8.1f}║  {eps_str:<8}║  {r['bleu_score']:<11.3f}║  {r['privacy_gain_pct']:<17.1f}║")
    print("╚══════════╩══════════╩═════════════╩═══════════════════╝\n")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HFR-VLM Privacy Framework Demo")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (default: synthetic)")
    parser.add_argument("--gt", type=str,
                        default="A car is driving on the road.",
                        help="Ground-truth text for reward computation")
    parser.add_argument("--dp_table", action="store_true",
                        help="Print the DP trade-off table and exit")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Gaussian noise σ for Differential Privacy")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    if args.dp_table:
        print_dp_table()
        sys.exit(0)

    # ── Load / create image ──────────────────────────────────────────────────
    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"[Demo] Loaded image: {args.image}  ({image.size})")
    else:
        image = make_synthetic_image()
        print("[Demo] Using synthetic test image.")

    # ── Build framework ──────────────────────────────────────────────────────
    config = ModelConfig(dp_noise_sigma=args.sigma)
    prompt_cfg = PromptConfig()
    framework = HierarchicalFeedbackRLVLM(config, prompt_cfg)

    # ── Run inference ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Running Hierarchical Feedback RL-VLM Inference")
    print("=" * 60)
    result = framework.process_image(image,
                                     ground_truth_text=args.gt,
                                     verbose=True)

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    print("\n📝 Final Privacy-Preserved Description:")
    print(textwrap.fill(result["final_description"], width=70,
                        initial_indent="  ", subsequent_indent="  "))

    print("\n📊 Per-Level Results:")
    for lr in result["level_results"]:
        print(f"\n  ── Level {lr.level} ──")
        print(f"     Prompt   : {lr.selected_prompt}")
        print(f"     Text     : {textwrap.shorten(lr.generated_text, 80)}")
        print(f"     Privacy  : {lr.privacy_score:.4f}  "
              f"(lower = more private)")
        print(f"     Reward   : base={lr.base_reward:.4f}  "
              f"aug={lr.augmented_reward:.4f}")
        if lr.ppo_loss_info:
            print(f"     PPO Loss : {lr.ppo_loss_info.get('total_loss', 'N/A'):.4f}  "
                  f"entropy={lr.ppo_loss_info.get('entropy', 'N/A'):.4f}")

    print("\n🔐 Differential Privacy Metrics:")
    pm = result["privacy_metrics"]
    print(f"     σ (noise)   = {pm['dp_sigma']}")
    print(f"     ε (budget)  = {pm['dp_epsilon']:.3f}  "
          f"(lower = stronger privacy)")
    print(f"     δ           = {pm['dp_delta']}")
    print(f"     PII score   = {pm['final_privacy_score']:.4f}  "
          f"(lower = less PII)")

    print("\n📈 Text Quality Metrics:")
    metrics = framework.evaluate_privacy(args.gt, result["final_description"])
    for k, v in metrics.items():
        print(f"     {k:<25} = {v}")

    print_dp_table()

    # ── Optionally save ──────────────────────────────────────────────────────
    if args.output:
        save_data = {
            "final_description": result["final_description"],
            "privacy_metrics":   result["privacy_metrics"],
            "text_quality":      metrics,
            "level_summaries": [
                {
                    "level":       lr.level,
                    "prompt":      lr.selected_prompt,
                    "text":        lr.generated_text,
                    "reward":      lr.augmented_reward,
                    "pii_score":   lr.privacy_score,
                }
                for lr in result["level_results"]
            ],
        }
        Path(args.output).write_text(json.dumps(save_data, indent=2))
        print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
