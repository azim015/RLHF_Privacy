"""
face_demo.py
-------------
Demo runner for CFP-FP and AgeDB-30 datasets.
Replicates the evaluation protocol from paper Tables 1 & 2.

Usage
-----

# Synthetic (no downloads needed):
python face_demo.py --synthetic

# CFP-FP real dataset:
python face_demo.py --cfp_root /path/to/cfp-dataset --split both

# AgeDB-30 real dataset:
python face_demo.py --agedb_root /path/to/AgeDB

# Both datasets together:
python face_demo.py --synthetic --both

# Replicate paper Tables 1 & 2 (synthetic approximation):
python face_demo.py --synthetic --both --paper_tables
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

from config import ModelConfig
from face_datasets import (
    CFPFPDataset, SyntheticCFPFPDataset,
    AgeDB30Dataset, SyntheticAgeDB30Dataset,
)
from face_pipeline import FacePrivacyPipeline, DatasetReport


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_sample_result(res, idx: int):
    print(f"\n  ── Sample {idx+1} ──────────────────────────────────")
    print(f"     ID       : {res.image_id}")
    print(f"     Dataset  : {res.dataset}")
    if res.pose:
        print(f"     Pose     : {res.pose}")
    if res.age is not None:
        print(f"     Age      : {res.age}")
    print(f"     PII score: {res.privacy_score:.4f}  (↓ better)")
    print(f"     SSIM     : {res.ssim:.4f}  (↓ better privacy)")
    print(f"     PSNR     : {res.psnr:.2f}  (↓ better privacy)")
    print(f"     MSE      : {res.mse:.1f}   (↑ better privacy)")
    print(f"     DP ε     : {res.dp_epsilon:.3f}")
    print(f"     Words    : {res.word_count}  Unique: {res.unique_words}")
    print(f"     Text     :", textwrap.shorten(res.privacy_text, 100))


def print_dataset_report(report: DatasetReport):
    w = 52
    print(f"\n{'═'*w}")
    print(f"  {report.dataset_name.upper()} — Summary ({report.num_samples} samples)")
    print(f"{'═'*w}")
    print(f"  {'Metric':<28} {'Value':>10}  {'Direction'}")
    print(f"  {'─'*48}")
    print(f"  {'Mean SSIM':<28} {report.mean_ssim:>10.4f}  ↓ better")
    print(f"  {'Mean PSNR':<28} {report.mean_psnr:>10.2f}  ↓ better")
    print(f"  {'Mean MSE':<28} {report.mean_mse:>10.1f}  ↑ better")
    print(f"  {'Mean PII score':<28} {report.mean_pii_score:>10.4f}  ↓ better")
    print(f"  {'Mean word count':<28} {report.mean_word_count:>10.1f}  ↑ richer")
    print(f"  {'Mean unique words':<28} {report.mean_unique_words:>10.1f}  ↑ richer")
    dd = report.mean_unique_words / (report.mean_word_count + 1e-8)
    print(f"  {'Detail density':<28} {dd:>10.4f}  ↑ richer")
    print(f"{'═'*w}\n")


def print_paper_table(cfp_report: DatasetReport,
                       agedb_report: DatasetReport):
    """Replicate the layout of Tables 1 & 2 from the paper."""
    print(f"\n{'═'*70}")
    print("  TABLE 1: Privacy Metrics — CFP-FP Dataset")
    print(f"{'═'*70}")
    print(f"  {'Method':<35} {'SSIM↓':>7} {'PSNR↓':>7} {'MSE↑':>8}")
    print(f"  {'─'*68}")
    baselines = [
        ("Ahmad et al.",          0.93, 22.96, 294.0),
        ("AdvFace (Wang et al.)", 0.89, 23.54, 314.7),
        ("FLDATN (Peng et al.)",  0.91, 24.32, 349.1),
        ("PPO-based RL",          0.86, 23.02, 333.7),
    ]
    for name, ssim, psnr, mse in baselines:
        print(f"  {name:<35} {ssim:>7.2f} {psnr:>7.2f} {mse:>8.1f}")
    print(f"  {'─'*68}")
    print(f"  {'Feedback-based RL (Ours)':<35} "
          f"{cfp_report.mean_ssim:>7.4f} "
          f"{cfp_report.mean_psnr:>7.2f} "
          f"{cfp_report.mean_mse:>8.1f}")
    print(f"{'═'*70}")

    print(f"\n{'═'*70}")
    print("  TABLE 2: Privacy Metrics — AgeDB-30 Dataset")
    print(f"{'═'*70}")
    print(f"  {'Method':<35} {'SSIM↓':>7} {'PSNR↓':>7} {'MSE↑':>8}")
    print(f"  {'─'*68}")
    baselines_age = [
        ("Ahmad et al.",          0.93, 25.71, 371.1),
        ("AdvFace (Wang et al.)", 0.92, 23.73, 329.2),
        ("FLDATN (Peng et al.)",  0.89, 24.59, 369.1),
        ("PPO-based",             0.86, 21.98, 355.5),
    ]
    for name, ssim, psnr, mse in baselines_age:
        print(f"  {name:<35} {ssim:>7.2f} {psnr:>7.2f} {mse:>8.1f}")
    print(f"  {'─'*68}")
    print(f"  {'Feedback-based RL (Ours)':<35} "
          f"{agedb_report.mean_ssim:>7.4f} "
          f"{agedb_report.mean_psnr:>7.2f} "
          f"{agedb_report.mean_mse:>8.1f}")
    print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HFR-VLM on CFP-FP / AgeDB-30")
    parser.add_argument("--cfp_root",   type=str, default=None,
                        help="Path to cfp-dataset/ root")
    parser.add_argument("--agedb_root", type=str, default=None,
                        help="Path to AgeDB/ root")
    parser.add_argument("--split",      type=str, default="both",
                        choices=["frontal", "profile", "both"],
                        help="CFP-FP split to use")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Use synthetic data (no downloads needed)")
    parser.add_argument("--both",       action="store_true",
                        help="Run both CFP-FP and AgeDB-30")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Samples per dataset")
    parser.add_argument("--output_dir", type=str, default="face_results",
                        help="Directory for JSON reports")
    parser.add_argument("--sigma",      type=float, default=0.1,
                        help="DP noise sigma")
    parser.add_argument("--paper_tables", action="store_true",
                        help="Print paper-style Table 1 & 2 comparison")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    config = ModelConfig(dp_noise_sigma=args.sigma)

    # Decide what to run
    run_cfp   = args.both or args.cfp_root   or (args.synthetic and not args.agedb_root)
    run_agedb = args.both or args.agedb_root or (args.synthetic and not args.cfp_root)
    if args.both or (not args.cfp_root and not args.agedb_root):
        run_cfp = run_agedb = True

    cfp_report   = None
    agedb_report = None

    # ── CFP-FP ──────────────────────────────────────────────────────────────
    if run_cfp:
        print("\n" + "="*60)
        print("  CFP-FP DATASET")
        print("="*60)

        if args.cfp_root:
            dataset = CFPFPDataset(
                root=args.cfp_root, split=args.split,
                max_samples=args.num_samples)
        else:
            n_ids = max(2, args.num_samples // 10)
            dataset = SyntheticCFPFPDataset(
                num_identities=n_ids, images_per_identity=10)
            # Trim to num_samples
            dataset.samples = dataset.samples[:args.num_samples]

        pipeline = FacePrivacyPipeline(
            config=config, dataset_name="cfp_fp")

        out_cfp = Path(args.output_dir) / "cfp_fp"
        cfp_report = pipeline.process_dataset(
            dataset, output_dir=str(out_cfp), verbose=args.verbose)

        # Print first few sample results
        for i, res in enumerate(cfp_report.results[:3]):
            print_sample_result(res, i)

        print_dataset_report(cfp_report)

    # ── AgeDB-30 ─────────────────────────────────────────────────────────────
    if run_agedb:
        print("\n" + "="*60)
        print("  AgeDB-30 DATASET")
        print("="*60)

        if args.agedb_root:
            dataset = AgeDB30Dataset(
                root=args.agedb_root, max_samples=args.num_samples)
        else:
            n_ids = max(2, args.num_samples // 12)
            dataset = SyntheticAgeDB30Dataset(
                num_identities=n_ids, images_per_identity=12)
            dataset.samples = dataset.samples[:args.num_samples]

        pipeline = FacePrivacyPipeline(
            config=config, dataset_name="agedb_30")

        out_agedb = Path(args.output_dir) / "agedb_30"
        agedb_report = pipeline.process_dataset(
            dataset, output_dir=str(out_agedb), verbose=args.verbose)

        for i, res in enumerate(agedb_report.results[:3]):
            print_sample_result(res, i)

        print_dataset_report(agedb_report)

    # ── Paper tables ─────────────────────────────────────────────────────────
    if args.paper_tables and cfp_report and agedb_report:
        print_paper_table(cfp_report, agedb_report)

    print(f"✅  All results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
