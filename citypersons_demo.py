"""
citypersons_demo.py
--------------------
End-to-end demo of the HFR-VLM privacy framework on CityPersons.

Usage
-----

# With real CityPersons dataset:
python citypersons_demo.py \
    --image_root /path/to/leftImg8bit/val \
    --anno_file  /path/to/anno_val.json \
    --num_samples 10

# With synthetic data (no download needed):
python citypersons_demo.py --synthetic --num_samples 5

# On a single image:
python citypersons_demo.py --image path/to/street.jpg

# Save annotated outputs:
python citypersons_demo.py --synthetic --output_dir ./citypersons_results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config import ModelConfig
from citypersons_dataset import CityPersonsDataset, SyntheticCityPersonsDataset
from citypersons_pipeline import CityPersonsPipeline


def print_report_summary(report):
    print(f"\n{'─'*60}")
    print(f"  Image : {report.image_name}")
    print(f"  Dets  : {report.num_detections}  |  Time: {report.total_time:.1f}s")

    if report.scene_text:
        print(f"\n  📷 Scene description (privacy-preserved):")
        import textwrap
        print(textwrap.fill(report.scene_text, width=70,
                            initial_indent="    ",
                            subsequent_indent="    "))

    for ped in report.pedestrians:
        print(f"\n  🚶 Pedestrian #{ped.detection_id + 1}")
        print(f"     BBox      : {ped.bbox}")
        print(f"     Confidence: {ped.confidence:.3f}")
        print(f"     PII score : {ped.privacy_score:.4f}  (lower = better)")
        print(f"     DP ε      : {ped.dp_epsilon:.3f}")
        print(f"     Words     : {ped.word_count}  |  Unique: {ped.unique_words}")
        print(f"     Text      : {ped.privacy_text[:120]}...")

    if report.detection_metrics:
        m = report.detection_metrics
        print(f"\n  📊 Detection vs GT: "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    print(f"{'─'*60}")


def aggregate_stats(reports):
    all_pii     = []
    all_words   = []
    all_unique  = []
    all_dets    = []
    for r in reports:
        all_dets.append(r.num_detections)
        for p in r.pedestrians:
            all_pii.append(p.privacy_score)
            all_words.append(p.word_count)
            all_unique.append(p.unique_words)

    print(f"\n{'='*60}")
    print("  AGGREGATE STATISTICS")
    print(f"{'='*60}")
    print(f"  Images processed      : {len(reports)}")
    print(f"  Total pedestrians     : {sum(all_dets)}")
    print(f"  Avg dets / image      : {np.mean(all_dets):.1f}")
    if all_pii:
        print(f"  Avg PII score         : {np.mean(all_pii):.4f}  (↓ better)")
        print(f"  Avg word count        : {np.mean(all_words):.1f}")
        print(f"  Avg unique words      : {np.mean(all_unique):.1f}")
        detail_density = np.array(all_unique) / (np.array(all_words) + 1e-8)
        print(f"  Avg detail density    : {np.mean(detail_density):.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="HFR-VLM on CityPersons")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Path to Cityscapes leftImg8bit/ root")
    parser.add_argument("--anno_file",  type=str, default=None,
                        help="Path to CityPersons anno_val.json")
    parser.add_argument("--image",      type=str, default=None,
                        help="Run on a single image file")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Use synthetic dataset (no download needed)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of images to process")
    parser.add_argument("--output_dir", type=str, default="citypersons_results",
                        help="Directory to save JSON reports")
    parser.add_argument("--score_thresh", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--sigma",  type=float, default=0.1,
                        help="DP Gaussian noise sigma")
    parser.add_argument("--no_full_image", action="store_true",
                        help="Skip whole-image scene description")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save annotated visualisation images")
    args = parser.parse_args()

    # ── Build pipeline ──────────────────────────────────────────────────────
    config = ModelConfig(dp_noise_sigma=args.sigma)
    pipeline = CityPersonsPipeline(
        config=config,
        score_threshold=args.score_thresh,
        use_full_image=not args.no_full_image,
    )

    # ── Load dataset / image ────────────────────────────────────────────────
    if args.image:
        # Single image mode
        image = Image.open(args.image).convert("RGB")
        report = pipeline.process_image(
            image, image_name=Path(args.image).name, verbose=True)
        print_report_summary(report)
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            out = Path(args.output_dir) / "report_single.json"
            from citypersons_pipeline import _save_report
            _save_report(report, out)
            print(f"\n✅  Report saved → {out}")
        if args.save_vis:
            vis = pipeline.visualise(image, report)
            vis_path = Path(args.output_dir) / "vis_single.jpg"
            vis.save(vis_path)
            print(f"✅  Visualisation saved → {vis_path}")
        return

    if args.synthetic:
        dataset = SyntheticCityPersonsDataset(num_samples=args.num_samples)
    elif args.image_root and args.anno_file:
        dataset = CityPersonsDataset(
            image_root=args.image_root,
            anno_file=args.anno_file,
            max_samples=args.num_samples,
        )
    else:
        print("No dataset source specified. Using synthetic data.\n"
              "Pass --image_root + --anno_file for real CityPersons, "
              "or --image for a single file.\n")
        dataset = SyntheticCityPersonsDataset(num_samples=args.num_samples)

    # ── Run pipeline ─────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    reports = []

    for i, sample in enumerate(dataset):
        report = pipeline.process_image(
            image=sample["image"],
            image_name=sample["image_name"],
            gt_boxes=sample.get("boxes"),
            verbose=False,
        )
        reports.append(report)
        print_report_summary(report)

        # Save JSON
        from citypersons_pipeline import _save_report
        _save_report(report, Path(args.output_dir) / f"report_{i:04d}.json")

        # Save visualisation
        if args.save_vis:
            vis = pipeline.visualise(sample["image"], report)
            vis.save(Path(args.output_dir) / f"vis_{i:04d}.jpg")

    aggregate_stats(reports)
    print(f"✅  Reports saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
