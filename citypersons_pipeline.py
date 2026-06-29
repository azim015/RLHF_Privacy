"""
citypersons_pipeline.py
------------------------
End-to-end pipeline:

  CityPersons image
      ↓
  PedestrianDetector  (Faster R-CNN)
      ↓ detections [bbox, score, crop]
  Per-crop → HierarchicalFeedbackRLVLM
      ↓ privacy-preserved text per pedestrian
  Scene-level aggregation
      ↓
  Final structured report (no image stored)

The pipeline replaces raw image storage with semantically rich text,
implementing the paper's core idea for pedestrian surveillance data.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import ModelConfig
from citypersons_dataset import (
    CityPersonsDataset, SyntheticCityPersonsDataset, CATEGORY_NAMES)
from citypersons_prompts import CityPersonsPromptConfig, PEDESTRIAN_PII_TERMS
from pedestrian_detector import PedestrianDetector
from vlm_module import PrivacyClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Lazy import of the heavy VLM framework (avoids loading unless needed)
# ─────────────────────────────────────────────────────────────────────────────

def _load_framework(config, prompt_cfg):
    from hierarchical_framework import HierarchicalFeedbackRLVLM
    return HierarchicalFeedbackRLVLM(config, prompt_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Output data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PedestrianDescription:
    detection_id:    int
    bbox:            List[int]           # [x, y, w, h]
    confidence:      float
    privacy_text:    str                 # HFR-VLM output
    privacy_score:   float               # lower = more private
    dp_epsilon:      float
    word_count:      int
    unique_words:    int
    generation_time: float               # seconds


@dataclass
class SceneReport:
    image_name:       str
    num_detections:   int
    scene_text:       str                # whole-image description
    pedestrians:      List[PedestrianDescription]
    detection_metrics: Dict              # TP/FP/FN vs GT
    total_time:       float


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class CityPersonsPipeline:
    """
    Full privacy-preserving pedestrian analysis pipeline.

    Parameters
    ----------
    config            : ModelConfig (HFR-VLM hyperparameters)
    score_threshold   : minimum detection confidence to keep
    use_full_image    : if True, also run HFR-VLM on the whole image
                        in addition to per-pedestrian crops
    max_pedestrians   : cap per-image detections (speed vs coverage trade-off)
    """

    def __init__(self,
                 config: Optional[ModelConfig] = None,
                 score_threshold: float = 0.5,
                 use_full_image: bool = True,
                 max_pedestrians: int = 10,
                 device: Optional[str] = None):
        self.config           = config or ModelConfig()
        self.score_threshold  = score_threshold
        self.use_full_image   = use_full_image
        self.max_pedestrians  = max_pedestrians

        prompt_cfg = CityPersonsPromptConfig()

        # Extend PII term list with pedestrian-specific terms
        self.privacy_clf = PrivacyClassifier(pii_terms=PEDESTRIAN_PII_TERMS)

        print("[Pipeline] Loading pedestrian detector ...")
        self.detector = PedestrianDetector(
            score_threshold=score_threshold, device=device)

        print("[Pipeline] Loading HFR-VLM framework ...")
        self.framework = _load_framework(self.config, prompt_cfg)

        # Share privacy classifier with framework
        self.framework.privacy_clf = self.privacy_clf
        print("[Pipeline] Ready.\n")

    # ------------------------------------------------------------------
    # Process a single image
    # ------------------------------------------------------------------

    def process_image(self,
                      image: Image.Image,
                      image_name: str = "unknown",
                      gt_boxes: Optional[List[List[int]]] = None,
                      verbose: bool = True) -> SceneReport:
        t0 = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Processing: {image_name}")
            print(f"{'='*60}")

        # ── Step 1: Detect pedestrians ──────────────────────────────────────
        detections = self.detector.detect(image)
        # Sort by confidence, cap count
        detections.sort(key=lambda d: d["score"], reverse=True)
        detections = detections[:self.max_pedestrians]

        if verbose:
            print(f"[Detect] Found {len(detections)} pedestrian(s) "
                  f"(score ≥ {self.score_threshold})")

        # ── Step 2: Scene-level description ─────────────────────────────────
        scene_text = ""
        if self.use_full_image:
            if verbose:
                print("[Scene] Running HFR-VLM on full image ...")
            scene_result = self.framework.process_image(
                image,
                ground_truth_text="An urban street scene with pedestrians.",
                verbose=False,
            )
            scene_text = scene_result["final_description"]
            if verbose:
                print(f"[Scene] → {scene_text[:120]}...")

        # ── Step 3: Per-pedestrian privacy description ───────────────────────
        pedestrian_descriptions: List[PedestrianDescription] = []

        for i, det in enumerate(detections):
            if verbose:
                print(f"\n[Ped {i+1}/{len(detections)}] "
                      f"bbox={det['bbox']} score={det['score']:.3f}")

            t_ped = time.time()
            crop  = det["crop"]

            # Run HFR-VLM on the cropped pedestrian region
            ped_result = self.framework.process_image(
                crop,
                ground_truth_text=(
                    "A person walking in an urban environment."),
                verbose=False,
            )
            priv_text  = ped_result["final_description"]
            priv_score = ped_result["privacy_metrics"]["final_privacy_score"]
            dp_eps     = ped_result["privacy_metrics"]["dp_epsilon"]
            words      = priv_text.split()

            if verbose:
                print(f"  Privacy text: {priv_text[:100]}...")
                print(f"  PII score:    {priv_score:.4f}  "
                      f"DP ε: {dp_eps:.3f}")

            pedestrian_descriptions.append(PedestrianDescription(
                detection_id=i,
                bbox=det["bbox"],
                confidence=det["score"],
                privacy_text=priv_text,
                privacy_score=priv_score,
                dp_epsilon=dp_eps,
                word_count=len(words),
                unique_words=len(set(words)),
                generation_time=time.time() - t_ped,
            ))

        # ── Step 4: Detection evaluation vs GT ──────────────────────────────
        det_metrics = {}
        if gt_boxes is not None:
            det_metrics = self.detector.evaluate(gt_boxes, detections)
            if verbose:
                print(f"\n[Eval] P={det_metrics['precision']:.3f}  "
                      f"R={det_metrics['recall']:.3f}  "
                      f"F1={det_metrics['f1']:.3f}")

        total_time = time.time() - t0
        if verbose:
            print(f"\n[Pipeline] Done in {total_time:.1f}s")

        return SceneReport(
            image_name=image_name,
            num_detections=len(detections),
            scene_text=scene_text,
            pedestrians=pedestrian_descriptions,
            detection_metrics=det_metrics,
            total_time=total_time,
        )

    # ------------------------------------------------------------------
    # Process full dataset
    # ------------------------------------------------------------------

    def process_dataset(self,
                        dataset,
                        output_dir: str = "results",
                        verbose: bool = False) -> List[SceneReport]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        reports, all_tp, all_fp, all_fn = [], 0, 0, 0

        for i, sample in enumerate(dataset):
            report = self.process_image(
                image=sample["image"],
                image_name=sample["image_name"],
                gt_boxes=sample.get("boxes"),
                verbose=verbose,
            )
            reports.append(report)

            m = report.detection_metrics
            all_tp += m.get("tp", 0)
            all_fp += m.get("fp", 0)
            all_fn += m.get("fn", 0)

            # Save per-image report
            report_path = out_path / f"report_{i:04d}.json"
            _save_report(report, report_path)

            print(f"[{i+1}/{len(dataset)}] {sample['image_name']} "
                  f"| dets={report.num_detections} "
                  f"| {report.total_time:.1f}s")

        # Dataset-level metrics
        prec = all_tp / (all_tp + all_fp + 1e-8)
        rec  = all_tp / (all_tp + all_fn + 1e-8)
        print(f"\n{'='*50}")
        print(f"  Dataset Results ({len(reports)} images)")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {2*prec*rec/(prec+rec+1e-8):.4f}")
        print(f"  Avg dets  : "
              f"{np.mean([r.num_detections for r in reports]):.1f}")
        print(f"  Avg time  : "
              f"{np.mean([r.total_time for r in reports]):.1f}s")
        print(f"{'='*50}\n")

        return reports

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def visualise(self, image: Image.Image,
                  report: SceneReport,
                  show_text: bool = True) -> Image.Image:
        """
        Draw detection boxes and privacy text overlays on the image.
        Returns annotated PIL Image.
        """
        vis = image.copy()
        draw = ImageDraw.Draw(vis)

        for ped in report.pedestrians:
            x, y, w, h = ped.bbox
            colour = (0, 200, 100)   # green = privacy-preserved
            draw.rectangle([x, y, x + w, y + h],
                           outline=colour, width=3)
            draw.rectangle([x, y, x + w, y + 18],
                           fill=(0, 0, 0, 180))
            label = (f"PED #{ped.detection_id+1} "
                     f"conf={ped.confidence:.2f} "
                     f"PII={ped.privacy_score:.2f}")
            draw.text((x + 3, y + 2), label, fill=colour)

            if show_text and ped.privacy_text:
                # Wrap text below the box
                snippet = ped.privacy_text[:80] + "..."
                draw.text((x + 3, y + h + 4), snippet,
                          fill=(255, 255, 200))

        return vis


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_report(report: SceneReport, path: Path):
    data = {
        "image_name":        report.image_name,
        "num_detections":    report.num_detections,
        "scene_text":        report.scene_text,
        "detection_metrics": report.detection_metrics,
        "total_time":        report.total_time,
        "pedestrians": [
            {
                "id":           p.detection_id,
                "bbox":         p.bbox,
                "confidence":   p.confidence,
                "privacy_text": p.privacy_text,
                "privacy_score": p.privacy_score,
                "dp_epsilon":   p.dp_epsilon,
                "word_count":   p.word_count,
                "unique_words": p.unique_words,
            }
            for p in report.pedestrians
        ],
    }
    path.write_text(json.dumps(data, indent=2))
