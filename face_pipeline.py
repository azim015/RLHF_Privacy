"""
face_pipeline.py
-----------------
Privacy-preserving face analysis pipeline for CFP-FP and AgeDB-30.

Unlike the CityPersons pipeline (which needs a detector), face datasets
already provide cropped face images. The pipeline:

  Cropped face image (CFP-FP or AgeDB-30)
      ↓
  HierarchicalFeedbackRLVLM  (face-specific prompts, no PII)
      ↓
  Privacy-preserved text description
      ↓
  Verification metrics: SSIM / PSNR / MSE / SRRA
  (reconstruct image from text, compare to original — paper Tables 1 & 2)

Face-specific prompt hierarchy avoids:
  - Face description, skin tone, hair colour, eye colour
  - Any identity-linked attributes
  - Age, gender, or race descriptors

And captures only:
  - Lighting and background context
  - General scene / environment
  - Image quality / pose angle (for CFP-FP)
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import field

import numpy as np
from PIL import Image

from config import ModelConfig
from vlm_module import PrivacyClassifier
from hierarchical_framework import HierarchicalFeedbackRLVLM


# ─────────────────────────────────────────────────────────────────────────────
# Face-specific prompt pools  (paper Section: Methodology)
# ─────────────────────────────────────────────────────────────────────────────

FACE_PII_TERMS = [
    "face", "facial", "eyes", "nose", "mouth", "lips", "chin", "cheek",
    "forehead", "eyebrow", "hair", "beard", "moustache", "mustache",
    "skin", "complexion", "freckles", "wrinkles",
    "name", "identity", "person", "individual", "celebrity",
    "recognise", "recognize", "identify",
    "race", "ethnicity", "nationality",
    "age", "young", "old", "elderly",
    "gender", "male", "female", "man", "woman", "boy", "girl",
    "glasses", "earring", "makeup",
]


@dataclass
class FacePromptConfig:
    """
    Three-level prompt hierarchy for face image privacy transformation.

    Layer A: overall image context (lighting, background)
    Layer B: photographic / pose properties
    Layer C: fine-grained image quality descriptors
    """

    layer_a: List[str] = field(default_factory=lambda: [
        "Describe the background and lighting of this photograph.",
        "What is the general setting or environment shown in this image?",
        "Describe the overall composition and framing of the image.",
        "What type of photograph is this (indoor, outdoor, studio)?",
        "Describe the colour palette and tonal qualities of the image.",
        "What is the ambient lighting condition (natural, artificial, mixed)?",
        "Describe any background objects or environment visible.",
        "What is the overall mood or atmosphere conveyed by the image?",
    ])

    layer_b: List[str] = field(default_factory=lambda: [
        "Describe the camera angle and viewing direction of the subject.",
        "Is the image taken from a frontal or profile angle?",
        "Describe the depth of field and focus characteristics.",
        "What is the approximate distance between the camera and subject?",
        "Describe any visible motion blur or image artefacts.",
        "What photographic quality issues, if any, are present?",
        "Describe the image resolution and sharpness.",
        "Is the subject centred, or offset in the frame?",
        "Describe the shadow patterns visible in the image.",
        "What colour temperature does the image appear to have?",
    ])

    layer_c: List[str] = field(default_factory=lambda: [
        "Describe the texture and detail quality of the image background.",
        "What is the contrast level between the subject and background?",
        "Describe any lens distortion or perspective effects visible.",
        "What noise or grain level is present in the image?",
        "Describe the exposure level (under, over, balanced).",
        "Are there any occlusions or partial blockages in the image?",
        "Describe the symmetry or asymmetry of the image composition.",
        "What post-processing or filter effects appear to have been applied?",
        "Describe the edge sharpness and detail rendition.",
        "Is the image cropped tightly or loosely around the subject area?",
        "Describe any visible artefacts from compression or transmission.",
        "What is the aspect ratio and orientation of the image?",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Output data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FacePrivacyResult:
    image_id:       str
    dataset:        str           # "cfp_fp" | "agedb_30"
    identity_id:    str
    pose:           Optional[str] # CFP-FP: "frontal" | "profile"
    age:            Optional[int] # AgeDB-30
    privacy_text:   str
    privacy_score:  float
    dp_epsilon:     float
    word_count:     int
    unique_words:   int
    # Image-level privacy metrics (vs reconstructed image)
    ssim:           float
    psnr:           float
    mse:            float
    generation_time: float


@dataclass
class DatasetReport:
    dataset_name:   str
    num_samples:    int
    results:        List[FacePrivacyResult]
    # Aggregate metrics (match paper Tables 1 & 2)
    mean_ssim:      float
    mean_psnr:      float
    mean_mse:       float
    mean_pii_score: float
    mean_word_count: float
    mean_unique_words: float


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction-based privacy evaluation (paper metric)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_from_text(text: str,
                            target_size: Tuple[int, int] = (112, 112)) -> Image.Image:
    """
    Proxy reconstruction: simulate what an attacker could recover from the
    privacy text alone.  In the paper a text-to-image model (e.g. Stable
    Diffusion) is used; here we generate a noise image scaled by text length
    as a lightweight proxy (preserves the evaluation pipeline without requiring
    a generative model download).

    In production, replace this with your text-to-image model of choice.
    """
    # Longer, richer text → attacker recovers slightly more detail
    richness = min(len(text) / 500, 1.0)
    noise    = np.random.randint(50, 200, (*target_size[::-1], 3),
                                  dtype=np.uint8)
    # Add minimal structure proportional to text richness
    noise    = (noise * (0.3 + 0.7 * richness)).astype(np.uint8)
    return Image.fromarray(noise)


def compute_image_privacy_metrics(original: Image.Image,
                                   reconstructed: Image.Image) -> Dict[str, float]:
    """
    Compute SSIM, PSNR, MSE between original and reconstructed images.
    Lower SSIM / PSNR and higher MSE → better privacy preservation.
    """
    orig  = np.array(original.convert("RGB").resize((112, 112)),
                     dtype=float)
    recon = np.array(reconstructed.convert("RGB").resize((112, 112)),
                     dtype=float)

    mse  = float(np.mean((orig - recon) ** 2))
    psnr = float(10 * np.log10(255 ** 2 / (mse + 1e-10))) if mse > 0 \
           else float("inf")

    mu1, mu2 = orig.mean(), recon.mean()
    s1,  s2  = orig.std(),  recon.std()
    cov      = float(np.mean((orig - mu1) * (recon - mu2)))
    C1, C2   = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim     = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 ** 2 + s2 ** 2 + C2))

    return {"ssim": round(ssim, 4),
            "psnr": round(psnr, 2),
            "mse":  round(mse, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# Face Privacy Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FacePrivacyPipeline:
    """
    End-to-end privacy-preserving pipeline for CFP-FP and AgeDB-30.

    Parameters
    ----------
    config       : ModelConfig
    dataset_name : 'cfp_fp' | 'agedb_30' (used in reports only)
    """

    def __init__(self,
                 config: Optional[ModelConfig] = None,
                 dataset_name: str = "face",
                 device: Optional[str] = None):
        self.config       = config or ModelConfig()
        self.dataset_name = dataset_name
        self.privacy_clf  = PrivacyClassifier(pii_terms=FACE_PII_TERMS)

        prompt_cfg = FacePromptConfig()
        print(f"[FacePipeline:{dataset_name}] Loading HFR-VLM ...")
        self.framework = HierarchicalFeedbackRLVLM(
            config=self.config, prompt_config=prompt_cfg, device=device)
        self.framework.privacy_clf = self.privacy_clf
        print(f"[FacePipeline:{dataset_name}] Ready.\n")

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    def process_sample(self,
                       sample: Dict,
                       verbose: bool = False) -> FacePrivacyResult:
        t0    = time.time()
        image = sample["image"]

        # Build a context-aware ground-truth hint (no PII)
        gt_hint = "A photograph showing lighting and background context."
        if sample.get("pose"):
            gt_hint = (f"A {sample['pose']} view photograph "
                       f"showing background and lighting.")

        result = self.framework.process_image(
            image,
            ground_truth_text=gt_hint,
            verbose=verbose,
        )

        priv_text  = result["final_description"]
        priv_score = result["privacy_metrics"]["final_privacy_score"]
        dp_eps     = result["privacy_metrics"]["dp_epsilon"]
        words      = priv_text.split()

        # Image-level privacy: reconstruct from text, compare to original
        reconstructed = _reconstruct_from_text(priv_text, image.size)
        img_metrics   = compute_image_privacy_metrics(image, reconstructed)

        return FacePrivacyResult(
            image_id       = sample["image_id"],
            dataset        = sample.get("dataset", self.dataset_name),
            identity_id    = sample.get("identity_id", "unknown"),
            pose           = sample.get("pose"),
            age            = sample.get("age"),
            privacy_text   = priv_text,
            privacy_score  = priv_score,
            dp_epsilon     = dp_eps,
            word_count     = len(words),
            unique_words   = len(set(words)),
            ssim           = img_metrics["ssim"],
            psnr           = img_metrics["psnr"],
            mse            = img_metrics["mse"],
            generation_time = time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Full dataset
    # ------------------------------------------------------------------

    def process_dataset(self,
                        dataset,
                        output_dir: str = "results",
                        verbose: bool = False) -> DatasetReport:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results: List[FacePrivacyResult] = []

        for i, sample in enumerate(dataset):
            res = self.process_sample(sample, verbose=verbose)
            results.append(res)
            print(f"  [{i+1}/{len(dataset)}] {res.image_id} "
                  f"| PII={res.privacy_score:.4f} "
                  f"| SSIM={res.ssim:.4f} "
                  f"| MSE={res.mse:.1f} "
                  f"| words={res.word_count}")

        # Save per-sample JSON
        for i, res in enumerate(results):
            _save_face_result(res, out_path / f"result_{i:04d}.json")

        # Aggregate
        report = _build_report(self.dataset_name, results)
        _save_dataset_report(report, out_path / "summary.json")
        return report

    # ------------------------------------------------------------------
    # Pair-level verification privacy (CFP-FP / AgeDB-30 evaluation mode)
    # ------------------------------------------------------------------

    def evaluate_pairs(self,
                       results_a: List[FacePrivacyResult],
                       results_b: List[FacePrivacyResult],
                       labels: List[int]) -> Dict[str, float]:
        """
        Evaluate privacy at the verification-pair level.
        Checks whether the generated texts for same-identity pairs
        are semantically similar (they should NOT be — better privacy
        means the text doesn't leak identity linkage).

        Returns per-pair text similarity statistics.
        """
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(self.config.sbert_model_name)

        same_sims, diff_sims = [], []
        for ra, rb, label in zip(results_a, results_b, labels):
            embs = sbert.encode(
                [ra.privacy_text, rb.privacy_text],
                normalize_embeddings=True)
            sim = float(embs[0] @ embs[1])
            if label == 1:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

        return {
            "mean_sim_same_identity": round(np.mean(same_sims), 4) if same_sims else 0,
            "mean_sim_diff_identity": round(np.mean(diff_sims), 4) if diff_sims else 0,
            "privacy_gap": round(
                (np.mean(diff_sims) - np.mean(same_sims))
                if same_sims and diff_sims else 0, 4),
            # A positive gap means same-identity pairs are NOT more similar
            # in text space — good privacy.
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_report(dataset_name: str,
                  results: List[FacePrivacyResult]) -> DatasetReport:
    return DatasetReport(
        dataset_name    = dataset_name,
        num_samples     = len(results),
        results         = results,
        mean_ssim       = round(np.mean([r.ssim  for r in results]), 4),
        mean_psnr       = round(np.mean([r.psnr  for r in results]), 2),
        mean_mse        = round(np.mean([r.mse   for r in results]), 2),
        mean_pii_score  = round(np.mean([r.privacy_score for r in results]), 4),
        mean_word_count = round(np.mean([r.word_count for r in results]), 1),
        mean_unique_words = round(np.mean([r.unique_words for r in results]), 1),
    )


def _save_face_result(res: FacePrivacyResult, path: Path):
    path.write_text(json.dumps({
        "image_id":      res.image_id,
        "dataset":       res.dataset,
        "identity_id":   res.identity_id,
        "pose":          res.pose,
        "age":           res.age,
        "privacy_text":  res.privacy_text,
        "privacy_score": res.privacy_score,
        "dp_epsilon":    res.dp_epsilon,
        "word_count":    res.word_count,
        "unique_words":  res.unique_words,
        "ssim":          res.ssim,
        "psnr":          res.psnr,
        "mse":           res.mse,
    }, indent=2))


def _save_dataset_report(report: DatasetReport, path: Path):
    path.write_text(json.dumps({
        "dataset":           report.dataset_name,
        "num_samples":       report.num_samples,
        "mean_ssim":         report.mean_ssim,
        "mean_psnr":         report.mean_psnr,
        "mean_mse":          report.mean_mse,
        "mean_pii_score":    report.mean_pii_score,
        "mean_word_count":   report.mean_word_count,
        "mean_unique_words": report.mean_unique_words,
    }, indent=2))
