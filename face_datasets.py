"""
face_datasets.py
----------------
Loaders for the two face-verification benchmarks used in the paper:

  CFP-FP  – Celebrities in Frontal-Profile: 500 identities, 7000 images,
             frontal vs profile face-verification pairs.
             Download: http://www.cfpw.io/
             Folder layout expected:
               cfp-dataset/
                 Data/Images/<id>/<01..10>.jpg   (frontal: 01-05, profile: 06-10)
                 Protocol/Pair_list_F.txt         (frontal pairs)
                 Protocol/Pair_list_P.txt         (profile pairs)

  AgeDB-30 – Age Database: 570 identities, 12,240 images, ±30-year age gap.
             Download: https://ibug.doc.ic.ac.uk/resources/agedb/
             Folder layout expected:
               AgeDB/
                 <id>_<name>_<age>_<gender>.jpg  (flat directory)
             Pair file (optional): pairs.txt  (LFW-style)

Both datasets share the same sample dict schema so they can be used
interchangeably with the HFR-VLM pipeline:

  {
    "image":       PIL.Image  (face crop, RGB),
    "image_name":  str,
    "image_id":    str,
    "identity_id": str,       # folder / subject name
    "age":         int|None,  # AgeDB only
    "pose":        str|None,  # CFP-FP only  ("frontal" | "profile")
    "pair_label":  int|None,  # 1=same person, 0=different (if pairs loaded)
    "dataset":     str,       # "cfp_fp" | "agedb_30"
  }

Synthetic generators are provided so the pipeline can be tested
without downloading the datasets.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_face(width: int = 112, height: int = 112,
                    seed: int = 0) -> Image.Image:
    """Draw a minimal synthetic face for testing (no real identity)."""
    rng = np.random.default_rng(seed)
    img  = Image.new("RGB", (width, height),
                     tuple(rng.integers(180, 220, 3).tolist()))
    draw = ImageDraw.Draw(img)
    # Skin-tone oval
    skin = tuple(rng.integers(150, 210, 3).tolist())
    draw.ellipse([15, 10, width - 15, height - 10], fill=skin)
    # Eyes
    ey = height // 3
    for ex in [width // 3, 2 * width // 3]:
        draw.ellipse([ex - 8, ey - 5, ex + 8, ey + 5], fill=(30, 30, 30))
    # Nose
    draw.line([(width // 2, ey + 10), (width // 2, ey + 25)],
              fill=(120, 80, 60), width=2)
    # Mouth
    my = 2 * height // 3
    draw.arc([width // 3, my - 8, 2 * width // 3, my + 8],
             start=0, end=180, fill=(160, 60, 60), width=2)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# CFP-FP
# ─────────────────────────────────────────────────────────────────────────────

class CFPFPDataset:
    """
    CFP-FP (Celebrities in Frontal-Profile) dataset loader.

    Parameters
    ----------
    root         : path to cfp-dataset/ root
    split        : 'frontal' | 'profile' | 'both'
    max_samples  : cap total images loaded
    load_pairs   : if True, load verification pairs as well
    """

    FRONTAL_IDX = list(range(1, 6))    # images 01–05 are frontal
    PROFILE_IDX = list(range(6, 11))   # images 06–10 are profile

    def __init__(self,
                 root: str,
                 split: str = "both",
                 max_samples: Optional[int] = None,
                 load_pairs: bool = False):
        self.root        = Path(root)
        self.split       = split
        self.max_samples = max_samples
        self.samples: List[Dict] = []
        self.pairs:   List[Tuple] = []

        self._load_images()
        if load_pairs:
            self._load_pairs()
        print(f"[CFP-FP] Loaded {len(self.samples)} images "
              f"({self.split} split)")

    def _load_images(self):
        images_dir = self.root / "Data" / "Images"
        if not images_dir.exists():
            raise FileNotFoundError(f"CFP-FP images not found at {images_dir}")

        for identity_dir in sorted(images_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            identity_id = identity_dir.name

            for img_file in sorted(identity_dir.glob("*.jpg")):
                idx = int(img_file.stem)
                if self.split == "frontal" and idx not in self.FRONTAL_IDX:
                    continue
                if self.split == "profile" and idx not in self.PROFILE_IDX:
                    continue
                pose = "frontal" if idx in self.FRONTAL_IDX else "profile"

                self.samples.append({
                    "image_path":  str(img_file),
                    "image_name":  f"{identity_id}/{img_file.name}",
                    "image_id":    f"cfp_{identity_id}_{img_file.stem}",
                    "identity_id": identity_id,
                    "pose":        pose,
                    "age":         None,
                    "pair_label":  None,
                    "dataset":     "cfp_fp",
                })

            if self.max_samples and len(self.samples) >= self.max_samples:
                break

    def _load_pairs(self):
        """Load verification pairs from Protocol/Pair_list_{F,P}.txt"""
        for fname in ["Pair_list_F.txt", "Pair_list_P.txt"]:
            pair_file = self.root / "Protocol" / fname
            if not pair_file.exists():
                continue
            with open(pair_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        self.pairs.append((parts[0], parts[1], int(parts[2])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.samples[idx])
        try:
            sample["image"] = Image.open(
                sample["image_path"]).convert("RGB")
        except FileNotFoundError:
            sample["image"] = _synthetic_face(seed=idx)
        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class SyntheticCFPFPDataset:
    """
    Synthetic CFP-FP dataset for testing without downloads.
    Generates fake frontal and profile face images with random identity IDs.
    """

    def __init__(self, num_identities: int = 10,
                 images_per_identity: int = 10):
        self.samples: List[Dict] = []
        for iid in range(num_identities):
            identity_id = f"id_{iid:03d}"
            for img_idx in range(1, images_per_identity + 1):
                pose = "frontal" if img_idx <= 5 else "profile"
                seed = iid * 100 + img_idx
                self.samples.append({
                    "image":       _synthetic_face(seed=seed),
                    "image_path":  f"synthetic/{identity_id}/{img_idx:02d}.jpg",
                    "image_name":  f"{identity_id}/{img_idx:02d}.jpg",
                    "image_id":    f"cfp_{identity_id}_{img_idx:02d}",
                    "identity_id": identity_id,
                    "pose":        pose,
                    "age":         None,
                    "pair_label":  None,
                    "dataset":     "cfp_fp",
                })
        print(f"[Synthetic CFP-FP] Generated {len(self.samples)} images "
              f"({num_identities} identities × {images_per_identity} images)")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def __iter__(self):
        for s in self.samples: yield s


# ─────────────────────────────────────────────────────────────────────────────
# AgeDB-30
# ─────────────────────────────────────────────────────────────────────────────

# Filename pattern: <id>_<name>_<age>_<gender>.jpg
_AGEDB_PATTERN = re.compile(
    r"^(\d+)_(.+?)_(\d+)_(m|f|M|F|male|female)\.jpg$", re.IGNORECASE)


class AgeDB30Dataset:
    """
    AgeDB-30 dataset loader.

    Parameters
    ----------
    root        : path to AgeDB/ directory (flat structure)
    pair_file   : optional LFW-style pairs.txt for verification labels
    max_samples : cap total images
    age_gap_min : only include pairs with age gap ≥ this value (if using pairs)
    """

    def __init__(self,
                 root: str,
                 pair_file: Optional[str] = None,
                 max_samples: Optional[int] = None,
                 age_gap_min: int = 0):
        self.root        = Path(root)
        self.max_samples = max_samples
        self.age_gap_min = age_gap_min
        self.samples: List[Dict] = []
        self.pairs:   List[Dict] = []

        self._load_images()
        if pair_file:
            self._load_pairs(pair_file)
        print(f"[AgeDB-30] Loaded {len(self.samples)} images")

    def _load_images(self):
        if not self.root.exists():
            raise FileNotFoundError(f"AgeDB root not found: {self.root}")

        for img_file in sorted(self.root.glob("*.jpg")):
            m = _AGEDB_PATTERN.match(img_file.name)
            if not m:
                continue
            img_id, name, age, gender = (
                m.group(1), m.group(2), int(m.group(3)), m.group(4).lower())

            self.samples.append({
                "image_path":  str(img_file),
                "image_name":  img_file.name,
                "image_id":    f"agedb_{img_id}",
                "identity_id": name,
                "age":         age,
                "gender":      gender,
                "pose":        None,
                "pair_label":  None,
                "dataset":     "agedb_30",
            })

            if self.max_samples and len(self.samples) >= self.max_samples:
                break

    def _load_pairs(self, pair_file: str):
        """LFW-style pairs.txt: name1 img1 name2 img2 [label]"""
        # Build name→samples index
        name_idx: Dict[str, List[int]] = {}
        for i, s in enumerate(self.samples):
            name_idx.setdefault(s["identity_id"], []).append(i)

        with open(pair_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                n1, i1, n2, i2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                label = int(parts[4]) if len(parts) > 4 else int(n1 == n2)

                # Check age gap if available
                s1 = self._find_sample(n1, i1)
                s2 = self._find_sample(n2, i2)
                if s1 and s2:
                    a1 = s1.get("age") or 0
                    a2 = s2.get("age") or 0
                    if abs(a1 - a2) < self.age_gap_min:
                        continue
                    self.pairs.append({
                        "sample_a": s1, "sample_b": s2,
                        "label": label,
                        "age_gap": abs(a1 - a2),
                    })

    def _find_sample(self, name: str, img_idx: int) -> Optional[Dict]:
        for s in self.samples:
            if s["identity_id"] == name:
                return s
        return None

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.samples[idx])
        try:
            sample["image"] = Image.open(
                sample["image_path"]).convert("RGB")
        except FileNotFoundError:
            sample["image"] = _synthetic_face(seed=idx)
        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------
    # Convenience: group samples by identity for verification tasks
    # ------------------------------------------------------------------

    def group_by_identity(self) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = {}
        for s in self.samples:
            groups.setdefault(s["identity_id"], []).append(s)
        return groups

    def age_range(self) -> Tuple[int, int]:
        ages = [s["age"] for s in self.samples if s["age"] is not None]
        return (min(ages), max(ages)) if ages else (0, 0)


class SyntheticAgeDB30Dataset:
    """
    Synthetic AgeDB-30 dataset for testing without downloads.
    Generates fake face images across a range of simulated ages.
    """

    def __init__(self, num_identities: int = 10,
                 images_per_identity: int = 12):
        self.samples: List[Dict] = []
        genders = ["m", "f"]
        base_ages = list(range(20, 80, 5))  # 20–75

        for iid in range(num_identities):
            name   = f"person_{iid:03d}"
            gender = genders[iid % 2]
            for j in range(images_per_identity):
                age  = base_ages[j % len(base_ages)]
                seed = iid * 200 + j
                self.samples.append({
                    "image":       _synthetic_face(seed=seed),
                    "image_path":  f"synthetic/agedb/{iid:04d}_{name}_{age}_{gender}.jpg",
                    "image_name":  f"{iid:04d}_{name}_{age}_{gender}.jpg",
                    "image_id":    f"agedb_{iid:04d}_{j:02d}",
                    "identity_id": name,
                    "age":         age,
                    "gender":      gender,
                    "pose":        None,
                    "pair_label":  None,
                    "dataset":     "agedb_30",
                })

        print(f"[Synthetic AgeDB-30] Generated {len(self.samples)} images "
              f"({num_identities} identities × {images_per_identity} ages)")

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def __iter__(self):
        for s in self.samples: yield s
