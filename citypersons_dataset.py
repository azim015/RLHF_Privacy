"""
citypersons_dataset.py
----------------------
CityPersons dataset loader compatible with the HFR-VLM privacy framework.

CityPersons is built on top of Cityscapes:
  images/  → leftImg8bit/  (from Cityscapes)
  annotations/ → anno_train.json, anno_val.json  (CityPersons format)

Annotation JSON structure (per image):
  {
    "image_id": ...,
    "image_name": "frankfurt/frankfurt_000000_000294_leftImg8bit.png",
    "annotations": [
      {
        "category_id": 1,          # 1=pedestrian, 2=rider, 3=sitting, 4=unusual, 5=group
        "bbox": [x, y, w, h],      # visible bounding box
        "bbox_full": [x, y, w, h], # full body bbox
        "instance_id": ...,
        "height": ...,             # person height in pixels
      }, ...
    ]
  }

Usage
-----
    ds = CityPersonsDataset(image_root="path/to/leftImg8bit",
                            anno_file="path/to/anno_val.json")
    for sample in ds:
        image       = sample["image"]          # PIL Image
        boxes       = sample["boxes"]          # list of [x,y,w,h]
        categories  = sample["categories"]
        image_name  = sample["image_name"]
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


# Category names matching CityPersons label set
CATEGORY_NAMES = {
    1: "pedestrian",
    2: "rider",
    3: "sitting person",
    4: "person in unusual pose",
    5: "group of people",
    6: "ignore region",
}

# Only evaluate these categories by default (paper standard)
EVAL_CATEGORIES = {1, 2}


class CityPersonsDataset:
    """
    Iterable dataset for CityPersons.

    Parameters
    ----------
    image_root  : root of Cityscapes leftImg8bit/ tree
    anno_file   : path to anno_train.json or anno_val.json
    split       : 'train' | 'val' (used only when anno_file is auto-detected)
    max_samples : cap number of samples (useful for quick experiments)
    min_height  : ignore annotations shorter than this many pixels
    categories  : set of category_ids to keep (default: pedestrian + rider)
    """

    def __init__(self,
                 image_root: str,
                 anno_file: str,
                 max_samples: Optional[int] = None,
                 min_height: int = 50,
                 categories: Optional[set] = None):
        self.image_root  = Path(image_root)
        self.anno_file   = Path(anno_file)
        self.max_samples = max_samples
        self.min_height  = min_height
        self.categories  = categories or EVAL_CATEGORIES

        self.samples: List[Dict] = []
        self._load_annotations()

    # ------------------------------------------------------------------

    def _load_annotations(self):
        with open(self.anno_file) as f:
            data = json.load(f)

        # Support both list-of-dicts and {'annotations': [...]} formats
        records = data if isinstance(data, list) else data.get("annotations", data)

        for record in records:
            image_name = record.get("image_name", record.get("file_name", ""))
            image_path = self.image_root / image_name

            boxes, cats, heights = [], [], []
            for ann in record.get("annotations", []):
                cat = ann.get("category_id", 1)
                if cat not in self.categories:
                    continue
                h = ann.get("height", ann["bbox"][3])
                if h < self.min_height:
                    continue
                boxes.append(ann["bbox"])           # [x, y, w, h]
                cats.append(cat)
                heights.append(h)

            self.samples.append({
                "image_path": str(image_path),
                "image_name": image_name,
                "image_id":   record.get("image_id", image_name),
                "boxes":      boxes,
                "categories": cats,
                "heights":    heights,
            })

            if self.max_samples and len(self.samples) >= self.max_samples:
                break

        print(f"[CityPersons] Loaded {len(self.samples)} images "
              f"from {self.anno_file.name}")

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = dict(self.samples[idx])
        try:
            sample["image"] = Image.open(sample["image_path"]).convert("RGB")
        except FileNotFoundError:
            # Return a synthetic image if the real one isn't present
            sample["image"] = _synthetic_street_scene()
        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def draw_boxes(self, sample: Dict,
                   color: Tuple = (255, 0, 0),
                   width: int = 2) -> Image.Image:
        """Return a copy of the image with GT bounding boxes drawn."""
        img  = sample["image"].copy()
        draw = ImageDraw.Draw(img)
        for (x, y, w, h), cat in zip(sample["boxes"], sample["categories"]):
            draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
            draw.text((x + 2, y + 2), CATEGORY_NAMES.get(cat, str(cat)),
                      fill=color)
        return img

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Simple collator for use with torch DataLoader."""
        return {
            "images":      [s["image"]      for s in batch],
            "boxes":       [s["boxes"]      for s in batch],
            "categories":  [s["categories"] for s in batch],
            "image_names": [s["image_name"] for s in batch],
        }


# ------------------------------------------------------------------
# Synthetic CityPersons-style dataset (for testing without downloads)
# ------------------------------------------------------------------

def _synthetic_street_scene(width: int = 2048,
                              height: int = 1024) -> Image.Image:
    """Draw a minimal synthetic urban street scene."""
    img  = Image.new("RGB", (width, height), (120, 140, 160))
    draw = ImageDraw.Draw(img)

    # Sky gradient approximation
    draw.rectangle([0, 0, width, height // 3], fill=(160, 190, 220))
    # Road
    draw.rectangle([0, height * 2 // 3, width, height], fill=(60, 60, 65))
    # Sidewalk
    draw.rectangle([0, height * 3 // 5, width, height * 2 // 3],
                   fill=(150, 150, 140))
    # Buildings
    for bx in range(0, width, 220):
        bh = np.random.randint(height // 4, height // 2)
        draw.rectangle([bx, height // 3 - bh, bx + 200, height * 3 // 5],
                       fill=(np.random.randint(80, 140),
                             np.random.randint(80, 130),
                             np.random.randint(80, 120)))
    # Pedestrians (simple silhouettes)
    for px in np.random.randint(100, width - 100, size=6):
        py  = height * 3 // 5 - np.random.randint(80, 150)
        ph  = np.random.randint(100, 180)
        pw  = ph // 3
        draw.rectangle([px, py, px + pw, py + ph], fill=(40, 40, 40))
        draw.ellipse([px + pw // 4, py - pw, px + pw * 3 // 4, py],
                     fill=(200, 160, 120))
    return img


class SyntheticCityPersonsDataset:
    """
    Drop-in replacement for CityPersonsDataset when the real dataset
    is not available.  Generates synthetic street scenes with random
    pedestrian bounding boxes, useful for pipeline testing.
    """

    def __init__(self, num_samples: int = 20,
                 img_w: int = 2048, img_h: int = 1024,
                 max_persons: int = 8):
        self.num_samples  = num_samples
        self.img_w, self.img_h = img_w, img_h
        self.max_persons  = max_persons
        self.samples = [self._make_sample(i) for i in range(num_samples)]
        print(f"[SyntheticCityPersons] Generated {num_samples} synthetic samples.")

    def _make_sample(self, idx: int) -> Dict:
        image  = _synthetic_street_scene(self.img_w, self.img_h)
        n_ped  = np.random.randint(1, self.max_persons + 1)
        boxes, cats, heights = [], [], []
        for _ in range(n_ped):
            h  = np.random.randint(80, 300)
            w  = h // 3
            x  = np.random.randint(0, max(1, self.img_w - w))
            y  = np.random.randint(self.img_h // 2, max(self.img_h // 2 + 1,
                                                         self.img_h - h))
            boxes.append([x, y, w, h])
            cats.append(1)   # pedestrian
            heights.append(h)
        return {
            "image":      image,
            "image_path": f"synthetic_{idx:04d}.png",
            "image_name": f"synthetic/synthetic_{idx:04d}.png",
            "image_id":   idx,
            "boxes":      boxes,
            "categories": cats,
            "heights":    heights,
        }

    def __len__(self):  return self.num_samples
    def __getitem__(self, idx): return self.samples[idx]
    def __iter__(self):
        for s in self.samples: yield s
