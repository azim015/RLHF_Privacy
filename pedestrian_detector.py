"""
pedestrian_detector.py
-----------------------
Pedestrian detection module that wraps a torchvision Faster R-CNN model
pre-trained on COCO and fine-tuned for pedestrian detection (person class).

Responsibilities
----------------
1. Run inference on a PIL image → list of detections [x,y,w,h,score].
2. Crop each detected pedestrian region.
3. Apply privacy blurring to detected regions (baseline comparison).
4. Feed cropped regions into the HFR-VLM framework for privacy-preserving
   text generation instead of storing/transmitting the image patches.

Detection output format (per detection):
  {
    "bbox":     [x, y, w, h],    # pixel coords
    "score":    float,            # confidence
    "category": "pedestrian",
    "crop":     PIL.Image,        # cropped region
  }
"""

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter
from torchvision.transforms import functional as TF
from typing import List, Dict, Optional, Tuple


# COCO person class index
PERSON_CLASS_ID = 1


class PedestrianDetector:
    """
    Wraps torchvision Faster R-CNN (ResNet-50 FPN) for pedestrian detection.

    On CityPersons the model is used in a zero-shot manner (COCO pre-trained)
    and filters only the PERSON class.  For best results fine-tune on the
    CityPersons training split.
    """

    def __init__(self,
                 score_threshold: float = 0.5,
                 nms_iou_threshold: float = 0.4,
                 device: Optional[str] = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold  = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

        print(f"[Detector] Loading Faster R-CNN on {self.device} ...")
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()
        print("[Detector] Ready.")

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Detect pedestrians in *image*.
        Returns list of detection dicts with keys:
          bbox, score, category, crop
        """
        tensor = TF.to_tensor(image).to(self.device)
        outputs = self.model([tensor])[0]

        boxes  = outputs["boxes"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        # Keep only PERSON class above threshold
        mask = (labels == PERSON_CLASS_ID) & (scores >= self.score_threshold)
        boxes  = boxes[mask]
        scores = scores[mask]

        # NMS (torchvision already applies NMS internally, but we re-apply
        # with our custom threshold for tighter control)
        if len(boxes) > 0:
            keep = torchvision.ops.nms(
                torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(scores, dtype=torch.float32),
                self.nms_iou_threshold,
            ).numpy()
            boxes  = boxes[keep]
            scores = scores[keep]

        detections = []
        w_img, h_img = image.size
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            # Clip to image bounds
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            detections.append({
                "bbox":     [x1, y1, x2 - x1, y2 - y1],   # [x,y,w,h]
                "score":    float(score),
                "category": "pedestrian",
                "crop":     crop,
            })

        return detections

    def detect_batch(self, images: List[Image.Image]) -> List[List[Dict]]:
        return [self.detect(img) for img in images]

    # ------------------------------------------------------------------
    # Privacy blurring (baseline / comparison method)
    # ------------------------------------------------------------------

    @staticmethod
    def blur_detections(image: Image.Image,
                        detections: List[Dict],
                        radius: int = 15) -> Image.Image:
        """
        Apply Gaussian blur over each detected pedestrian region.
        This is the traditional privacy baseline the paper argues against.
        Returns the image with blurred pedestrian regions.
        """
        img_out = image.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            region = img_out.crop((x, y, x + w, y + h))
            blurred = region.filter(ImageFilter.GaussianBlur(radius=radius))
            img_out.paste(blurred, (x, y))
        return img_out

    # ------------------------------------------------------------------
    # Metrics: IoU and detection evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def iou(box_a: List[float], box_b: List[float]) -> float:
        """
        Compute IoU between two [x,y,w,h] boxes.
        """
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        # Convert to x1y1x2y2
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx); inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter   = inter_w * inter_h

        union = aw * ah + bw * bh - inter
        return inter / (union + 1e-8)

    def evaluate(self, gt_boxes: List[List[float]],
                 pred_detections: List[Dict],
                 iou_threshold: float = 0.5) -> Dict:
        """
        Compute per-image TP/FP/FN counts.
        Returns dict with tp, fp, fn, precision, recall.
        """
        pred_boxes = [d["bbox"] for d in pred_detections]
        matched_gt = set()
        tp = fp = 0

        for pb in pred_boxes:
            best_iou, best_gt = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = self.iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(2 * precision * recall /
                                   (precision + recall + 1e-8), 4)}
