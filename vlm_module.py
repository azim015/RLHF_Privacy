"""
Step 1: VLM for Text Generation
--------------------------------
Wraps a Vision-Language Model (BLIP) to generate textual descriptions
from images guided by privacy-aware prompts.  The module also implements
the privacy-annotated fine-tuning logic (loss penalisation when PII terms
appear) described in the paper.
"""

import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Optional

# PII terms that should NOT appear in generated text
PII_TERMS = [
    "face", "name", "identity", "person named", "man named", "woman named",
    "recognise", "recognize", "identify", "licence plate", "license plate",
    "phone number", "address", "social security", "passport",
]


class PrivacyClassifier:
    """
    Simple keyword-based privacy classifier that returns a penalty score
    in [0, 1] reflecting how much PII is present in a text string.
    A score of 0 means perfectly private; 1 means severe PII exposure.
    """

    def __init__(self, pii_terms: List[str] = PII_TERMS):
        self.pii_terms = [t.lower() for t in pii_terms]

    def score(self, text: str) -> float:
        text_lower = text.lower()
        hits = sum(1 for term in self.pii_terms if term in text_lower)
        return min(hits / max(len(self.pii_terms), 1), 1.0)

    def is_private(self, text: str, threshold: float = 0.05) -> bool:
        return self.score(text) <= threshold


class VLMModule:
    """
    Wraps BLIP for image captioning.  At inference time it accepts a list
    of prompt strings and generates one description per prompt, then picks
    the most privacy-preserving one according to PrivacyClassifier.

    Fine-tuning interface (privacy_finetune_step) allows gradient-based
    training with a PII penalty as described in the paper.
    """

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VLM] Loading {model_name} on {self.device} ...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.privacy_classifier = PrivacyClassifier()
        print("[VLM] Ready.")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, image: Image.Image, prompt: str = "",
                 max_new_tokens: int = 120) -> str:
        """Generate a single caption for *image* conditioned on *prompt*."""
        inputs = self.processor(
            images=image,
            text=prompt if prompt else None,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        # Strip the prompt prefix if it echoes back
        if prompt and caption.lower().startswith(prompt.lower()):
            caption = caption[len(prompt):].strip()
        return caption

    def generate_with_prompts(self, image: Image.Image,
                              prompts: List[str],
                              max_new_tokens: int = 120) -> List[str]:
        """Generate one caption per prompt and return all of them."""
        descriptions = []
        for prompt in prompts:
            desc = self.generate(image, prompt, max_new_tokens)
            descriptions.append(desc)
        return descriptions

    def best_private_description(self, image: Image.Image,
                                 prompts: List[str]) -> str:
        """
        Generate descriptions for every prompt and return the one
        with the lowest PII score (most privacy-preserving).
        """
        descriptions = self.generate_with_prompts(image, prompts)
        scored = [(d, self.privacy_classifier.score(d)) for d in descriptions]
        scored.sort(key=lambda x: x[1])   # ascending PII score
        return scored[0][0]

    # ------------------------------------------------------------------
    # Object extraction loss (paper Eq. 1)  – used during fine-tuning
    # ------------------------------------------------------------------

    @staticmethod
    def object_detection_loss(pred_probs: torch.Tensor,
                              true_probs: torch.Tensor,
                              obj_mask: torch.Tensor) -> torch.Tensor:
        """
        L = Σ_i Σ_c 1^obj_i · (p_i(c) - p̂_i(c))²
        pred_probs : (B, S², C)  predicted class probabilities
        true_probs : (B, S², C)  ground-truth
        obj_mask   : (B, S²)     binary, 1 where an object exists
        """
        diff_sq = (pred_probs - true_probs) ** 2          # (B, S², C)
        mask = obj_mask.unsqueeze(-1).float()              # (B, S², 1)
        loss = (mask * diff_sq).sum()
        return loss

    # ------------------------------------------------------------------
    # Privacy-aware fine-tuning step
    # ------------------------------------------------------------------

    def privacy_finetune_step(self, image: Image.Image,
                              prompt: str,
                              optimizer: torch.optim.Optimizer,
                              pii_penalty_weight: float = 2.0) -> dict:
        """
        One gradient update step that combines:
          - cross-entropy language modelling loss
          - additive PII penalty (soft, proportional to privacy_classifier score)
        Returns a dict with loss components.
        """
        self.model.train()
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        labels = inputs["input_ids"].clone()
        outputs = self.model(**inputs, labels=labels)
        lm_loss = outputs.loss

        # Generate text to evaluate PII (detached – no gradient through this)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=80)
        gen_text = self.processor.decode(gen_ids[0], skip_special_tokens=True)
        pii_score = self.privacy_classifier.score(gen_text)

        # PII penalty as a scalar added to the LM loss
        pii_penalty = torch.tensor(pii_score * pii_penalty_weight,
                                   device=self.device, dtype=lm_loss.dtype)
        total_loss = lm_loss + pii_penalty

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        self.model.eval()

        return {
            "lm_loss": lm_loss.item(),
            "pii_penalty": pii_penalty.item(),
            "total_loss": total_loss.item(),
            "pii_score": pii_score,
            "generated_text": gen_text,
        }

    # ------------------------------------------------------------------
    # Encode text to embeddings via the text encoder part of BLIP
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> np.ndarray:
        """
        Return a numpy embedding for *text* using the BLIP text encoder.
        Shape: (hidden_dim,)
        """
        inputs = self.processor(text=text, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            # BLIP's text encoder is accessible via model.text_encoder
            enc_out = self.model.text_encoder(**inputs)
            emb = enc_out.last_hidden_state[:, 0, :]   # [CLS] token
        return emb.squeeze(0).cpu().numpy()
