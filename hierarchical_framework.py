"""
Step 4: Hierarchical Prompt Selection and Text Generation Iteration
--------------------------------------------------------------------
Orchestrates the full three-level hierarchy (Algorithm 1 from the paper).
Each level:
  1. Encodes the current text with SBERT.
  2. RL agent selects a prompt from the level's pool.
  3. VLM generates a refined description using the selected prompt.
  4. RAG validates and augments the reward.
  5. PPO update is applied.

Composite reward (paper):
  R_t = α · SimSBERT(x_t, x_gt) − β · SimVAE(I_orig, I_rec)
  R'_t = R_t + λ · F(s_t, a_t)

Output aggregation: concatenate all level descriptions and summarise.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import ModelConfig, PromptConfig
from vlm_module import VLMModule, PrivacyClassifier
from rl_module import PPOAgent, SBERTPromptRanker
from rag_module import RAGModule


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def semantic_similarity(sbert: SentenceTransformer,
                        text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts (SBERT)."""
    embs = sbert.encode([text_a, text_b],
                        convert_to_numpy=True,
                        normalize_embeddings=True)
    return float(embs[0] @ embs[1])


def vae_reconstruction_similarity(orig_image: Image.Image,
                                   rec_image: Optional[Image.Image]) -> float:
    """
    Proxy for SimVAE(I_orig, I_rec).
    In a full deployment a VAE would reconstruct the image from generated text
    and compare pixel-level similarity.  Here we use a fast pixel MSE as proxy:
      similarity = exp(−MSE / 255²)   ∈ (0, 1]
    Lower value ⟹ better privacy (less visual information recoverable).
    """
    if rec_image is None:
        return 0.0   # no reconstruction possible → perfect privacy proxy
    orig_arr = np.array(orig_image.convert("RGB"), dtype=float)
    rec_arr  = np.array(rec_image.resize(orig_image.size).convert("RGB"),
                        dtype=float)
    mse = np.mean((orig_arr - rec_arr) ** 2)
    return float(np.exp(-mse / (255 ** 2)))


def compute_reward(sbert: SentenceTransformer,
                   generated_text: str,
                   ground_truth_text: str,
                   alpha: float = 0.7,
                   beta: float = 0.3,
                   orig_image: Optional[Image.Image] = None,
                   rec_image: Optional[Image.Image] = None) -> float:
    """
    R_t = α · SimSBERT(x_t, x_gt) − β · SimVAE(I_orig, I_rec)
    """
    sim_sbert = semantic_similarity(sbert, generated_text, ground_truth_text)
    sim_vae   = vae_reconstruction_similarity(orig_image, rec_image)
    return alpha * sim_sbert - beta * sim_vae


# ---------------------------------------------------------------------------
# Per-iteration result
# ---------------------------------------------------------------------------

@dataclass
class LevelResult:
    level: int
    selected_prompt: str
    selected_prompt_idx: int
    generated_text: str
    base_reward: float
    augmented_reward: float
    ppo_loss_info: Dict[str, Any]
    privacy_score: float
    sbert_similarity: float


# ---------------------------------------------------------------------------
# Hierarchical Feedback RL-VLM Orchestrator
# ---------------------------------------------------------------------------

class HierarchicalFeedbackRLVLM:
    """
    Full implementation of the paper's framework.

    Usage
    -----
    model = HierarchicalFeedbackRLVLM(config, prompt_config)
    result = model.process_image(image, ground_truth_text="A car on the road.")
    print(result["final_description"])
    """

    def __init__(self,
                 config: ModelConfig = None,
                 prompt_config: PromptConfig = None,
                 device: Optional[str] = None):
        self.config = config or ModelConfig()
        self.prompt_config = prompt_config or PromptConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── Components ──────────────────────────────────────────────────────
        self.vlm = VLMModule(self.config.vlm_model_name, self.device)
        self.privacy_clf = PrivacyClassifier()

        print("[Framework] Loading SBERT ...")
        self.sbert_model = SentenceTransformer(
            self.config.sbert_model_name, device=self.device)
        self.ranker = SBERTPromptRanker(
            self.config.sbert_model_name, self.device)
        # share the same loaded model object to avoid double loading
        self.ranker.model = self.sbert_model

        self.rag = RAGModule(
            sbert_model=self.sbert_model,
            embedding_dim=self.config.embedding_dim,
            top_k=self.config.rag_top_k,
        )

        # State dim = SBERT embedding dim + 1 (privacy score)
        state_dim = self.config.embedding_dim + 1

        # One PPO agent per hierarchy level (they share architecture but
        # have separate weights and action spaces)
        prompt_pools = self._get_prompt_pools()
        self.agents: List[PPOAgent] = [
            PPOAgent(
                state_dim=state_dim,
                action_dim=len(pool),
                lr=self.config.ppo_learning_rate,
                clip_epsilon=self.config.ppo_clip_epsilon,
                entropy_coef=self.config.entropy_start,
                hidden1=self.config.policy_hidden_1,
                hidden2=self.config.policy_hidden_2,
                device=self.device,
            )
            for pool in prompt_pools
        ]

        # Seed the RAG index with all prompts as reference documents
        all_prompts = sum(prompt_pools, [])
        self.rag.add_documents(all_prompts)

        print("[Framework] Initialisation complete.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self,
                      image: Image.Image,
                      ground_truth_text: str = "",
                      reconstructed_image: Optional[Image.Image] = None,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Run the full hierarchical inference loop for a single image.

        Parameters
        ----------
        image               : PIL Image from surveillance camera.
        ground_truth_text   : Reference text for semantic reward.
                              If empty, the first VLM description is used.
        reconstructed_image : If provided, used for SimVAE penalty.
        verbose             : Print per-level logs.

        Returns
        -------
        dict with keys:
          final_description, level_results, privacy_metrics, dp_embedding
        """
        prompt_pools = self._get_prompt_pools()
        level_results: List[LevelResult] = []
        all_descriptions: List[str] = []

        # ── Bootstrap: VLM generates initial description (no prompt) ────────
        current_text = self.vlm.generate(image, prompt="Describe this scene.")
        all_descriptions.append(current_text)
        if not ground_truth_text:
            ground_truth_text = current_text   # self-referential baseline

        if verbose:
            print(f"\n[Hierarchy] Initial description:\n  {current_text}\n")

        # ── Three-level iterative refinement (Algorithm 1) ───────────────────
        for level_idx, pool in enumerate(prompt_pools):
            level_num = level_idx + 1
            if verbose:
                print(f"[Level {level_num}] Processing ({len(pool)} prompts in pool)...")

            # Step 1 – Build state  s_t = [E_SBERT(x_t), P_score(x_t)]
            text_emb = self.sbert_model.encode(
                [current_text], convert_to_numpy=True, normalize_embeddings=True)[0]
            privacy_score = self.privacy_clf.score(current_text)

            # Apply Differential Privacy noise to embedding
            noisy_emb = RAGModule.apply_dp_noise(text_emb, self.config.dp_noise_sigma)
            state = PPOAgent.build_state(noisy_emb, privacy_score)

            # Step 2 – RL agent selects prompt action
            agent = self.agents[level_idx]
            action, log_prob, value_est = agent.select_action(state)
            selected_prompt = pool[action]

            if verbose:
                print(f"  Selected prompt (action={action}):\n  → {selected_prompt}")

            # Step 3 – VLM generates refined description with selected prompt
            refined_text = self.vlm.generate(image, prompt=selected_prompt)

            # Also run SBERT-based top-k selection for additional enrichment
            top_k_idx, top_k_scores = self.ranker.rank_prompts(
                refined_text, pool, top_k=self.config.top_k_prompts)
            if verbose:
                print(f"  SBERT top-{self.config.top_k_prompts} prompt indices: "
                      f"{top_k_idx} (scores: {[f'{s:.3f}' for s in top_k_scores]})")

            # Further refine using top-1 SBERT prompt if different from RL's choice
            if top_k_idx and top_k_idx[0] != action:
                sbert_prompt  = pool[top_k_idx[0]]
                sbert_text    = self.vlm.generate(image, prompt=sbert_prompt)
                # Pick the more privacy-preserving of the two
                if (self.privacy_clf.score(sbert_text) <
                        self.privacy_clf.score(refined_text)):
                    refined_text = sbert_text

            # Step 4 – Compute base reward  R_t = α·SimSBERT − β·SimVAE
            base_reward = compute_reward(
                self.sbert_model,
                generated_text=refined_text,
                ground_truth_text=ground_truth_text,
                alpha=self.config.alpha,
                beta=self.config.beta,
                orig_image=image,
                rec_image=reconstructed_image,
            )

            # Step 4 cont. – RAG augments reward  R'_t = R_t + λ·F(s,a)
            aug_reward = self.rag.augment_reward(
                base_reward, refined_text, selected_prompt,
                lambda_rag=self.config.lambda_rag)

            if verbose:
                print(f"  Base reward: {base_reward:.4f} | "
                      f"Augmented reward: {aug_reward:.4f}")

            # Step 5 – Store transition and update PPO
            agent.store_transition(state, action, aug_reward, log_prob, value_est)
            ppo_info = agent.update(decay_entropy=True)

            # Update RAG index with newly generated text for next iteration
            self.rag.add_documents([refined_text])

            # Advance current_text for next level
            current_text = refined_text
            all_descriptions.append(refined_text)

            level_results.append(LevelResult(
                level=level_num,
                selected_prompt=selected_prompt,
                selected_prompt_idx=action,
                generated_text=refined_text,
                base_reward=base_reward,
                augmented_reward=aug_reward,
                ppo_loss_info=ppo_info,
                privacy_score=self.privacy_clf.score(refined_text),
                sbert_similarity=semantic_similarity(
                    self.sbert_model, refined_text, ground_truth_text),
            ))

        # ── Aggregation: concatenate + lightweight summarisation ─────────────
        final_description = self._aggregate(all_descriptions)

        if verbose:
            print(f"\n[Framework] Final description:\n  {final_description}\n")

        # ── DP embedding for storage / transmission ──────────────────────────
        final_emb = self.sbert_model.encode(
            [final_description], convert_to_numpy=True, normalize_embeddings=True)[0]
        dp_emb = RAGModule.apply_dp_noise(final_emb, self.config.dp_noise_sigma)
        eps = RAGModule.compute_epsilon(sigma=self.config.dp_noise_sigma)

        return {
            "final_description":   final_description,
            "all_descriptions":    all_descriptions,
            "level_results":       level_results,
            "privacy_metrics": {
                "final_privacy_score": self.privacy_clf.score(final_description),
                "dp_epsilon":          eps,
                "dp_delta":            self.config.dp_delta,
                "dp_sigma":            self.config.dp_noise_sigma,
            },
            "dp_embedding": dp_emb,
        }

    # ------------------------------------------------------------------
    # Training loop  (Algorithm 1 extended)
    # ------------------------------------------------------------------

    def train(self,
              images: List[Image.Image],
              ground_truth_texts: Optional[List[str]] = None,
              num_epochs: int = 3,
              verbose: bool = True) -> List[Dict]:
        """
        Train the PPO agents on a list of images for *num_epochs* passes.
        """
        if ground_truth_texts is None:
            ground_truth_texts = [""] * len(images)

        history = []
        for epoch in range(1, num_epochs + 1):
            epoch_rewards = []
            for i, (img, gt) in enumerate(zip(images, ground_truth_texts)):
                result = self.process_image(img, gt, verbose=False)
                rewards = [lr.augmented_reward for lr in result["level_results"]]
                epoch_rewards.extend(rewards)
                if verbose:
                    print(f"  Epoch {epoch} | Image {i+1}/{len(images)} | "
                          f"Avg reward: {np.mean(rewards):.4f}")

            entry = {"epoch": epoch, "mean_reward": np.mean(epoch_rewards)}
            history.append(entry)
            if verbose:
                print(f"[Train] Epoch {epoch} complete | "
                      f"Mean reward: {entry['mean_reward']:.4f}\n")
        return history

    # ------------------------------------------------------------------
    # Evaluation metrics (paper Table 1 / 2 style)
    # ------------------------------------------------------------------

    def evaluate_privacy(self, original_text: str,
                         generated_text: str) -> Dict[str, float]:
        """
        Compute text-level privacy and quality metrics.
        For image-level metrics (SSIM, PSNR, MSE) pass images to
        evaluate_image_privacy() below.
        """
        sim = semantic_similarity(self.sbert_model,
                                  original_text, generated_text)
        pii = self.privacy_clf.score(generated_text)

        words = generated_text.split()
        unique_words = set(words)
        detail_density = len(unique_words) / max(len(words), 1)

        return {
            "semantic_similarity": round(sim, 4),
            "pii_score":           round(pii, 4),
            "word_count":          len(words),
            "unique_word_count":   len(unique_words),
            "detail_density":      round(detail_density, 4),
        }

    @staticmethod
    def evaluate_image_privacy(original: np.ndarray,
                                reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Compute SSIM, PSNR, MSE between original and reconstructed images.
        Arrays should be uint8 H×W×3.
        """
        orig  = original.astype(float)
        recon = reconstructed.astype(float)
        mse   = float(np.mean((orig - recon) ** 2))
        psnr  = 10 * np.log10(255 ** 2 / (mse + 1e-10)) if mse > 0 else float("inf")

        # Simplified SSIM (luminance + contrast + structure)
        mu1, mu2 = orig.mean(), recon.mean()
        s1,  s2  = orig.std(),  recon.std()
        cov      = float(np.mean((orig - mu1) * (recon - mu2)))
        C1, C2   = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        ssim     = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 ** 2 + s2 ** 2 + C2))

        return {"ssim": round(ssim, 4), "psnr": round(psnr, 2), "mse": round(mse, 2)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_prompt_pools(self) -> List[List[str]]:
        return [
            self.prompt_config.layer_a,
            self.prompt_config.layer_b,
            self.prompt_config.layer_c,
        ]

    @staticmethod
    def _aggregate(descriptions: List[str]) -> str:
        """
        Aggregate descriptions from all hierarchy levels.
        Strategy: deduplicate sentences, then join.
        """
        seen = set()
        sentences = []
        for desc in descriptions:
            for sent in desc.replace(".", ".\n").split("\n"):
                sent = sent.strip().rstrip(".")
                if sent and sent.lower() not in seen:
                    seen.add(sent.lower())
                    sentences.append(sent)
        return ". ".join(sentences) + "."
