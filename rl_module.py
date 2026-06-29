"""
Step 2: Prompt Selection Using Reinforcement Learning (PPO)
------------------------------------------------------------
Implements the PPO agent whose:
  - State  : [SBERT embedding of current text, privacy score]
  - Action : index into the active prompt pool (discrete)
  - Reward : computed externally and passed in via update()

Key equations from the paper:
  r_θ  = π_θ(a|s) / π_θ_old(a|s)
  A_t  = Q(s,a) − V(s)
  L_PPO(θ) = E[min(r_θ A_t, clip(r_θ, 1−ε, 1+ε) A_t)]
  L_total  = L_PPO + η · H[π_θ(a|s)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Policy & Value networks
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Maps state → action logits."""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden1: int = 512, hidden2: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw logits


class ValueNetwork(nn.Module):
    """Maps state → scalar value V(s)."""

    def __init__(self, state_dim: int,
                 hidden1: int = 512, hidden2: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# SBERT-based prompt ranker
# ---------------------------------------------------------------------------

class SBERTPromptRanker:
    """
    Uses SBERT cosine similarity to rank prompts against generated text
    (paper Step 2, reward / prompt selection).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SBERT] Loading {model_name} ...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("[SBERT] Ready.")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) array of L2-normalised sentence embeddings."""
        embs = self.model.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=True)
        return embs

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between a (D,) and b (N, D)."""
        a = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return b_norm @ a    # (N,)

    def rank_prompts(self, generated_text: str,
                     prompts: List[str],
                     top_k: int = 3) -> Tuple[List[int], List[float]]:
        """
        Returns indices and scores of the top-k most relevant prompts.
        """
        text_emb  = self.encode([generated_text])[0]   # (D,)
        prompt_embs = self.encode(prompts)              # (N, D)
        scores = self.cosine_similarity(text_emb, prompt_embs)
        ranked_idx = np.argsort(scores)[::-1][:top_k].tolist()
        ranked_scores = scores[ranked_idx].tolist()
        return ranked_idx, ranked_scores


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO agent for hierarchical prompt selection.

    State dimension = SBERT embedding dim (384 for MiniLM) + 1 (privacy score)
    Action space    = number of prompts in the current level's pool
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-5, clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 hidden1: int = 512, hidden2: int = 256,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

        self.policy = PolicyNetwork(state_dim, action_dim, hidden1, hidden2).to(self.device)
        self.value  = ValueNetwork(state_dim, hidden1, hidden2).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

        # Buffers for one update batch
        self._states:      List[np.ndarray] = []
        self._actions:     List[int]        = []
        self._rewards:     List[float]      = []
        self._log_probs:   List[float]      = []
        self._values:      List[float]      = []

        self.iteration = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _policy_dist(self, state: np.ndarray):
        """Return a Categorical distribution over actions."""
        s = self._to_tensor(state).unsqueeze(0)
        logits = self.policy(s)
        return torch.distributions.Categorical(logits=logits)

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Sample an action.
        Returns (action_index, log_prob, value_estimate).
        """
        dist  = self._policy_dist(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).item()

        s = self._to_tensor(state).unsqueeze(0)
        value = self.value(s).item()

        return action.item(), log_prob, value

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, log_prob: float, value: float):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._log_probs.append(log_prob)
        self._values.append(value)

    # ------------------------------------------------------------------
    # PPO update  (paper Eq. for L_PPO and L_total)
    # ------------------------------------------------------------------

    def update(self, decay_entropy: bool = True) -> dict:
        """
        Perform one PPO gradient step using stored transitions.
        Returns dict of loss components.
        """
        if not self._states:
            return {}

        states    = torch.tensor(np.array(self._states),   dtype=torch.float32, device=self.device)
        actions   = torch.tensor(self._actions,             dtype=torch.long,    device=self.device)
        rewards   = torch.tensor(self._rewards,             dtype=torch.float32, device=self.device)
        old_lp    = torch.tensor(self._log_probs,           dtype=torch.float32, device=self.device)

        # Normalise rewards
        if rewards.std() > 1e-6:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Advantage  A_t = Q(s,a) − V(s)
        values   = self.value(states)
        advantage = rewards - values.detach()

        # New log-probs under current policy
        logits   = self.policy(states)
        dist     = torch.distributions.Categorical(logits=logits)
        new_lp   = dist.log_prob(actions)
        entropy  = dist.entropy().mean()

        # Probability ratio  r_θ
        ratio = torch.exp(new_lp - old_lp)

        # Clipped surrogate
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                             1 + self.clip_epsilon) * advantage
        ppo_loss = -torch.min(surr1, surr2).mean()

        # Decay entropy coefficient
        if decay_entropy:
            self.entropy_coef = max(0.001, self.entropy_coef * 0.9995)

        # Value loss (MSE)
        value_loss = F.mse_loss(values, rewards)

        # Total loss
        total_loss = ppo_loss + 0.5 * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()), 0.5)
        self.optimizer.step()

        self.iteration += 1
        self._clear_buffers()

        return {
            "ppo_loss":    ppo_loss.item(),
            "value_loss":  value_loss.item(),
            "entropy":     entropy.item(),
            "total_loss":  total_loss.item(),
            "entropy_coef": self.entropy_coef,
        }

    def _clear_buffers(self):
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()

    # ------------------------------------------------------------------
    # State construction  s_t = [E_VLM(x_t), P_score(x_t)]
    # ------------------------------------------------------------------

    @staticmethod
    def build_state(text_embedding: np.ndarray,
                    privacy_score: float) -> np.ndarray:
        """
        Concatenate SBERT embedding + privacy indicator to form state.
        privacy_score should be in [0,1] (0 = private, 1 = PII-heavy).
        """
        return np.concatenate([text_embedding, [privacy_score]])
