"""
Configuration for the Hierarchical Feedback RL-VLM Framework.
Based on: "Hierarchical Feedback Reinforcement Learning for
Privacy-Preserving Visual-to-Text Transformation in Public Surveillance"
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # VLM
    vlm_model_name: str = "Salesforce/blip-image-captioning-base"

    # SBERT for semantic similarity / reward
    sbert_model_name: str = "all-MiniLM-L6-v2"

    # RL / PPO hyperparameters
    ppo_clip_epsilon: float = 0.2
    ppo_learning_rate: float = 3e-5
    ppo_batch_size: int = 32
    ppo_max_iterations: int = 500
    entropy_start: float = 0.01
    entropy_end: float = 0.001
    entropy_decay_epochs: int = 50

    # Reward coefficients
    alpha: float = 0.7   # semantic similarity weight
    beta: float = 0.3    # visual privacy weight (reconstruction penalty)
    lambda_rag: float = 0.4  # RAG feedback weight

    # Hierarchical iterations
    num_hierarchy_levels: int = 3
    top_k_prompts: int = 3

    # Differential Privacy
    dp_noise_sigma: float = 0.1   # Gaussian noise std for embedding perturbation
    dp_delta: float = 1e-5

    # Policy / value network
    policy_hidden_1: int = 512
    policy_hidden_2: int = 256

    # RAG / FAISS
    rag_top_k: int = 3

    # Embedding dimension (MiniLM-L6)
    embedding_dim: int = 384


@dataclass
class PromptConfig:
    """
    Three-level hierarchical prompt pool.
    Layer A: high-level / broad topic
    Layer B: mid-level / thematic
    Layer C: fine-grained / specific
    """
    layer_a: List[str] = field(default_factory=lambda: [
        "Describe the overall scene in this image.",
        "What is the general environment shown?",
        "Provide a broad summary of what is happening.",
        "Describe the main subject and setting.",
        "What type of location or situation is depicted?",
        "Summarize the key context of this image.",
        "What is the primary activity or event shown?",
        "Describe the scene without identifying any individuals.",
    ])

    layer_b: List[str] = field(default_factory=lambda: [
        "Describe the vehicles present without identifying occupants.",
        "What objects or infrastructure are visible?",
        "Describe any road or traffic conditions shown.",
        "What behavioral context is apparent from the scene?",
        "Describe the environmental conditions (lighting, weather, surroundings).",
        "Identify any safety-relevant elements in the image.",
        "Describe the spatial layout and key structural elements.",
        "What events or interactions are taking place in this scene?",
        "Describe the posture or actions of any figures without revealing identity.",
        "What equipment or devices are visible?",
    ])

    layer_c: List[str] = field(default_factory=lambda: [
        "Describe specific vehicle types, colors, and positions.",
        "What fine-grained details are visible about road markings or signage?",
        "Describe the precise actions or behaviors occurring, omitting faces.",
        "What accessories or objects are held or used, without naming who holds them?",
        "Describe clothing types and colors without identifying the individual.",
        "What specific safety violations or hazards, if any, are observable?",
        "Detail the number of moving versus stationary objects.",
        "Describe the interior layout visible without revealing personal identity.",
        "What precise environmental detail (e.g., time of day, lighting) is apparent?",
        "Describe any text, labels, or markings visible in the scene.",
        "What specific gestures or movements are detectable without identifying a person?",
        "Describe the distance and relative positions of objects in the scene.",
    ])
