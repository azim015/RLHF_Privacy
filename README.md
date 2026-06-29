# HFR-VLM: Hierarchical Feedback RL-VLM Privacy Framework

Python implementation of the paper:
> *"Hierarchical Feedback Reinforcement Learning for Privacy-Preserving
> Visual-to-Text Transformation in Public Surveillance"*

---

## Overview

The framework converts surveillance images into **privacy-preserved textual
descriptions** by combining:

| Component | Role | Paper Section |
|-----------|------|---------------|
| **VLM** (BLIP) | Image → text generation with PII-penalised fine-tuning | Step 1 |
| **PPO Agent** | Hierarchical prompt selection via RL | Step 2 |
| **SBERT** | Semantic similarity scoring & prompt ranking | Step 2 |
| **RAG** (FAISS) | External feedback & augmented reward via MIPS | Step 3 |
| **DP Mechanism** | Gaussian noise on embeddings for (ε,δ)-DP | Step 3 |
| **Hierarchical Loop** | 3-level iterative refinement (broad→thematic→specific) | Step 4 |

---

## Project Structure

```
hfrl_vlm/
├── config.py               # ModelConfig & PromptConfig dataclasses
├── vlm_module.py           # Step 1 – BLIP VLM wrapper + PrivacyClassifier
├── rl_module.py            # Step 2 – PPOAgent + SBERTPromptRanker
├── rag_module.py           # Step 3 – RAGModule (FAISS MIPS + DP noise)
├── hierarchical_framework.py  # Step 4 – Full orchestration (Algorithm 1)
├── demo.py                 # End-to-end demo script
├── tests.py                # Unit tests
└── README.md
```

---

## Installation

```bash
pip install torch torchvision transformers sentence-transformers faiss-cpu pillow numpy tqdm
```

---

## Quick Start

```python
from PIL import Image
from config import ModelConfig, PromptConfig
from hierarchical_framework import HierarchicalFeedbackRLVLM

# Load any surveillance image
image = Image.open("camera_frame.jpg").convert("RGB")

# Build the framework
framework = HierarchicalFeedbackRLVLM(ModelConfig(), PromptConfig())

# Run inference
result = framework.process_image(
    image,
    ground_truth_text="A vehicle on the road.",
)

print(result["final_description"])
print(result["privacy_metrics"])
```

---

## Demo Script

```bash
# Synthetic image test
python demo.py

# Real image
python demo.py --image camera_frame.jpg --gt "A car at a junction."

# Print DP trade-off table (replicates paper Table 3)
python demo.py --dp_table

# Save JSON results
python demo.py --image frame.jpg --output results.json
```

---

## Run Tests

```bash
python tests.py
```

---

## Key Equations Implemented

### Step 2 – PPO Objective
```
L_PPO(θ) = E[min(r_θ · A_t, clip(r_θ, 1−ε, 1+ε) · A_t)]
L_total   = L_PPO + η · H[π_θ(a|s)]

r_θ = π_θ(a|s) / π_θ_old(a|s)
A_t = Q(s,a) − V(s)
```

### Step 3 – Composite Reward
```
R_t  = α · SimSBERT(x_t, x_gt) − β · SimVAE(I_orig, I_rec)
R'_t = R_t + λ · F(s_t, a_t)          # RAG-augmented
```

### Step 3 – Differential Privacy
```
Ẽ_VLM(x_t) = E_VLM(x_t) + N(0, σ²)
ε = Δ₂f / σ,   δ < 10⁻⁵
```

### Step 4 – State Representation
```
s_t = [E_SBERT(x_t), P_score(x_t)]
```

---

## Configuration

Edit `config.py` to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.7 | Semantic similarity reward weight |
| `beta` | 0.3 | Visual privacy penalty weight |
| `lambda_rag` | 0.4 | RAG feedback influence |
| `dp_noise_sigma` | 0.1 | DP Gaussian noise level |
| `ppo_clip_epsilon` | 0.2 | PPO clipping range |
| `entropy_start` | 0.01 | Initial entropy coefficient |
| `top_k_prompts` | 3 | SBERT top-k prompt selection |
