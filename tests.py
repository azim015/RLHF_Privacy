"""
tests.py – offline unit tests (no HuggingFace downloads required)
Run with:  python tests.py
"""

import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import MagicMock, patch


def make_blank_image(w=64, h=64):
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 50, 40], fill=(200, 50, 50))
    return img


def make_mock_sbert(dim=384):
    """Return a SentenceTransformer mock whose encode() returns random L2-normed vecs."""
    mock = MagicMock()
    def _encode(texts, **kw):
        arr = np.random.randn(len(texts), dim).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        return arr / norms
    mock.encode = _encode
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):

    def test_model_config_defaults(self):
        from config import ModelConfig
        cfg = ModelConfig()
        self.assertEqual(cfg.num_hierarchy_levels, 3)
        self.assertAlmostEqual(cfg.alpha + cfg.beta, 1.0, places=5)
        self.assertGreater(cfg.ppo_max_iterations, 0)

    def test_prompt_config_layers(self):
        from config import PromptConfig
        pc = PromptConfig()
        self.assertGreater(len(pc.layer_a), 0)
        self.assertGreater(len(pc.layer_b), 0)
        self.assertGreater(len(pc.layer_c), 0)
        for p in pc.layer_a + pc.layer_b + pc.layer_c:
            self.assertIsInstance(p, str)


# ─────────────────────────────────────────────────────────────────────────────
# Privacy classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestPrivacyClassifier(unittest.TestCase):

    def setUp(self):
        from vlm_module import PrivacyClassifier
        self.clf = PrivacyClassifier()

    def test_clean_text_low_score(self):
        score = self.clf.score("A red car is parked on the street.")
        self.assertLess(score, 0.1)

    def test_pii_text_high_score(self):
        score = self.clf.score("The person's face and name were visible on the licence plate.")
        self.assertGreater(score, 0.0)

    def test_is_private(self):
        self.assertTrue(self.clf.is_private("A vehicle is moving forward."))
        self.assertFalse(self.clf.is_private(
            "I can recognise their face and identity clearly."))


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────────────────────

class TestPPOAgent(unittest.TestCase):

    def setUp(self):
        from rl_module import PPOAgent
        self.state_dim  = 385
        self.action_dim = 8
        self.agent = PPOAgent(
            state_dim=self.state_dim, action_dim=self.action_dim,
            lr=1e-3, clip_epsilon=0.2, entropy_coef=0.01)

    def test_select_action_valid_range(self):
        state = np.random.randn(self.state_dim).astype(np.float32)
        action, log_prob, value = self.agent.select_action(state)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_dim)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value,    float)

    def test_update_returns_loss_dict(self):
        for _ in range(4):
            state = np.random.randn(self.state_dim).astype(np.float32)
            action, lp, val = self.agent.select_action(state)
            self.agent.store_transition(state, action, 0.5, lp, val)
        info = self.agent.update()
        self.assertIn("ppo_loss",   info)
        self.assertIn("total_loss", info)
        self.assertIn("entropy",    info)

    def test_build_state_shape(self):
        from rl_module import PPOAgent
        emb   = np.random.randn(384).astype(np.float32)
        state = PPOAgent.build_state(emb, privacy_score=0.1)
        self.assertEqual(state.shape, (385,))

    def test_entropy_decay(self):
        coef_before = self.agent.entropy_coef
        state = np.random.randn(self.state_dim).astype(np.float32)
        action, lp, val = self.agent.select_action(state)
        self.agent.store_transition(state, action, 1.0, lp, val)
        self.agent.update(decay_entropy=True)
        self.assertLessEqual(self.agent.entropy_coef, coef_before)


# ─────────────────────────────────────────────────────────────────────────────
# SBERT Ranker (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestSBERTRanker(unittest.TestCase):

    def setUp(self):
        from rl_module import SBERTPromptRanker
        self.ranker = SBERTPromptRanker.__new__(SBERTPromptRanker)
        self.ranker.model  = make_mock_sbert()
        self.ranker.device = "cpu"

    def test_rank_prompts_returns_indices(self):
        prompts = [
            "Describe the vehicle.",
            "What is the weather like?",
            "Describe road conditions.",
        ]
        indices, scores = self.ranker.rank_prompts(
            "A red car on a wet road.", prompts, top_k=2)
        self.assertEqual(len(indices), 2)
        for idx in indices:
            self.assertIn(idx, range(len(prompts)))

    def test_rank_top_k_bounded_by_pool(self):
        prompts = ["A", "B"]
        indices, scores = self.ranker.rank_prompts("test", prompts, top_k=5)
        self.assertLessEqual(len(indices), len(prompts))


# ─────────────────────────────────────────────────────────────────────────────
# RAG Module (mocked SBERT)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGModule(unittest.TestCase):

    def setUp(self):
        from rag_module import RAGModule
        self.rag = RAGModule(
            sbert_model=make_mock_sbert(), embedding_dim=384, top_k=2)
        self.rag.add_documents([
            "A vehicle is moving forward.",
            "The road shows heavy congestion.",
            "Pedestrians are crossing the street.",
        ])

    def test_retrieve_returns_results(self):
        docs, scores = self.rag.retrieve("car on the road")
        self.assertGreater(len(docs), 0)
        for s in scores:
            self.assertGreaterEqual(s, -1.0 - 1e-6)
            self.assertLessEqual(s,  1.0 + 1e-6)

    def test_validation_score_in_range(self):
        score = self.rag.validation_score(
            "A car is driving.", "Describe the vehicle.")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0 + 1e-6)

    def test_dp_noise_changes_embedding(self):
        from rag_module import RAGModule
        emb   = np.ones(384, dtype=float)
        noisy = RAGModule.apply_dp_noise(emb, sigma=0.5)
        self.assertFalse(np.allclose(emb, noisy))

    def test_compute_epsilon(self):
        from rag_module import RAGModule
        eps = RAGModule.compute_epsilon(delta2_f=1.0, sigma=0.1)
        self.assertAlmostEqual(eps, 10.0, places=3)

    def test_augment_reward_increases_reward(self):
        aug = self.rag.augment_reward(
            0.5, "A car is driving.", "Describe the vehicle.")
        self.assertGreater(aug, 0.5)

    def test_privacy_gain_table(self):
        from rag_module import RAGModule
        r = RAGModule.privacy_gain(sigma=0.1)
        self.assertIn("epsilon",          r)
        self.assertIn("bleu_score",       r)
        self.assertIn("privacy_gain_pct", r)
        self.assertGreater(r["privacy_gain_pct"], 0)

    def test_add_documents_increases_index(self):
        before = self.rag.index.ntotal
        self.rag.add_documents(["New document about traffic."])
        self.assertEqual(self.rag.index.ntotal, before + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical framework (VLM + SBERT mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestHierarchicalFramework(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from config import ModelConfig, PromptConfig
        from hierarchical_framework import HierarchicalFeedbackRLVLM

        cfg = ModelConfig()

        # Patch VLMModule so no HF download is needed
        with patch("hierarchical_framework.VLMModule") as MockVLM, \
             patch("hierarchical_framework.SentenceTransformer") as MockST:

            mock_vlm_instance = MagicMock()
            mock_vlm_instance.generate.return_value = (
                "A vehicle is moving on the road without revealing any faces.")
            mock_vlm_instance.privacy_classifier = MagicMock()
            mock_vlm_instance.privacy_classifier.score.return_value = 0.0
            MockVLM.return_value = mock_vlm_instance

            mock_st_instance = make_mock_sbert()
            MockST.return_value = mock_st_instance

            cls.framework = HierarchicalFeedbackRLVLM.__new__(
                HierarchicalFeedbackRLVLM)

            # Manually wire up components so we can test process_image
            from rl_module import SBERTPromptRanker, PPOAgent
            from rag_module import RAGModule
            from vlm_module import PrivacyClassifier

            cls.framework.config       = cfg
            cls.framework.prompt_config = PromptConfig()
            cls.framework.device       = "cpu"
            cls.framework.vlm          = mock_vlm_instance
            cls.framework.privacy_clf  = PrivacyClassifier()
            cls.framework.sbert_model  = mock_st_instance

            ranker = SBERTPromptRanker.__new__(SBERTPromptRanker)
            ranker.model  = mock_st_instance
            ranker.device = "cpu"
            cls.framework.ranker = ranker

            cls.framework.rag = RAGModule(
                sbert_model=mock_st_instance,
                embedding_dim=cfg.embedding_dim,
                top_k=cfg.rag_top_k)

            pools = [
                cls.framework.prompt_config.layer_a,
                cls.framework.prompt_config.layer_b,
                cls.framework.prompt_config.layer_c,
            ]
            state_dim = cfg.embedding_dim + 1
            cls.framework.agents = [
                PPOAgent(state_dim=state_dim, action_dim=len(p),
                         lr=cfg.ppo_learning_rate,
                         clip_epsilon=cfg.ppo_clip_epsilon,
                         entropy_coef=cfg.entropy_start,
                         hidden1=cfg.policy_hidden_1,
                         hidden2=cfg.policy_hidden_2)
                for p in pools
            ]
            all_prompts = sum(pools, [])
            cls.framework.rag.add_documents(all_prompts)

        cls.image = make_blank_image()

    def test_process_image_returns_keys(self):
        result = self.framework.process_image(
            self.image, ground_truth_text="A car on the road.", verbose=False)
        for key in ("final_description", "level_results",
                    "privacy_metrics", "dp_embedding"):
            self.assertIn(key, result)

    def test_three_level_results(self):
        result = self.framework.process_image(self.image, verbose=False)
        self.assertEqual(len(result["level_results"]), 3)

    def test_final_description_is_string(self):
        result = self.framework.process_image(self.image, verbose=False)
        self.assertIsInstance(result["final_description"], str)
        self.assertGreater(len(result["final_description"]), 5)

    def test_dp_metrics_present(self):
        result = self.framework.process_image(self.image, verbose=False)
        pm = result["privacy_metrics"]
        self.assertIn("dp_epsilon", pm)
        self.assertIn("dp_sigma",   pm)

    def test_evaluate_privacy_metrics(self):
        metrics = self.framework.evaluate_privacy(
            "A car is moving.", "A vehicle is travelling on the road.")
        for key in ("semantic_similarity", "word_count",
                    "unique_word_count", "detail_density"):
            self.assertIn(key, metrics)

    def test_evaluate_image_privacy(self):
        from hierarchical_framework import HierarchicalFeedbackRLVLM
        orig  = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        recon = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        m = HierarchicalFeedbackRLVLM.evaluate_image_privacy(orig, recon)
        for key in ("ssim", "psnr", "mse"):
            self.assertIn(key, m)
        self.assertGreater(m["mse"], 0)

    def test_aggregate_deduplicates(self):
        from hierarchical_framework import HierarchicalFeedbackRLVLM
        descs = ["A car is on the road.", "A car is on the road.", "A truck is parked."]
        out = HierarchicalFeedbackRLVLM._aggregate(descs)
        # "A car is on the road" should appear only once
        self.assertEqual(out.lower().count("a car is on the road"), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardComputation(unittest.TestCase):

    def setUp(self):
        from hierarchical_framework import compute_reward, vae_reconstruction_similarity
        self.compute_reward = compute_reward
        self.vae_sim = vae_reconstruction_similarity

    def test_vae_sim_no_reconstruction(self):
        img = make_blank_image()
        score = self.vae_sim(img, None)
        self.assertEqual(score, 0.0)

    def test_vae_sim_identical_images(self):
        img   = make_blank_image()
        score = self.vae_sim(img, img)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_compute_reward_with_mock_sbert(self):
        mock_sbert = make_mock_sbert()
        reward = self.compute_reward(
            mock_sbert,
            generated_text="A car is driving.",
            ground_truth_text="A vehicle is moving.",
            alpha=0.7, beta=0.3,
        )
        self.assertIsInstance(reward, float)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HFR-VLM Framework Unit Tests  (offline)")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
