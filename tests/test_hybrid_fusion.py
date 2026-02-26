import math
import unittest

from src.hybrid_fusion import fuse_two_way, rrf_fuse


def _hit(chunk_id: str, rank: int, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "parent_id": f"p-{chunk_id}",
        "pdf": "demo.pdf",
        "pdf_stem": "demo",
        "page_numbers": [1],
        "section_path": ["sec"],
        "rank": rank,
        "score": score,
    }


class HybridFusionTest(unittest.TestCase):
    def test_rrf_equal_weight_matches_old_formula(self) -> None:
        emb_hits = [_hit("c1", 1, 0.9), _hit("c2", 3, 0.2)]
        bm_hits = [_hit("c2", 1, 100.0), _hit("c1", 5, 10.0)]

        fused = rrf_fuse(emb_hits, bm_hits, rrf_k=60, missing_rank=9999, top_k=None)
        score_map = {item["chunk_id"]: item["score"] for item in fused}

        expected_c1 = 1.0 / (60 + 1) + 1.0 / (60 + 5)
        expected_c2 = 1.0 / (60 + 3) + 1.0 / (60 + 1)

        self.assertAlmostEqual(score_map["c1"], expected_c1, places=12)
        self.assertAlmostEqual(score_map["c2"], expected_c2, places=12)
        self.assertEqual(fused[0]["chunk_id"], "c2")
        self.assertEqual(fused[0]["rank"], 1)

    def test_weighted_rrf_changes_ranking(self) -> None:
        emb_hits = [_hit("c1", 1, 0.9), _hit("c2", 10, 0.8)]
        bm_hits = [_hit("c1", 50, 10.0), _hit("c2", 1, 100.0)]

        equal_weight = fuse_two_way(emb_hits, bm_hits, fusion_method="rrf", rrf_k=60, top_k=None)
        weighted = fuse_two_way(
            emb_hits,
            bm_hits,
            fusion_method="rrf",
            embedding_weight=5.0,
            bm25_weight=1.0,
            rrf_k=60,
            top_k=None,
        )

        self.assertEqual(equal_weight[0]["chunk_id"], "c2")
        self.assertEqual(weighted[0]["chunk_id"], "c1")

    def test_weighted_sum_uses_minmax_normalization(self) -> None:
        emb_hits = [_hit("c1", 1, 0.9), _hit("c2", 2, 0.1)]
        bm_hits = [_hit("c1", 2, 10.0), _hit("c2", 1, 1000.0)]

        fused = fuse_two_way(
            emb_hits,
            bm_hits,
            fusion_method="weighted_sum",
            embedding_weight=0.7,
            bm25_weight=0.3,
            top_k=None,
        )
        score_map = {item["chunk_id"]: item["score"] for item in fused}

        self.assertAlmostEqual(score_map["c1"], 0.7, places=9)
        self.assertAlmostEqual(score_map["c2"], 0.3, places=9)
        self.assertEqual(fused[0]["chunk_id"], "c1")

    def test_max_uses_normalized_weighted_components(self) -> None:
        emb_hits = [_hit("c1", 1, 0.9), _hit("c2", 2, 0.1)]
        bm_hits = [_hit("c1", 2, 10.0), _hit("c2", 1, 1000.0)]

        fused = fuse_two_way(
            emb_hits,
            bm_hits,
            fusion_method="max",
            embedding_weight=0.7,
            bm25_weight=0.3,
            top_k=None,
        )
        score_map = {item["chunk_id"]: item["score"] for item in fused}

        self.assertAlmostEqual(score_map["c1"], 0.7, places=9)
        self.assertAlmostEqual(score_map["c2"], 0.3, places=9)
        self.assertEqual(fused[0]["chunk_id"], "c1")

    def test_missing_one_side_hit_is_handled(self) -> None:
        emb_hits = [_hit("c1", 1, 0.9)]
        bm_hits = [_hit("c2", 5, 100.0)]

        fused = fuse_two_way(
            emb_hits,
            bm_hits,
            fusion_method="rrf",
            rrf_k=60,
            missing_rank=9999,
            top_k=None,
        )
        score_map = {item["chunk_id"]: item["score"] for item in fused}

        expected_c1 = 1.0 / (60 + 1) + 1.0 / (60 + 9999)
        expected_c2 = 1.0 / (60 + 9999) + 1.0 / (60 + 5)

        self.assertAlmostEqual(score_map["c1"], expected_c1, places=12)
        self.assertAlmostEqual(score_map["c2"], expected_c2, places=12)
        self.assertGreater(score_map["c1"], score_map["c2"])
        for item in fused:
            self.assertIn("embedding_score", item)
            self.assertIn("bm25_score", item)

    def test_minmax_boundary_max_equals_min_sets_zero(self) -> None:
        emb_hits = [_hit("c1", 1, 5.0), _hit("c2", 2, 5.0)]
        bm_hits = [_hit("c1", 2, 2.0), _hit("c2", 1, 1.0)]

        fused = fuse_two_way(
            emb_hits,
            bm_hits,
            fusion_method="weighted_sum",
            embedding_weight=0.7,
            bm25_weight=0.3,
            top_k=None,
        )
        score_map = {item["chunk_id"]: item["score"] for item in fused}

        self.assertAlmostEqual(score_map["c1"], 0.3, places=9)
        self.assertAlmostEqual(score_map["c2"], 0.0, places=9)
        self.assertFalse(any(math.isnan(item["score"]) for item in fused))


if __name__ == "__main__":
    unittest.main()
