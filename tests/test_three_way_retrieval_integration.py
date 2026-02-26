import argparse
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "05_three_way_retrieval.py"


class _FakeChunkIndex:
    def __init__(self) -> None:
        self._chunks = {
            "fake": [
                {
                    "chunk_id": "c1",
                    "parent_id": "p1",
                    "content": "chunk one",
                    "metadata": {"pdf": "fake.pdf", "page_numbers": [1], "section_path": ["A"]},
                },
                {
                    "chunk_id": "c2",
                    "parent_id": "p2",
                    "content": "chunk two",
                    "metadata": {"pdf": "fake.pdf", "page_numbers": [2], "section_path": ["B"]},
                },
            ]
        }

    def get_chunks_by_pdf(self, pdf_stem: str) -> List[Dict[str, Any]]:
        return list(self._chunks.get(pdf_stem, []))


class _FakeEmbeddingRetriever:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def encode_chunks(self, chunk_texts: List[str]) -> np.ndarray:
        return np.zeros((len(chunk_texts), 2), dtype=float)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return np.zeros((len(queries), 2), dtype=float)

    def retrieve_top_k(
        self,
        query_embeddings: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunk_metadata: List[Dict[str, Any]],
        top_k: int,
    ) -> List[List[Dict[str, Any]]]:
        _ = chunk_embeddings
        rank_and_score = {"c1": (1, 0.9), "c2": (3, 0.2)}
        results: List[List[Dict[str, Any]]] = []
        for _ in range(query_embeddings.shape[0]):
            hits: List[Dict[str, Any]] = []
            for meta in chunk_metadata:
                cid = str(meta.get("chunk_id"))
                if cid not in rank_and_score:
                    continue
                rank, score = rank_and_score[cid]
                hit = dict(meta)
                hit["rank"] = rank
                hit["score"] = score
                hits.append(hit)
            hits.sort(key=lambda x: x["rank"])
            results.append(hits[:top_k])
        return results


class _FakeBM25Retriever:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def add_pdf(self, pdf_stem: str, chunks: List[Dict[str, Any]]) -> None:
        _ = (pdf_stem, chunks)

    def retrieve(self, query: str, pdf_stem: str, top_k: int) -> List[Dict[str, Any]]:
        _ = (query, pdf_stem)
        hits = [
            {
                "chunk_id": "c2",
                "parent_id": "p2",
                "pdf": "fake.pdf",
                "pdf_stem": "fake",
                "page_numbers": [2],
                "section_path": ["B"],
                "score": 100.0,
                "rank": 1,
            },
            {
                "chunk_id": "c1",
                "parent_id": "p1",
                "pdf": "fake.pdf",
                "pdf_stem": "fake",
                "page_numbers": [1],
                "section_path": ["A"],
                "score": 10.0,
                "rank": 5,
            },
        ]
        return hits[:top_k]


class ThreeWayRetrievalIntegrationTest(unittest.TestCase):
    def _load_script_module(self):
        spec = importlib.util.spec_from_file_location("three_way_retrieval_under_test", SCRIPT_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载脚本: {SCRIPT_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _repo_output_snapshot(self) -> set[str]:
        tracked: set[str] = set()
        for rel in ("data/output/retrieval", "data/output/checkpoints"):
            path = REPO_ROOT / rel
            if not path.exists():
                continue
            for item in path.rglob("*"):
                if item.is_file():
                    tracked.add(str(item.relative_to(REPO_ROOT)))
        return tracked

    def _run_case(self, fusion_method: str, include_weights: bool) -> Dict[str, Any]:
        before_snapshot = self._repo_output_snapshot()
        tmp_path_obj: Path | None = None
        with tempfile.TemporaryDirectory(prefix="qanchor-test-") as tmp:
            tmp_path = Path(tmp)
            tmp_path_obj = tmp_path

            output_dir = tmp_path / "retrieval"
            checkpoint_path = tmp_path / "checkpoints" / "ckpt.json"
            config_path = tmp_path / "config.yaml"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            cfg = {
                "data": {
                    "qa_mapping": str(tmp_path / "qa.csv"),
                    "answers": str(tmp_path / "answers.jsonl"),
                    "pdf_dir": str(tmp_path / "pdf"),
                    "summary": str(tmp_path / "summary.csv"),
                    "chunk_output": str(tmp_path / "chunks"),
                    "retrieval_output": str(output_dir),
                },
                "stages": {"stage1": {"qa_count": 1}},
                "retrieval": {
                    "embedding_model": "fake/model",
                    "batch_size": 2,
                    "normalize_embeddings": True,
                    "hybrid": {
                        "fusion_method": fusion_method,
                        "rrf_k": 60,
                        "missing_rank": 9999,
                    },
                },
            }
            if include_weights:
                cfg["retrieval"]["hybrid"]["embedding_weight"] = 0.7
                cfg["retrieval"]["hybrid"]["bm25_weight"] = 0.3

            config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

            qa_df = pd.DataFrame(
                [
                    {
                        "master_id": 1,
                        "question": "fake question for integration",
                        "report_paths": "fake.pdf",
                        "company": "Acme",
                        "year": 2024,
                    }
                ]
            )
            answers = [{"master_id": 1, "answers": ["fake answer"]}]
            subset = {"records": [{"pdf_path": "fake.pdf"}]}
            chunk_index = _FakeChunkIndex()

            module = self._load_script_module()
            args = argparse.Namespace(
                stage="stage1",
                config=str(config_path),
                output_dir=str(output_dir),
                top_k_train=3,
                top_k_eval=2,
                exclude_pdfs=None,
                checkpoint_path=str(checkpoint_path),
            )

            buffer = io.StringIO()
            with (
                mock.patch.object(module, "load_qa_mapping", return_value=qa_df),
                mock.patch.object(module, "load_answers", return_value=answers),
                mock.patch.object(module, "select_pdf_subset", return_value=subset),
                mock.patch.object(module, "load_chunks", return_value=chunk_index),
                mock.patch.object(module, "EmbeddingRetriever", _FakeEmbeddingRetriever),
                mock.patch.object(module, "BM25Retriever", _FakeBM25Retriever),
                redirect_stdout(buffer),
            ):
                module.run_three_way_retrieval(args)

            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            effective_method = ckpt["params"]["fusion_method_effective"]
            hybrid_path = output_dir / f"hybrid_{effective_method}_top3_stage1.jsonl"
            hybrid_exists = hybrid_path.exists()
            lines = hybrid_path.read_text(encoding="utf-8").strip().splitlines()
            first_record = json.loads(lines[0]) if lines else {}

            after_snapshot = self._repo_output_snapshot()
            self.assertEqual(before_snapshot, after_snapshot)

            result = {
                "stdout": buffer.getvalue(),
                "checkpoint": ckpt,
                "first_record": first_record,
                "hybrid_exists": hybrid_exists,
            }
        self.assertIsNotNone(tmp_path_obj)
        self.assertFalse(tmp_path_obj.exists())
        return result

    def test_rrf_matches_historical_formula_ranking(self) -> None:
        data = self._run_case("rrf", include_weights=False)
        params = data["checkpoint"]["params"]
        hits = data["first_record"]["hits"]
        score_map = {h["chunk_id"]: h["score"] for h in hits}

        expected_c1 = 1.0 / (60 + 1) + 1.0 / (60 + 5)
        expected_c2 = 1.0 / (60 + 3) + 1.0 / (60 + 1)

        self.assertEqual(params["fusion_method_effective"], "rrf")
        self.assertAlmostEqual(score_map["c1"], expected_c1, places=12)
        self.assertAlmostEqual(score_map["c2"], expected_c2, places=12)
        self.assertEqual(hits[0]["chunk_id"], "c2")
        self.assertEqual(hits[0]["rank"], 1)

    def test_weighted_sum_runs_and_outputs(self) -> None:
        data = self._run_case("weighted_sum", include_weights=True)
        params = data["checkpoint"]["params"]
        hits = data["first_record"]["hits"]

        self.assertEqual(params["fusion_method_effective"], "weighted_sum")
        self.assertEqual(params["score_normalization"], "minmax")
        self.assertTrue(data["hybrid_exists"])
        self.assertEqual(hits[0]["chunk_id"], "c1")

    def test_max_runs_and_outputs(self) -> None:
        data = self._run_case("max", include_weights=True)
        params = data["checkpoint"]["params"]
        hits = data["first_record"]["hits"]

        self.assertEqual(params["fusion_method_effective"], "max")
        self.assertEqual(params["score_normalization"], "minmax")
        self.assertTrue(data["hybrid_exists"])
        self.assertEqual(hits[0]["chunk_id"], "c1")

    def test_unknown_method_warns_and_falls_back_to_rrf(self) -> None:
        data = self._run_case("unknown", include_weights=False)
        params = data["checkpoint"]["params"]

        self.assertEqual(params["fusion_method_config"], "unknown")
        self.assertEqual(params["fusion_method_effective"], "rrf")
        self.assertIn("回退到 fusion_method='rrf'", data["stdout"])
        self.assertTrue(data["hybrid_exists"])


if __name__ == "__main__":
    unittest.main()
