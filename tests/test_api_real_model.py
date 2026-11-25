import json
import os
import platform
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.service import Settings
from backend.rag.generator import QwenGenerator
from backend.rag.retriever import FaissRetriever
from tests.pdf_fixtures import make_text_pdf


ROOT = Path(__file__).resolve().parents[1]
RUN_INTEGRATION = os.environ.get("RUN_QWEN_INTEGRATION") == "1"


def cached_model_path(repository):
    cache_home = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    )
    hub_cache = Path(
        os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            str(cache_home / "hub"),
        )
    )
    return hub_cache / ("models--" + repository.replace("/", "--"))


@unittest.skipUnless(
    RUN_INTEGRATION,
    "Set RUN_QWEN_INTEGRATION=1 to run the real API smoke test",
)
class RealModelAPISmokeTests(unittest.TestCase):
    def test_upload_to_real_model_answer_abstention_and_citation(self):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        model_id = FaissRetriever.MODEL_ID
        generator_id = "Qwen/Qwen2.5-0.5B-Instruct"
        if not cached_model_path(model_id).exists():
            self.skipTest("Embedding weights are not cached")
        if not cached_model_path(generator_id).exists():
            self.skipTest("Qwen weights are not cached")

        from sentence_transformers import SentenceTransformer

        retrieval_data = json.loads(
            (ROOT / "evaluation" / "retrieval_cases.json").read_text(
                encoding="utf-8"
            )
        )
        calibration_data = json.loads(
            (ROOT / "evaluation" / "rag_cases.json").read_text(
                encoding="utf-8"
            )
        )
        embedding_load_started = time.perf_counter()
        embedding_model = SentenceTransformer(
            model_id,
            local_files_only=True,
        )
        embedding_load_seconds = time.perf_counter() - embedding_load_started
        index_started = time.perf_counter()
        calibration_retriever = FaissRetriever(
            retrieval_data["documents"],
            model=embedding_model,
        )
        calibration_index_seconds = time.perf_counter() - index_started
        calibration = [
            {
                "score": calibration_retriever.search(
                    case["question"],
                    top_k=1,
                )[0]["score"],
                "answerable": case["answerable"],
            }
            for case in calibration_data["calibration_cases"]
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(calibration),
                encoding="utf-8",
            )
            settings = Settings(
                data_dir=root / "private-data",
                calibration_file=calibration_path,
                max_prompt_tokens=512,
                top_k=3,
            )
            generators = []

            def create_generator():
                generator = QwenGenerator(
                    max_new_tokens=80,
                    local_files_only=True,
                )
                generators.append(generator)
                return generator

            app = create_app(
                settings=settings,
                embedding_model_factory=lambda requested_model: embedding_model,
                generator_factory=create_generator,
            )
            startup_started = time.perf_counter()
            with TestClient(app) as client:
                startup_seconds = time.perf_counter() - startup_started
                upload_started = time.perf_counter()
                upload = client.post(
                    "/documents/upload",
                    files={
                        "file": (
                            "sample-handbook.pdf",
                            make_text_pdf(
                                "Full-time employees receive 25 days of "
                                "annual leave each year."
                            ),
                            "application/pdf",
                        )
                    },
                )
                upload_seconds = time.perf_counter() - upload_started
                self.assertEqual(upload.status_code, 201, upload.text)
                document_id = upload.json()["document_id"]

                first_answer_started = time.perf_counter()
                answer = client.post(
                    "/ask",
                    json={
                        "question": (
                            "How many annual leave days are full-time "
                            "employees entitled to?"
                        ),
                        "document_ids": [document_id],
                    },
                )
                first_answer_seconds = (
                    time.perf_counter() - first_answer_started
                )
                self.assertEqual(answer.status_code, 200, answer.text)
                answer_data = answer.json()
                self.assertEqual(answer_data["status"], "answered")
                self.assertIn("25", answer_data["answer"])
                self.assertTrue(
                    any(
                        citation["document_id"] == document_id
                        and citation["page"] == 1
                        for citation in answer_data["citations"]
                    ),
                    answer_data,
                )

                warm_answer_started = time.perf_counter()
                warm_answer = client.post(
                    "/ask",
                    json={
                        "question": (
                            "How many annual leave days are full-time "
                            "employees entitled to?"
                        ),
                        "document_ids": [document_id],
                    },
                )
                warm_answer_seconds = (
                    time.perf_counter() - warm_answer_started
                )
                self.assertEqual(warm_answer.status_code, 200)
                self.assertEqual(warm_answer.json()["status"], "answered")

                abstention_started = time.perf_counter()
                abstention = client.post(
                    "/ask",
                    json={
                        "question": "Where is the company headquarters?",
                        "document_ids": [document_id],
                    },
                )
                abstention_seconds = (
                    time.perf_counter() - abstention_started
                )
                self.assertEqual(abstention.status_code, 200)
                self.assertEqual(abstention.json()["status"], "abstained")
                print(
                    "RAG API performance: platform={}, machine={}, "
                    "embedding_model={}, generation_model={}, "
                    "embedding_load_seconds={:.3f}, "
                    "calibration_index_build_seconds={:.3f}, "
                    "api_startup_seconds={:.3f}, "
                    "synthetic_pdf_upload_index_seconds={:.3f}, "
                    "first_answer_cold_seconds={:.3f}, "
                    "warm_answer_seconds={:.3f}, "
                    "calibrated_abstention_seconds={:.3f}, "
                    "generation_device={}".format(
                        platform.platform(),
                        platform.machine(),
                        model_id,
                        generator_id,
                        embedding_load_seconds,
                        calibration_index_seconds,
                        startup_seconds,
                        upload_seconds,
                        first_answer_seconds,
                        warm_answer_seconds,
                        abstention_seconds,
                        generators[0].device,
                    )
                )


if __name__ == "__main__":
    unittest.main()
