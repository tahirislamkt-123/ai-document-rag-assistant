import json
import os
import platform
import time
import unittest
from pathlib import Path

from backend.rag.answerability import (
    CalibrationExample,
    CalibratedRetrievalGate,
)
from backend.rag.generator import QwenGenerator
from backend.rag.pipeline import RAGPipeline
from backend.rag.retriever import FaissRetriever


RUN_INTEGRATION = os.environ.get("RUN_QWEN_INTEGRATION") == "1"
ROOT = Path(__file__).resolve().parents[1]


def model_cache_path():
    hf_home = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    )
    hub_cache = Path(
        os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            str(hf_home / "hub"),
        )
    )
    return hub_cache / "models--Qwen--Qwen2.5-0.5B-Instruct"


def embedding_cache_path():
    hf_home = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    )
    hub_cache = Path(
        os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            str(hf_home / "hub"),
        )
    )
    return hub_cache / (
        "models--sentence-transformers--all-MiniLM-L6-v2"
    )


@unittest.skipUnless(
    RUN_INTEGRATION,
    "Set RUN_QWEN_INTEGRATION=1 to run the real Qwen integration test",
)
class QwenIntegrationTests(unittest.TestCase):
    def test_answer_reference_abstention_and_citation_checks(self):
        if not model_cache_path().exists():
            self.skipTest(
                "Qwen weights are not cached; this test never downloads weights"
            )
        if not embedding_cache_path().exists():
            self.skipTest(
                "Embedding weights are not cached; this test never downloads weights"
            )

        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ):
            os.environ.setdefault(variable, "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        retrieval_data = json.loads(
            (ROOT / "evaluation" / "retrieval_cases.json").read_text()
        )
        rag_data = json.loads(
            (ROOT / "evaluation" / "rag_cases.json").read_text()
        )
        from sentence_transformers import SentenceTransformer

        embedding_load_started = time.perf_counter()
        embedding_model = SentenceTransformer(
            FaissRetriever.MODEL_ID,
            local_files_only=True,
        )
        embedding_load_seconds = time.perf_counter() - embedding_load_started

        index_started = time.perf_counter()
        retriever = FaissRetriever(
            retrieval_data["documents"],
            model=embedding_model,
            local_files_only=True,
        )
        index_build_seconds = time.perf_counter() - index_started

        calibration_examples = []
        for case in rag_data["calibration_cases"]:
            top_hit = retriever.search(case["question"], top_k=1)[0]
            calibration_examples.append(
                CalibrationExample(
                    score=top_hit["score"],
                    answerable=case["answerable"],
                )
            )
        gate = CalibratedRetrievalGate.fit(
            calibration_examples,
            minimum_precision=1.0,
            calibration_id="rag-cases-v1",
        )
        generator = QwenGenerator(
            max_new_tokens=80,
            local_files_only=True,
        )
        retrieved_by_question = {}

        def search_and_record(question, top_k):
            hits = retriever.search(question, top_k)
            retrieved_by_question[question] = hits
            return hits

        pipeline = RAGPipeline(
            search=search_and_record,
            generator=generator,
            answerability_gate=gate,
            max_prompt_tokens=512,
        )

        results = []
        cold_answer_seconds = None
        for case in rag_data["evaluation_cases"]:
            ask_started = time.perf_counter()
            response = pipeline.ask(case["question"])
            ask_seconds = time.perf_counter() - ask_started
            retrieved_hits = retrieved_by_question[case["question"]]
            if case["answerable"]:
                expected_hit = next(
                    (
                        hit
                        for hit in retrieved_hits
                        if hit.get("id") == case["expected_chunk_id"]
                    ),
                    None,
                )
                answer_reference_check = (
                    not response.abstained
                    and case["expected_answer_substring"] in response.answer
                )
                citations = set(response.citations)
                sources_by_citation = {
                    source.citation: source for source in response.sources
                }
                citation_provenance_valid = bool(citations) and all(
                    citation in sources_by_citation
                    and any(
                        hit.get("source")
                        == sources_by_citation[citation].document_id
                        and hit.get("page")
                        == sources_by_citation[citation].page
                        and hit.get("id")
                        == sources_by_citation[citation].chunk_id
                        for hit in retrieved_hits
                    )
                    for citation in citations
                )
                expected_sources = [
                    source
                    for source in response.sources
                    if source.citation in citations
                    and source.document_id == case["expected_source"]
                    and source.page == case["expected_page"]
                    and source.chunk_id == case["expected_chunk_id"]
                ]
                cited_support_span_present = bool(
                    expected_hit
                    and case["supporting_text"] in expected_hit["text"]
                    and expected_sources
                )
                if cold_answer_seconds is None:
                    cold_answer_seconds = ask_seconds
                results.append(
                    {
                        "answerable": True,
                        "retrieval_hit": expected_hit is not None,
                        "answer_reference_check": answer_reference_check,
                        "citation_provenance_valid": (
                            citation_provenance_valid
                        ),
                        "cited_support_span_present": (
                            cited_support_span_present
                        ),
                        "answer": response.answer,
                        "latency_seconds": ask_seconds,
                    }
                )
            else:
                results.append(
                    {
                        "answerable": False,
                        "abstained": response.abstained,
                        "calibration_rejected": (
                            response.abstention_reason
                            == "below_calibrated_cutoff"
                        ),
                        "answer": response.answer,
                        "latency_seconds": ask_seconds,
                    }
                )

        answer_results = [
            result for result in results if result["answerable"]
        ]
        abstention_results = [
            result for result in results if not result["answerable"]
        ]
        retrieval_hits = sum(
            result["retrieval_hit"] for result in answer_results
        )
        answer_reference_checks = sum(
            result["answer_reference_check"] for result in answer_results
        )
        provenance_checks = sum(
            result["citation_provenance_valid"] for result in answer_results
        )
        support_span_checks = sum(
            result["cited_support_span_present"] for result in answer_results
        )
        correct_abstentions = sum(
            result["abstained"] and result["calibration_rejected"]
            for result in abstention_results
        )
        retrieval_errors = len(answer_results) - retrieval_hits
        generation_or_citation_errors = sum(
            result["retrieval_hit"]
            and not (
                result["answer_reference_check"]
                and result["citation_provenance_valid"]
                and result["cited_support_span_present"]
            )
            for result in answer_results
        )
        print(
            "Qwen RAG evaluation: cases={}, answerable={}, "
            "retrieval_hit@3={}/{}, "
            "answer_reference_check={}/{}, citation_provenance={}/{}, "
            "cited_support_span={}/{}, unanswerable={}, "
            "calibrated_abstention={}/{}, retrieval_errors={}, "
            "generation_or_citation_errors={}".format(
                len(results),
                len(answer_results),
                retrieval_hits,
                len(answer_results),
                answer_reference_checks,
                len(answer_results),
                provenance_checks,
                len(answer_results),
                support_span_checks,
                len(answer_results),
                len(abstention_results),
                correct_abstentions,
                len(abstention_results),
                retrieval_errors,
                generation_or_citation_errors,
            )
        )
        print(
            "RAG performance: platform={}, machine={}, embedding_model={}, "
            "embedding_device={}, embedding_load_seconds={:.3f}, "
            "index_build_seconds={:.3f}, first_answer_request_seconds={}".format(
                platform.platform(),
                platform.machine(),
                FaissRetriever.MODEL_ID,
                embedding_model.device,
                embedding_load_seconds,
                index_build_seconds,
                (
                    "not-run"
                    if cold_answer_seconds is None
                    else "{:.3f}".format(cold_answer_seconds)
                ),
            )
        )
        for result in results:
            print("  case:", result)

        self.assertEqual(retrieval_hits, len(answer_results), results)
        self.assertEqual(
            answer_reference_checks,
            len(answer_results),
            results,
        )
        self.assertEqual(provenance_checks, len(answer_results), results)
        self.assertEqual(support_span_checks, len(answer_results), results)
        self.assertEqual(
            correct_abstentions,
            len(abstention_results),
            results,
        )


if __name__ == "__main__":
    unittest.main()
