import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.service import Settings
from backend.documents.pdf import PDFProcessingError, extract_pdf_pages
from tests.pdf_fixtures import make_text_pdf


class FakeEmbeddingModel:
    def encode_document(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts), 1))

    def encode_query(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        return np.array([[1.0, 0.0]], dtype="float32")


class FakeGenerator:
    def __init__(self):
        self.answer_calls = 0

    def count_prompt_tokens(self, question, evidence):
        return len(question) + len(evidence) + 32

    def count_text_tokens(self, text):
        return len(text)

    def truncate_text(self, text, max_tokens):
        return text[:max_tokens]

    def answer(self, question, evidence):
        self.answer_calls += 1
        evidence_records = json.loads(evidence)
        return "[1] " + evidence_records[0]["text"][:60]


def extract_fake_pdf(path, max_pages, max_text_characters):
    del max_pages, max_text_characters
    content = Path(path).read_bytes()
    if b"BAD_PDF" in content:
        raise PDFProcessingError("PDF is unreadable or malformed")
    if b"EMPTY_PDF" in content:
        text = ""
    else:
        text = content.decode("latin-1").split("\n", 1)[-1].strip()
    return [{"page": 1, "text": text}]


class DocumentAPIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_dir = self.root / "private-data"
        self.calibration_path = self.root / "calibration.json"
        self.calibration_path.write_text(
            json.dumps(
                [
                    {"score": 0.99, "answerable": True},
                    {"score": 0.1, "answerable": False},
                ]
            ),
            encoding="utf-8",
        )
        self.generator = FakeGenerator()
        self.settings = Settings(
            data_dir=self.data_dir,
            max_upload_bytes=64 * 1024,
            max_pages=3,
            max_text_characters=10000,
            chunk_words=20,
            overlap_words=2,
            top_k=2,
            max_prompt_tokens=256,
            max_question_characters=100,
            max_selected_documents=2,
            calibration_file=self.calibration_path,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_app(self, pdf_extractor=extract_fake_pdf, settings=None):
        return create_app(
            settings=settings or self.settings,
            embedding_model_factory=lambda model_id: FakeEmbeddingModel(),
            generator_factory=lambda: self.generator,
            pdf_extractor=pdf_extractor,
        )

    @staticmethod
    def upload(client, text, filename="policy.pdf"):
        return client.post(
            "/documents/upload",
            files={
                "file": (
                    filename,
                    b"%PDF-1.4\n" + text.encode("utf-8"),
                    "application/pdf",
                )
            },
        )

    def test_upload_to_scoped_answer_and_persistence(self):
        model_loads = []
        app = create_app(
            settings=self.settings,
            embedding_model_factory=lambda model_id: (
                model_loads.append(model_id) or FakeEmbeddingModel()
            ),
            generator_factory=lambda: self.generator,
            pdf_extractor=extract_pdf_pages,
        )
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ready")

            first = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "../../leave-policy.pdf",
                        make_text_pdf(
                            "Full-time staff receive 25 annual leave days."
                        ),
                        "application/pdf",
                    )
                },
            )
            second = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "support-policy.pdf",
                        make_text_pdf(
                            "IT support operates Monday through Friday."
                        ),
                        "application/pdf",
                    )
                },
            )
            self.assertEqual(first.status_code, 201, first.text)
            self.assertEqual(second.status_code, 201, second.text)
            first_id = first.json()["document_id"]
            second_id = second.json()["document_id"]
            self.assertEqual(first.json()["filename"], "leave-policy.pdf")

            listed = client.get("/documents")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(len(listed.json()), 2)
            self.assertTrue(
                all(item["source_available"] for item in listed.json())
            )
            pdf_response = client.get(
                "/documents/{}/file".format(first_id)
            )
            self.assertEqual(pdf_response.status_code, 200)
            self.assertEqual(pdf_response.content[:5], b"%PDF-")
            self.assertIn("inline", pdf_response.headers["content-disposition"])
            self.assertEqual(
                pdf_response.headers["cache-control"],
                "private, no-store",
            )
            self.assertEqual(
                client.get("/documents/{}/file".format("0" * 32)).status_code,
                404,
            )
            self.assertEqual(
                client.get("/documents/not-a-document/file").status_code,
                404,
            )

            answer = client.post(
                "/ask",
                json={
                    "question": "How many annual leave days?",
                    "document_ids": [first_id],
                },
            )
            self.assertEqual(answer.status_code, 200, answer.text)
            self.assertEqual(answer.json()["status"], "answered")
            self.assertEqual(answer.json()["citations"][0]["document_id"], first_id)
            self.assertEqual(answer.json()["citations"][0]["page"], 1)

            scoped_answer = client.post(
                "/ask",
                json={
                    "question": "What days does support operate?",
                    "document_ids": [second_id],
                },
            )
            self.assertEqual(scoped_answer.status_code, 200)
            self.assertEqual(
                scoped_answer.json()["citations"][0]["document_id"],
                second_id,
            )

            self.assertFalse(any((self.data_dir / "tmp").iterdir()))
            self.assertEqual(len(model_loads), 1)
            stored_files = {
                path.name
                for path in (
                    self.data_dir / "documents" / first_id
                ).iterdir()
            }
            self.assertEqual(
                stored_files,
                {"chunks.json", "source.pdf", "vectors.faiss"},
            )
            self.assertEqual(
                (self.data_dir / "documents" / first_id / "source.pdf")
                .stat()
                .st_mode
                & 0o777,
                0o600,
            )
            self.assertEqual(
                (self.data_dir / "documents.json").stat().st_mode & 0o777,
                0o600,
            )

        restarted = self.make_app()
        with TestClient(restarted) as client:
            restored = client.get("/documents")
            self.assertEqual(restored.status_code, 200)
            self.assertEqual(
                {item["document_id"] for item in restored.json()},
                {first_id, second_id},
            )
            answer = client.post(
                "/ask",
                json={
                    "question": "How many annual leave days?",
                    "document_ids": [first_id],
                },
            )
            self.assertEqual(answer.status_code, 200)
            self.assertEqual(answer.json()["status"], "answered")

    def test_pdf_errors_oversize_and_file_type(self):
        with TestClient(self.make_app()) as client:
            bad_extension = self.upload(
                client,
                "evidence",
                filename="policy.txt",
            )
            self.assertEqual(bad_extension.status_code, 415)

            invalid_pdf = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "broken.pdf",
                        b"not a pdf",
                        "application/pdf",
                    )
                },
            )
            self.assertEqual(invalid_pdf.status_code, 422)
            self.assertEqual(
                invalid_pdf.json()["detail"]["code"],
                "invalid_pdf",
            )

            empty_pdf = self.upload(client, "EMPTY_PDF")
            self.assertEqual(empty_pdf.status_code, 422)
            self.assertEqual(
                empty_pdf.json()["detail"]["code"],
                "empty_document",
            )

            oversized = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "large.pdf",
                        b"%PDF-" + b"x" * (70 * 1024),
                        "application/pdf",
                    )
                },
            )
            self.assertEqual(oversized.status_code, 413)

            body_too_large = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "large.pdf",
                        b"%PDF-" + b"x" * (140 * 1024),
                        "application/pdf",
                    )
                },
            )
            self.assertEqual(body_too_large.status_code, 413)
            self.assertEqual(
                body_too_large.json()["detail"]["code"],
                "upload_too_large",
            )

            unreadable = self.upload(client, "BAD_PDF")
            self.assertEqual(unreadable.status_code, 422)
            self.assertEqual(
                unreadable.json()["detail"]["code"],
                "invalid_pdf",
            )
            self.assertFalse(any((self.data_dir / "tmp").iterdir()))

    def test_missing_document_and_question_validation(self):
        with TestClient(self.make_app()) as client:
            missing = client.post(
                "/ask",
                json={
                    "question": "A question",
                    "document_ids": ["0" * 32],
                },
            )
            self.assertEqual(missing.status_code, 404)

            no_scope = client.post(
                "/ask",
                json={"question": "A question", "document_ids": []},
            )
            self.assertEqual(no_scope.status_code, 422)

            blank_question = client.post(
                "/ask",
                json={"question": "   ", "document_ids": ["0" * 32]},
            )
            self.assertEqual(blank_question.status_code, 422)

            too_long = client.post(
                "/ask",
                json={
                    "question": "q" * 101,
                    "document_ids": ["0" * 32],
                },
            )
            self.assertEqual(too_long.status_code, 422)

    def test_absence_of_calibration_is_explicit(self):
        uncalibrated_settings = Settings(
            data_dir=self.data_dir,
            max_upload_bytes=64 * 1024,
            calibration_file=None,
        )
        with TestClient(
            self.make_app(settings=uncalibrated_settings)
        ) as client:
            uploaded = self.upload(client, "A document contains evidence.")
            answer = client.post(
                "/ask",
                json={
                    "question": "What is in this document?",
                    "document_ids": [uploaded.json()["document_id"]],
                },
            )
            self.assertEqual(answer.status_code, 200)
            self.assertEqual(
                answer.json()["status"],
                "calibration_required",
            )
            self.assertEqual(self.generator.answer_calls, 0)

    def test_unavailable_embedding_model_is_reported_by_health(self):
        app = create_app(
            settings=self.settings,
            embedding_model_factory=lambda model_id: self._raise_unavailable(),
            generator_factory=lambda: self.generator,
            pdf_extractor=extract_fake_pdf,
        )
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 503)
            self.assertEqual(health.json()["status"], "unavailable")

            documents = client.get("/documents")
            self.assertEqual(documents.status_code, 503)
            self.assertEqual(
                documents.json()["detail"]["code"],
                "embedding_model_unavailable",
            )

    def test_unavailable_generator_is_reported_by_api_and_health(self):
        class FailingGenerator(FakeGenerator):
            def answer(self, question, evidence):
                raise OSError("Qwen weights unavailable")

        self.generator = FailingGenerator()
        with TestClient(self.make_app()) as client:
            uploaded = self.upload(client, "A document contains evidence.")
            response = client.post(
                "/ask",
                json={
                    "question": "What is in this document?",
                    "document_ids": [uploaded.json()["document_id"]],
                },
            )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["detail"]["code"],
                "generation_model_unavailable",
            )
            health = client.get("/health")
            self.assertEqual(health.status_code, 503)
            self.assertEqual(health.json()["generation_model"], "unavailable")

    @staticmethod
    def _raise_unavailable():
        raise OSError("embedding weights unavailable")


if __name__ == "__main__":
    unittest.main()
