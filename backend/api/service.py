import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.documents.chunking import chunk_pages
from backend.documents.pdf import extract_pdf_pages
from backend.rag.answerability import (
    CalibrationExample,
    CalibratedRetrievalGate,
)
from backend.rag.generator import QwenGenerator
from backend.rag.pipeline import RAGPipeline
from backend.rag.retriever import FaissRetriever


class DocumentNotFoundError(LookupError):
    pass


class EmptyDocumentError(ValueError):
    pass


class DataStoreError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    def __init__(self, component):
        super().__init__("{} model is unavailable".format(component))
        self.component = component


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(".data")
    embedding_model_id: str = FaissRetriever.MODEL_ID
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pages: int = 200
    max_text_characters: int = 1000000
    chunk_words: int = 250
    overlap_words: int = 40
    top_k: int = 3
    max_prompt_tokens: int = 1536
    max_question_characters: int = 2000
    max_selected_documents: int = 20
    calibration_file: Optional[Path] = None

    def __post_init__(self):
        if self.max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be positive")
        if self.max_pages < 1 or self.max_text_characters < 1:
            raise ValueError("PDF limits must be positive")
        if (
            self.chunk_words < 1
            or not 0 <= self.overlap_words < self.chunk_words
        ):
            raise ValueError("Invalid chunking settings")
        if self.top_k < 1 or self.max_prompt_tokens < 1:
            raise ValueError("Retrieval and prompt limits must be positive")
        if (
            self.max_question_characters < 1
            or self.max_selected_documents < 1
        ):
            raise ValueError("Question and document limits must be positive")

    @classmethod
    def from_environment(cls):
        calibration_path = os.environ.get("RAG_CALIBRATION_FILE")
        return cls(
            data_dir=Path(os.environ.get("RAG_DATA_DIR", ".data")),
            embedding_model_id=os.environ.get(
                "RAG_EMBEDDING_MODEL",
                FaissRetriever.MODEL_ID,
            ),
            max_upload_bytes=int(
                os.environ.get("RAG_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
            ),
            max_pages=int(os.environ.get("RAG_MAX_PAGES", 200)),
            max_text_characters=int(
                os.environ.get("RAG_MAX_TEXT_CHARACTERS", 1000000)
            ),
            chunk_words=int(os.environ.get("RAG_CHUNK_WORDS", 250)),
            overlap_words=int(os.environ.get("RAG_OVERLAP_WORDS", 40)),
            top_k=int(os.environ.get("RAG_TOP_K", 3)),
            max_prompt_tokens=int(
                os.environ.get("RAG_MAX_PROMPT_TOKENS", 1536)
            ),
            max_question_characters=int(
                os.environ.get("RAG_MAX_QUESTION_CHARACTERS", 2000)
            ),
            max_selected_documents=int(
                os.environ.get("RAG_MAX_SELECTED_DOCUMENTS", 20)
            ),
            calibration_file=(
                Path(calibration_path) if calibration_path else None
            ),
        )


class DocumentStore:
    MANIFEST_NAME = "documents.json"

    def __init__(
        self,
        data_dir,
        embedding_model,
        settings,
        pdf_extractor=extract_pdf_pages,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.settings = settings
        self.embedding_model = embedding_model
        self.pdf_extractor = pdf_extractor
        self.documents_dir = self.data_dir / "documents"
        self.staging_dir = self.data_dir / ".staging"
        self.manifest_path = self.data_dir / self.MANIFEST_NAME
        self.documents = {}
        self.retrievers = {}
        self._lock = threading.RLock()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(self.data_dir), 0o700)
        os.chmod(str(self.documents_dir), 0o700)
        os.chmod(str(self.staging_dir), 0o700)

    def load(self):
        if not self.manifest_path.exists():
            return
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, list):
            raise DataStoreError("Document manifest must be a list")

        loaded_documents = {}
        loaded_retrievers = {}
        for metadata in manifest:
            if not isinstance(metadata, dict):
                raise DataStoreError("Invalid document metadata in manifest")
            document_id = metadata.get("document_id")
            if not self._valid_document_id(document_id):
                raise DataStoreError("Invalid document ID in manifest")
            if document_id in loaded_documents:
                raise DataStoreError("Duplicate document ID in manifest")
            required_fields = {
                "filename",
                "uploaded_at",
                "page_count",
                "chunk_count",
            }
            if not required_fields.issubset(metadata):
                raise DataStoreError("Incomplete document metadata in manifest")
            document_dir = self.documents_dir / document_id
            chunks_path = document_dir / "chunks.json"
            index_path = document_dir / "vectors.faiss"
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            if (
                not isinstance(chunks, list)
                or not chunks
                or any(
                    not isinstance(item, dict)
                    or item.get("document_id") != document_id
                    for item in chunks
                )
                or len(chunks) != metadata["chunk_count"]
            ):
                raise DataStoreError(
                    "Stored chunks do not match document metadata"
                )
            retriever = FaissRetriever.load(
                chunks,
                index_path,
                model=self.embedding_model,
            )
            loaded_documents[document_id] = metadata
            loaded_retrievers[document_id] = retriever

        with self._lock:
            self.documents = loaded_documents
            self.retrievers = loaded_retrievers

    def list_documents(self):
        with self._lock:
            return [
                {
                    **metadata,
                    "source_available": self._source_is_available(
                        metadata["document_id"]
                    ),
                }
                for metadata in sorted(
                    self.documents.values(),
                    key=lambda item: (
                        item["uploaded_at"],
                        item["document_id"],
                    ),
                )
            ]

    def get_source_pdf(self, document_id):
        if not self._valid_document_id(document_id):
            raise DocumentNotFoundError("Document does not exist")
        with self._lock:
            metadata = self.documents.get(document_id)
            if metadata is None:
                raise DocumentNotFoundError("Document does not exist")
            pdf_path = self.documents_dir / document_id / "source.pdf"
            if not self._source_is_available(document_id):
                raise DocumentNotFoundError("Original PDF is unavailable")
            return pdf_path, metadata["filename"]

    def _source_is_available(self, document_id):
        pdf_path = self.documents_dir / document_id / "source.pdf"
        document_dir = self.documents_dir / document_id
        return (
            pdf_path.is_file()
            and not pdf_path.is_symlink()
            and pdf_path.resolve().parent == document_dir.resolve()
        )

    def ingest_pdf(self, path, filename):
        pages = self.pdf_extractor(
            path,
            max_pages=self.settings.max_pages,
            max_text_characters=self.settings.max_text_characters,
        )
        document_id = uuid.uuid4().hex
        chunks = chunk_pages(
            pages,
            document_id=document_id,
            chunk_words=self.settings.chunk_words,
            overlap_words=self.settings.overlap_words,
        )
        if not chunks:
            raise EmptyDocumentError(
                "PDF contains no extractable text to index"
            )

        display_name = self._safe_filename(filename)
        metadata = {
            "document_id": document_id,
            "filename": display_name,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "source_available": True,
        }
        retriever = FaissRetriever(
            chunks,
            model=self.embedding_model,
        )

        stage = Path(
            tempfile.mkdtemp(prefix="document-", dir=str(self.staging_dir))
        )
        os.chmod(str(stage), 0o700)
        final_dir = self.documents_dir / document_id
        try:
            (stage / "chunks.json").write_text(
                json.dumps(chunks, ensure_ascii=False),
                encoding="utf-8",
            )
            os.chmod(str(stage / "chunks.json"), 0o600)
            shutil.copyfile(path, stage / "source.pdf")
            os.chmod(str(stage / "source.pdf"), 0o600)
            retriever.save(stage / "vectors.faiss")
            os.chmod(str(stage / "vectors.faiss"), 0o600)
            with self._lock:
                os.replace(str(stage), str(final_dir))
                self.documents[document_id] = metadata
                self.retrievers[document_id] = retriever
                try:
                    self._write_manifest()
                except OSError:
                    self.documents.pop(document_id, None)
                    self.retrievers.pop(document_id, None)
                    shutil.rmtree(str(final_dir), ignore_errors=True)
                    raise
        except (OSError, ValueError):
            if stage.exists():
                shutil.rmtree(str(stage), ignore_errors=True)
            raise
        finally:
            if stage.exists():
                shutil.rmtree(str(stage), ignore_errors=True)

        return dict(metadata)

    def search(self, question, top_k, document_ids):
        selected_ids = list(dict.fromkeys(document_ids))
        if not selected_ids:
            raise ValueError("At least one document ID is required")
        with self._lock:
            missing = [
                document_id
                for document_id in selected_ids
                if document_id not in self.retrievers
            ]
            if missing:
                raise DocumentNotFoundError(
                    "One or more selected documents do not exist"
                )
            retrievers = [
                self.retrievers[document_id]
                for document_id in selected_ids
            ]

        try:
            query_vector = self.embedding_model.encode_query(
                [question],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ModelUnavailableError("embedding") from error
        candidates = []
        for order, retriever in enumerate(retrievers):
            for hit in retriever.search_vector(query_vector, top_k):
                candidates.append((hit["score"], order, hit))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in candidates[:top_k]]

    def get_document_ids(self, document_ids):
        with self._lock:
            missing = [
                document_id
                for document_id in document_ids
                if document_id not in self.documents
            ]
            if missing:
                raise DocumentNotFoundError(
                    "One or more selected documents do not exist"
                )
            return [dict(self.documents[item]) for item in document_ids]

    def _write_manifest(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.data_dir),
                prefix=".documents-",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = output.name
                json.dump(
                    list(self.documents.values()),
                    output,
                    ensure_ascii=False,
                    indent=2,
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, str(self.manifest_path))
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _valid_document_id(value):
        return isinstance(value, str) and bool(
            re.fullmatch(r"[a-f0-9]{32}", value)
        )

    @staticmethod
    def _safe_filename(filename):
        basename = (
            str(filename or "document.pdf")
            .replace("\\", "/")
            .split("/")[-1]
        )
        basename = "".join(
            character
            for character in basename
            if character.isprintable() and character not in "\r\n"
        )
        return basename[:255] or "document.pdf"


class RAGService:
    def __init__(
        self,
        document_store,
        generator,
        answerability_gate,
        settings,
    ):
        self.document_store = document_store
        self.generator = generator
        self.answerability_gate = answerability_gate
        self.settings = settings

    def ask(self, question, document_ids, top_k=None):
        self.document_store.get_document_ids(document_ids)
        pipeline = RAGPipeline(
            search=lambda query, count: self.document_store.search(
                query,
                count,
                document_ids,
            ),
            generator=self.generator,
            top_k=top_k or self.settings.top_k,
            max_prompt_tokens=self.settings.max_prompt_tokens,
            answerability_gate=self.answerability_gate,
        )
        return pipeline.ask(question)


def load_answerability_gate(calibration_file):
    if calibration_file is None:
        return None
    try:
        data = json.loads(Path(calibration_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataStoreError(
            "Configured calibration file is unavailable or invalid"
        ) from error
    if not isinstance(data, list):
        raise DataStoreError("Calibration file must contain a JSON list")
    examples = []
    for item in data:
        if (
            not isinstance(item, dict)
            or "score" not in item
            or not isinstance(item.get("answerable"), bool)
        ):
            raise DataStoreError(
                "Calibration entries require numeric score and boolean answerable"
            )
        try:
            score = float(item["score"])
        except (TypeError, ValueError) as error:
            raise DataStoreError(
                "Calibration score must be numeric"
            ) from error
        examples.append(
            CalibrationExample(
                score=score,
                answerable=item["answerable"],
            )
        )
    try:
        return CalibratedRetrievalGate.fit(
            examples,
            minimum_precision=1.0,
            calibration_id=Path(calibration_file).stem,
        )
    except ValueError as error:
        raise DataStoreError("Calibration data is unusable") from error


def create_service(
    settings,
    embedding_model_factory=None,
    generator_factory=None,
    pdf_extractor=extract_pdf_pages,
):
    try:
        if embedding_model_factory is None:
            from sentence_transformers import SentenceTransformer

            embedding_model_factory = lambda model_id: SentenceTransformer(
                model_id,
                local_files_only=os.environ.get(
                    "HF_HUB_OFFLINE",
                    "0",
                ) == "1",
            )
        embedding_model = embedding_model_factory(settings.embedding_model_id)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ModelUnavailableError("embedding") from error
    store = DocumentStore(
        data_dir=settings.data_dir,
        embedding_model=embedding_model,
        settings=settings,
        pdf_extractor=pdf_extractor or extract_pdf_pages,
    )
    store.load()
    gate = load_answerability_gate(settings.calibration_file)
    if generator_factory is None:
        generator_factory = QwenGenerator
    try:
        generator = generator_factory()
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ModelUnavailableError("generation") from error
    return RAGService(
        document_store=store,
        generator=generator,
        answerability_gate=gate,
        settings=settings,
    )
