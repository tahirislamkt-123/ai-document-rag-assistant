import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from backend.api.models import (
    AskRequest,
    AskResponse,
    CitationResponse,
    DocumentResponse,
    HealthResponse,
    RetrievalAssessmentResponse,
)
from backend.api.service import (
    DataStoreError,
    DocumentNotFoundError,
    EmptyDocumentError,
    ModelUnavailableError,
    RAGService,
    Settings,
    create_service,
)
from backend.documents.pdf import PDFProcessingError


logger = logging.getLogger(__name__)
UPLOAD_PATH = "/documents/upload"
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def get_rag_service(request: Request) -> RAGService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": request.app.state.startup_error_code
                or "service_unavailable"
            },
        )
    return service


class UploadLimitMiddleware:
    def __init__(self, app, max_body_bytes):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["path"] == UPLOAD_PATH and scope["method"] == "POST":
            body_limit = self.max_body_bytes
        elif scope["path"] == "/ask" and scope["method"] == "POST":
            body_limit = 64 * 1024
        else:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._reject(send, 400, "invalid_content_length")
                return
            if declared_length > body_limit:
                await self._reject(
                    send,
                    413,
                    "upload_too_large"
                    if scope["path"] == UPLOAD_PATH
                    else "request_too_large",
                )
                return

        messages = []
        total_bytes = 0
        message_count = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            message_count += 1
            if message_count > 1024:
                await self._reject(send, 413, "too_many_request_chunks")
                return
            total_bytes += len(message.get("body", b""))
            if total_bytes > body_limit:
                await self._reject(
                    send,
                    413,
                    "upload_too_large"
                    if scope["path"] == UPLOAD_PATH
                    else "request_too_large",
                )
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        position = 0

        async def replay_receive():
            nonlocal position
            if position < len(messages):
                message = messages[position]
                position += 1
                return message
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(send, status_code, code):
        body = json.dumps({"detail": {"code": code}}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


def create_app(
    settings=None,
    embedding_model_factory=None,
    generator_factory=None,
    pdf_extractor=None,
):
    settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(application):
        application.state.service = None
        application.state.startup_error = None
        application.state.startup_error_code = None
        application.state.embedding_unavailable = False
        application.state.generation_unavailable = False
        application.state.inference_semaphore = asyncio.Semaphore(1)
        try:
            application.state.service = await run_in_threadpool(
                create_service,
                settings,
                embedding_model_factory,
                generator_factory,
                pdf_extractor,
            )
        except ModelUnavailableError as error:
            logger.error("RAG model initialization failed: %s", error)
            application.state.startup_error = error
            application.state.startup_error_code = (
                "embedding_model_unavailable"
                if error.component == "embedding"
                else "generation_model_unavailable"
            )
            application.state.embedding_unavailable = (
                error.component == "embedding"
            )
            application.state.generation_unavailable = (
                error.component == "generation"
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            logger.error("RAG service initialization failed: %s", error)
            application.state.startup_error = error
            application.state.startup_error_code = (
                "document_store_unavailable"
                if isinstance(error, DataStoreError)
                else "service_initialization_failed"
            )
        yield
        application.state.service = None

    application = FastAPI(
        title="AI Document RAG Assistant",
        version="0.4.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    application.add_middleware(
        UploadLimitMiddleware,
        max_body_bytes=(
            settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        ),
    )
    application.state.settings = settings

    @application.exception_handler(DataStoreError)
    async def datastore_error_handler(request, error):
        logger.exception("Document datastore failure", exc_info=error)
        return JSONResponse(
            status_code=500,
            content={"detail": {"code": "document_store_error"}},
        )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request):
        service = request.app.state.service
        embedding_unavailable = request.app.state.embedding_unavailable
        generation_unavailable = request.app.state.generation_unavailable
        if service is None or embedding_unavailable or generation_unavailable:
            startup_error = request.app.state.startup_error
            embedding_loaded_before_failure = (
                isinstance(startup_error, ModelUnavailableError)
                and startup_error.component == "generation"
            )
            return JSONResponse(
                status_code=503,
                content=_model_to_dict(
                    HealthResponse(
                        status="unavailable",
                        embedding_model=(
                            "loaded"
                            if service is not None or embedding_loaded_before_failure
                            else "unavailable"
                        ),
                        generation_model=(
                            "unavailable" if generation_unavailable else "lazy"
                        ),
                        document_count=(
                            len(service.document_store.documents)
                            if service is not None
                            else 0
                        ),
                        error_code=(
                            "generation_model_unavailable"
                            if generation_unavailable
                            else request.app.state.startup_error_code
                        ),
                    )
                ),
            )
        return HealthResponse(
            status="ready",
            embedding_model="loaded",
            generation_model="lazy",
            document_count=len(service.document_store.documents),
        )

    @application.post(
        UPLOAD_PATH,
        status_code=201,
        response_model=DocumentResponse,
        responses={
            413: {"description": "Upload exceeds the configured size limit"},
            415: {"description": "Only PDF files are accepted"},
            422: {"description": "Invalid or empty PDF"},
            503: {"description": "Embedding model unavailable"},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "format": "binary",
                                }
                            },
                        }
                    }
                },
            }
        },
    )
    async def upload_document(
        request: Request,
        service: RAGService = Depends(get_rag_service),
    ):
        try:
            form = await request.form(
                max_files=1,
                max_fields=1,
                max_part_size=1024 * 64,
            )
        except (ValueError, OSError) as error:
            raise HTTPException(
                status_code=400,
                detail={"code": "invalid_multipart_request"},
            ) from error
        try:
            if set(form.keys()) != {"file"}:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "exactly_one_pdf_required"},
                )
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(
                    status_code=422,
                    detail={"code": "pdf_file_required"},
                )
            filename = upload.filename or ""
            if Path(filename.replace("\\", "/")).suffix.lower() != ".pdf":
                raise HTTPException(
                    status_code=415,
                    detail={"code": "pdf_extension_required"},
                )

            temporary_dir = (
                settings.data_dir.resolve() / "tmp"
            )
            temporary_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(str(temporary_dir), 0o700)
            temporary_path = None
            size = 0
            prefix = bytearray()
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".pdf",
                    dir=str(temporary_dir),
                    delete=False,
                ) as output:
                    temporary_path = output.name
                    while True:
                        chunk = await upload.read(64 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > settings.max_upload_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail={"code": "upload_too_large"},
                            )
                        if len(prefix) < 5:
                            prefix.extend(chunk[:5 - len(prefix)])
                        output.write(chunk)
                if bytes(prefix) != b"%PDF-":
                    raise HTTPException(
                        status_code=422,
                        detail={"code": "invalid_pdf"},
                    )

                async with request.app.state.inference_semaphore:
                    metadata = await run_in_threadpool(
                        service.document_store.ingest_pdf,
                        temporary_path,
                        filename,
                    )
                return DocumentResponse(**metadata)
            except PDFProcessingError as error:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "invalid_pdf", "message": str(error)},
                ) from error
            except EmptyDocumentError as error:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "empty_document", "message": str(error)},
                ) from error
            except (ImportError, OSError, RuntimeError) as error:
                logger.exception("Document ingestion failed")
                raise HTTPException(
                    status_code=503,
                    detail={"code": "embedding_or_storage_unavailable"},
                ) from error
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
                await upload.close()
        finally:
            await form.close()

    @application.get(
        "/documents",
        response_model=list[DocumentResponse],
        responses={503: {"description": "Embedding model unavailable"}},
    )
    async def list_documents(
        service: RAGService = Depends(get_rag_service),
    ):
        return [
            DocumentResponse(**item)
            for item in service.document_store.list_documents()
        ]

    @application.get(
        "/documents/{document_id}/file",
        responses={404: {"description": "Original PDF is unavailable"}},
    )
    async def get_document_file(
        document_id: str,
        service: RAGService = Depends(get_rag_service),
    ):
        try:
            pdf_path, filename = service.document_store.get_source_pdf(
                document_id
            )
        except DocumentNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "document_pdf_not_found"},
            ) from error
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=filename,
            content_disposition_type="inline",
            headers={
                "Cache-Control": "private, no-store",
                "Cross-Origin-Resource-Policy": "same-origin",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @application.post(
        "/ask",
        response_model=AskResponse,
        responses={
            404: {"description": "A selected document does not exist"},
            503: {"description": "A required model is unavailable"},
        },
    )
    async def ask(
        request: Request,
        body: AskRequest,
        service: RAGService = Depends(get_rag_service),
    ):
        if len(body.question.strip()) == 0:
            raise HTTPException(
                status_code=422,
                detail={"code": "question_must_not_be_blank"},
            )
        if len(body.question) > settings.max_question_characters:
            raise HTTPException(
                status_code=422,
                detail={"code": "question_too_long"},
            )
        if len(body.document_ids) > settings.max_selected_documents:
            raise HTTPException(
                status_code=422,
                detail={"code": "too_many_selected_documents"},
            )

        try:
            async with request.app.state.inference_semaphore:
                result = await run_in_threadpool(
                    service.ask,
                    body.question,
                    body.document_ids,
                    body.top_k,
                )
        except DocumentNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "selected_document_not_found"},
            ) from error
        except ModelUnavailableError as error:
            logger.error("RAG model unavailable: %s", error)
            code = (
                "embedding_model_unavailable"
                if error.component == "embedding"
                else "generation_model_unavailable"
            )
            request.app.state.startup_error_code = code
            request.app.state.embedding_unavailable = (
                error.component == "embedding"
            )
            request.app.state.generation_unavailable = (
                error.component == "generation"
            )
            raise HTTPException(
                status_code=503,
                detail={"code": code},
            ) from error
        except (ImportError, OSError, RuntimeError) as error:
            logger.exception("RAG inference failed")
            request.app.state.generation_unavailable = True
            request.app.state.startup_error_code = (
                "generation_model_unavailable"
            )
            raise HTTPException(
                status_code=503,
                detail={"code": "generation_model_unavailable"},
            ) from error

        citations_by_number = {
            source.citation: source
            for source in result.sources
        }
        citations = [
            CitationResponse(
                citation=number,
                document_id=citations_by_number[number].document_id,
                page=citations_by_number[number].page,
                chunk_id=citations_by_number[number].chunk_id,
            )
            for number in result.citations
            if number in citations_by_number
        ]
        if result.abstention_reason == "answerability_not_calibrated":
            status = "calibration_required"
        elif result.abstained:
            status = "abstained"
        else:
            status = "answered"

        assessment = result.retrieval_assessment
        return AskResponse(
            answer=result.answer,
            status=status,
            citations=citations,
            abstention_reason=result.abstention_reason,
            retrieval_assessment=(
                RetrievalAssessmentResponse(
                    sufficient=assessment.sufficient,
                    empirical_precision=assessment.empirical_precision,
                    calibration_id=assessment.calibration_id,
                    reason=assessment.reason,
                )
                if assessment is not None
                else None
            ),
        )

    return application


app = create_app()
