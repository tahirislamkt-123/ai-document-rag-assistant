import type {
  AskResponse,
  DocumentRecord,
  HealthResponse,
} from "@/lib/types";

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

interface ApiErrorShape {
  detail?: {
    code?: string;
    message?: string;
  } | string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function requestJson<T>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(
      "The local document service is unavailable. Check that the app is running and try again.",
      0,
    );
  }

  const payload = (await response.json().catch(() => null)) as
    | (T & ApiErrorShape)
    | null;
  if (!response.ok) {
    const detail = payload && "detail" in payload ? payload.detail : undefined;
    const code = typeof detail === "object" ? detail?.code : undefined;
    const message = typeof detail === "object" ? detail?.message : undefined;
    throw new ApiError(
      message ?? errorMessage(code, response.status),
      response.status,
      code,
    );
  }
  if (payload === null) {
    throw new ApiError("The service returned an invalid response.", response.status);
  }
  return payload;
}

function errorMessage(code: string | undefined, status: number): string {
  const known: Record<string, string> = {
    invalid_pdf: "This PDF could not be read. Try a valid, unencrypted PDF.",
    empty_document: "No extractable text was found in this PDF.",
    upload_too_large: "This PDF exceeds the 10 MiB upload limit.",
    selected_document_not_found:
      "A selected document is no longer available. Refresh the document list.",
    calibration_required:
      "Answering is disabled until a retrieval calibration file is configured.",
    embedding_model_unavailable:
      "The embedding model is unavailable. Check the local backend setup.",
    generation_model_unavailable:
      "The answer model is unavailable. Check that its weights are installed.",
    backend_unavailable:
      "The local document service is unavailable. Start the FastAPI backend and retry.",
  };
  return (
    (code && known[code]) ||
    (status === 413
      ? "The request is too large."
      : status >= 500
        ? "The local service encountered an error. Please try again."
        : "The request could not be completed. Check the file or input and retry.")
  );
}

export function getDocuments(): Promise<DocumentRecord[]> {
  return requestJson<DocumentRecord[]>("/api/documents");
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/api/health");
}

export function uploadDocument(file: File): Promise<DocumentRecord> {
  const form = new FormData();
  form.set("file", file);
  return requestJson<DocumentRecord>("/api/documents/upload", {
    method: "POST",
    body: form,
  });
}

export function askQuestion(
  question: string,
  documentIds: string[],
): Promise<AskResponse> {
  return requestJson<AskResponse>("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_ids: documentIds }),
  });
}
