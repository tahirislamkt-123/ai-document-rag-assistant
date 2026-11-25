import { NextResponse } from "next/server";

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

function backendUrl(path: string): URL {
  const base = process.env.RAG_API_BASE_URL?.trim() || DEFAULT_BACKEND_URL;
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  const parsedBase = new URL(normalizedBase);
  if (
    parsedBase.protocol !== "http:" ||
    !["127.0.0.1", "localhost", "[::1]"].includes(parsedBase.hostname) ||
    parsedBase.username ||
    parsedBase.password
  ) {
    throw new Error("RAG_API_BASE_URL must point to a loopback HTTP address");
  }
  return new URL(path.replace(/^\/+/, ""), parsedBase);
}

export async function proxyJson(
  path: string,
  init?: RequestInit,
): Promise<NextResponse> {
  let response: Response;
  try {
    response = await fetch(backendUrl(path), {
      ...init,
      cache: "no-store",
    });
  } catch (error) {
    console.error("FastAPI proxy request failed", error);
    return NextResponse.json(
      { detail: { code: "backend_unavailable" } },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }

  const body = await response.text();
  return new NextResponse(body, {
    status: response.status,
    headers: {
      "Content-Type":
        response.headers.get("content-type") ?? "application/json",
      "Cache-Control": "no-store",
    },
  });
}

export async function proxyUpload(request: Request): Promise<NextResponse> {
  const contentType = request.headers.get("content-type");
  if (!contentType?.toLowerCase().startsWith("multipart/form-data;")) {
    return NextResponse.json(
      { detail: { code: "multipart_form_required" } },
      { status: 415 },
    );
  }

  const body = await readBoundedBody(
    request,
    uploadLimit() + 64 * 1024,
    "upload_too_large",
  );
  if (body instanceof NextResponse) return body;

  return proxyJson("documents/upload", {
    method: "POST",
    headers: { "Content-Type": contentType },
    body,
  });
}

export async function proxyAsk(request: Request): Promise<NextResponse> {
  const contentType = request.headers.get("content-type");
  if (!contentType?.toLowerCase().startsWith("application/json")) {
    return NextResponse.json(
      { detail: { code: "json_body_required" } },
      { status: 415 },
    );
  }

  const body = await readBoundedBody(
    request,
    64 * 1024,
    "request_too_large",
  );
  if (body instanceof NextResponse) return body;

  return proxyJson("ask", {
    method: "POST",
    headers: { "Content-Type": contentType },
    body,
  });
}

export async function proxyPdf(
  documentId: string,
): Promise<Response> {
  if (!/^[a-f0-9]{32}$/.test(documentId)) {
    return NextResponse.json(
      { detail: { code: "document_pdf_not_found" } },
      { status: 404 },
    );
  }

  let response: Response;
  try {
    response = await fetch(
      backendUrl(`documents/${documentId}/file`),
      { cache: "no-store" },
    );
  } catch (error) {
    console.error("FastAPI PDF proxy request failed", error);
    return NextResponse.json(
      { detail: { code: "backend_unavailable" } },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }

  if (!response.ok) {
    const body = await response.text();
    return new NextResponse(body, {
      status: response.status,
      headers: {
        "Content-Type":
          response.headers.get("content-type") ?? "application/json",
        "Cache-Control": "no-store",
      },
    });
  }

  const headers = new Headers();
  for (const name of [
    "content-type",
    "content-length",
    "content-disposition",
    "cache-control",
    "x-content-type-options",
    "cross-origin-resource-policy",
    "accept-ranges",
  ]) {
    const value = response.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set("Content-Security-Policy", "default-src 'none'; sandbox");
  headers.set("X-Frame-Options", "SAMEORIGIN");
  return new Response(response.body, {
    status: response.status,
    headers,
  });
}

function uploadLimit(): number {
  const configured = Number(process.env.RAG_MAX_UPLOAD_BYTES);
  return Number.isSafeInteger(configured) && configured > 0
    ? configured
    : 10 * 1024 * 1024;
}

async function readBoundedBody(
  request: Request,
  limit: number,
  errorCode: string,
): Promise<ArrayBuffer | NextResponse> {
  const declaredLength = request.headers.get("content-length");
  if (declaredLength !== null) {
    if (!/^\d+$/.test(declaredLength)) {
      return NextResponse.json(
        { detail: { code: "invalid_content_length" } },
        { status: 400 },
      );
    }
    if (Number(declaredLength) > limit) {
      return tooLarge(errorCode);
    }
  }
  if (!request.body) {
    return NextResponse.json(
      { detail: { code: "request_body_required" } },
      { status: 400 },
    );
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  let chunkCount = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunkCount += 1;
      totalBytes += value.byteLength;
      if (totalBytes > limit || chunkCount > 1024) {
        void reader
          .cancel()
          .catch((error) =>
            console.error("Could not stop oversized API request", error),
          );
        return tooLarge(errorCode);
      }
      chunks.push(value);
    }
  } catch (error) {
    console.error("Could not read bounded API request", error);
    return NextResponse.json(
      { detail: { code: "invalid_request_body" } },
      { status: 400 },
    );
  }

  const buffer = new ArrayBuffer(totalBytes);
  const body = new Uint8Array(buffer);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer;
}

function tooLarge(code: string): NextResponse {
  return NextResponse.json(
    { detail: { code } },
    { status: 413, headers: { "Cache-Control": "no-store" } },
  );
}
