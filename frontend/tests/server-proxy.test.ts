import { afterEach, describe, expect, it, vi } from "vitest";
import {
  proxyAsk,
  proxyJson,
  proxyPdf,
  proxyUpload,
} from "@/lib/server-proxy";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe("server-side FastAPI proxy", () => {
  it("forwards JSON responses without caching", async () => {
    vi.stubEnv("RAG_API_BASE_URL", "http://127.0.0.1:8123");
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([{ document_id: "id" }]), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await proxyJson("/documents");

    expect(fetchMock).toHaveBeenCalledWith(
      new URL("http://127.0.0.1:8123/documents"),
      { cache: "no-store" },
    );
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(await response.json()).toEqual([{ document_id: "id" }]);
  });

  it("rejects oversized uploads before contacting FastAPI", async () => {
    vi.stubEnv("RAG_MAX_UPLOAD_BYTES", "4");
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const response = await proxyUpload({
      headers: new Headers({
        "content-type": "multipart/form-data; boundary=example",
      }),
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array(70 * 1024));
          controller.close();
        },
      }),
    } as Request);

    expect(response.status).toBe(413);
    expect(await response.json()).toEqual({
      detail: { code: "upload_too_large" },
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("bounds chat request bodies before forwarding", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const response = await proxyAsk(
      new Request("http://localhost/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: "x".repeat(70 * 1024) }),
      }),
    );

    expect(response.status).toBe(413);
    expect(await response.json()).toEqual({
      detail: { code: "request_too_large" },
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("validates document IDs and adds safe headers to the PDF proxy", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(new Uint8Array([37, 80, 68, 70, 45]), {
        status: 200,
        headers: {
          "Content-Type": "application/pdf",
          "Content-Disposition": 'inline; filename="source.pdf"',
          "X-Content-Type-Options": "nosniff",
          "Cache-Control": "private, no-store",
        },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const invalid = await proxyPdf("../outside");
    expect(invalid.status).toBe(404);
    expect(fetchMock).not.toHaveBeenCalled();

    const valid = await proxyPdf("a".repeat(32));
    expect(valid.headers.get("content-type")).toBe("application/pdf");
    expect(valid.headers.get("content-security-policy")).toBe(
      "default-src 'none'; sandbox",
    );
    expect(valid.headers.get("x-frame-options")).toBe("SAMEORIGIN");
    expect(Array.from(new Uint8Array(await valid.arrayBuffer()))).toEqual([
      37, 80, 68, 70, 45,
    ]);
  });
});
