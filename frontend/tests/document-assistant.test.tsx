import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DocumentAssistant } from "@/components/document-assistant";
import type { AskResponse } from "@/lib/types";

const handbook = {
  document_id: "a".repeat(32),
  filename: "employee-handbook.pdf",
  uploaded_at: "2026-11-18T00:00:00+00:00",
  page_count: 12,
  chunk_count: 35,
  source_available: true,
};

const groundedAnswer: AskResponse = {
  answer: "Employees receive 25 annual leave days. [1]",
  status: "answered" as const,
  citations: [
    {
      citation: 1,
      document_id: handbook.document_id,
      page: 4,
      chunk_id: `${handbook.document_id}:p4:c1`,
    },
  ],
  abstention_reason: null,
  retrieval_assessment: {
    sufficient: true,
    empirical_precision: 1,
    calibration_id: "test-calibration",
    reason: null,
  },
};

function jsonResponse(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function mockApi(askResponse = groundedAnswer) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/documents") return jsonResponse([handbook]);
    if (url === "/api/health") {
      return jsonResponse({
        status: "ready",
        embedding_model: "loaded",
        generation_model: "lazy",
        document_count: 1,
      });
    }
    if (url === "/api/ask") return jsonResponse(askResponse);
    if (url === "/api/documents/upload") {
      const body = init?.body as FormData;
      const file = body.get("file") as File;
      return jsonResponse({
        ...handbook,
        document_id: "b".repeat(32),
        filename: file.name,
      });
    }
    return jsonResponse({ detail: { code: "not_found" } }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("DocumentAssistant", () => {
  it("sends only selected document IDs and links returned citations to their page", async () => {
    const user = userEvent.setup();
    const fetchMock = mockApi();
    render(<DocumentAssistant />);

    const selectDocument = await screen.findByRole("checkbox", {
      name: "Use employee-handbook.pdf as context",
    });
    await user.click(selectDocument);
    const composer = screen.getByRole("textbox", {
      name: "Ask a question about your selected documents",
    });
    await user.type(composer, "How much annual leave?{Enter}");

    await screen.findByText("Employees receive 25 annual leave days.");
    const askCall = fetchMock.mock.calls.find(
      ([input]) => String(input) === "/api/ask",
    );
    expect(askCall).toBeDefined();
    expect(JSON.parse(String(askCall?.[1]?.body))).toEqual({
      question: "How much annual leave?",
      document_ids: [handbook.document_id],
    });

    expect(
      screen.getByRole("link", {
        name: "Open employee-handbook.pdf, page 4",
      }),
    ).toHaveAttribute(
      "href",
      `/api/documents/${handbook.document_id}/file#page=4`,
    );
    expect(
      screen.getByText(`${handbook.document_id}:p4:c1`),
    ).toBeInTheDocument();
  });

  it("shows a clear abstention and rejects a file without a PDF signature", async () => {
    const user = userEvent.setup();
    mockApi({
      ...groundedAnswer,
      answer: "I cannot find the answer in the provided documents.",
      status: "abstained",
      citations: [],
      abstention_reason: "below_calibrated_cutoff",
    });
    render(<DocumentAssistant />);
    const selectDocument = await screen.findByRole("checkbox", {
      name: "Use employee-handbook.pdf as context",
    });
    await user.click(selectDocument);
    await user.type(
      screen.getByRole("textbox", {
        name: "Ask a question about your selected documents",
      }),
      "Where is the office?{Enter}",
    );
    expect(
      await screen.findByText("Not enough evidence"),
    ).toBeInTheDocument();

    const invalidFile = new File(["not a PDF"], "notes.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByLabelText("Add a document"), {
      target: { files: [invalidFile] },
    });
    expect(
      await screen.findByRole("alert"),
    ).toHaveTextContent("does not look like a PDF");
  });
});
