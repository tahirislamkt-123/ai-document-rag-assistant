"use client";

import {
  ArrowUp,
  BookOpen,
  Check,
  CircleAlert,
  FileText,
  LoaderCircle,
  LockKeyhole,
  Menu,
  Plus,
  Sparkles,
  X,
} from "lucide-react";
import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ApiError,
  askQuestion,
  getDocuments,
  getHealth,
  MAX_UPLOAD_BYTES,
  uploadDocument,
} from "@/lib/api";
import type {
  AskResponse,
  ChatMessage,
  DocumentRecord,
  HealthResponse,
} from "@/lib/types";

const STARTERS = [
  "Summarize the key points",
  "What are the important dates?",
  "What should I know before I sign?",
];

export function DocumentAssistant() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [uploadingName, setUploadingName] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);

  const documentsById = useMemo(
    () => new Map(documents.map((document) => [document.document_id, document])),
    [documents],
  );
  const selectedDocuments = documents.filter((document) =>
    selectedIds.has(document.document_id),
  );

  const refreshDocuments = useCallback(async () => {
    setLoadingDocuments(true);
    try {
      const [nextDocuments, nextHealth] = await Promise.all([
        getDocuments(),
        getHealth().catch(
          (): HealthResponse => ({
            status: "unavailable",
            embedding_model: "unavailable",
            generation_model: "unavailable",
            document_count: 0,
          }),
        ),
      ]);
      setDocuments(nextDocuments);
      setHealth(nextHealth);
      setSelectedIds((previous) => {
        const available = new Set(
          nextDocuments.map((document) => document.document_id),
        );
        return new Set([...previous].filter((id) => available.has(id)));
      });
      setError(null);
    } catch (reason) {
      setHealth({
        status: "unavailable",
        embedding_model: "unavailable",
        generation_model: "unavailable",
        document_count: 0,
      });
      setError(messageForError(reason));
    } finally {
      setLoadingDocuments(false);
    }
  }, []);

  useEffect(() => {
    void refreshDocuments();
  }, [refreshDocuments]);

  const startNewChat = useCallback(() => {
    setMessages([]);
    setQuestion("");
    setError(null);
    setSidebarOpen(false);
    composer.current?.focus();
  }, []);

  useEffect(() => {
    function handleShortcut(event: globalThis.KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        startNewChat();
      }
    }
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [startNewChat]);

  function toggleDocument(documentId: string) {
    setSelectedIds((previous) => {
      const next = new Set(previous);
      if (next.has(documentId)) next.delete(documentId);
      else next.add(documentId);
      return next;
    });
  }

  async function handleUpload(file: File | undefined) {
    if (!file) return;
    setError(null);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Choose a PDF file to add it to your library.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError("This PDF exceeds the 10 MiB upload limit.");
      return;
    }
    try {
      const signature = await readPdfSignature(file);
      if (signature !== "%PDF-") {
        setError("This file does not look like a PDF. Choose a valid PDF.");
        return;
      }
    } catch {
      setError("The PDF could not be read from this device.");
      return;
    }

    setUploadingName(file.name);
    try {
      const indexed = await uploadDocument(file);
      setDocuments((previous) => [
        ...previous.filter(
          (document) => document.document_id !== indexed.document_id,
        ),
        indexed,
      ]);
      setSelectedIds((previous) => new Set(previous).add(indexed.document_id));
      setHealth((previous) =>
        previous
          ? { ...previous, document_count: previous.document_count + 1 }
          : previous,
      );
    } catch (reason) {
      setError(messageForError(reason));
    } finally {
      setUploadingName(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function submitQuestion(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || selectedIds.size === 0 || asking) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: trimmed,
    };
    setMessages((previous) => [...previous, userMessage]);
    setQuestion("");
    setAsking(true);
    setError(null);
    requestAnimationFrame(() => composer.current?.focus());

    try {
      const response = await askQuestion(trimmed, [...selectedIds]);
      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          response,
        },
      ]);
    } catch (reason) {
      setMessages((previous) => [
        ...previous,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          error: messageForError(reason),
        },
      ]);
    } finally {
      setAsking(false);
      requestAnimationFrame(() => composer.current?.focus());
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitQuestion();
    }
  }

  const generationStatus =
    health?.status === "unavailable"
      ? "Service unavailable"
      : health?.generation_model === "unavailable"
        ? "Answer model unavailable"
        : "Private workspace";

  return (
    <main className="app-shell">
      {sidebarOpen && (
        <button
          aria-label="Close document sidebar"
          className="mobile-backdrop"
          onClick={() => setSidebarOpen(false)}
          type="button"
        />
      )}
      <aside
        aria-label="Document library"
        className={`sidebar${sidebarOpen ? " sidebar--open" : ""}`}
      >
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <BookOpen size={19} strokeWidth={2.2} />
          </div>
          <div>
            <span className="brand-name">lattice</span>
            <span className="brand-caption">DOCUMENT INTELLIGENCE</span>
          </div>
          <button
            aria-label="Close document sidebar"
            className="icon-button sidebar-close"
            onClick={() => setSidebarOpen(false)}
            type="button"
          >
            <X size={18} />
          </button>
        </div>

        <button className="new-chat-button" onClick={startNewChat} type="button">
          <Plus size={17} />
          <span>New conversation</span>
          <span className="shortcut">⌘ K</span>
        </button>

        <div className="library-heading">
          <span>YOUR LIBRARY</span>
          <span className="document-count">{documents.length}</span>
        </div>

        <div className="document-list" aria-live="polite">
          {loadingDocuments ? (
            <div className="library-message">
              <LoaderCircle className="spin" size={16} />
              Loading library…
            </div>
          ) : documents.length === 0 ? (
            <div className="library-empty">
              <div className="empty-file-icon">
                <FileText size={19} />
              </div>
              <p>No documents yet</p>
              <span>Upload a PDF to start asking questions.</span>
            </div>
          ) : (
            documents.map((document) => {
              const selected = selectedIds.has(document.document_id);
              return (
                <label
                  className={`document-row${selected ? " document-row--selected" : ""}`}
                  key={document.document_id}
                >
                  <input
                    checked={selected}
                    onChange={() => toggleDocument(document.document_id)}
                    type="checkbox"
                    aria-label={`Use ${document.filename} as context`}
                  />
                  <span className="document-check" aria-hidden="true">
                    {selected && <Check size={12} strokeWidth={3} />}
                  </span>
                  <span className="document-icon" aria-hidden="true">
                    <FileText size={17} />
                  </span>
                  <span className="document-copy">
                    <span className="document-name">{document.filename}</span>
                    <span className="document-meta">
                      <span className="indexed-dot" />
                      Indexed · {document.page_count}{" "}
                      {document.page_count === 1 ? "page" : "pages"}
                    </span>
                  </span>
                </label>
              );
            })
          )}
        </div>

        <div className="sidebar-bottom">
          {uploadingName && (
            <div className="upload-progress" role="status">
              <LoaderCircle className="spin" size={15} />
              <span>
                Indexing <strong>{uploadingName}</strong>…
              </span>
            </div>
          )}
          <input
            accept=".pdf,application/pdf"
            className="visually-hidden"
            disabled={uploadingName !== null}
            id="pdf-upload"
            onChange={(event) => void handleUpload(event.target.files?.[0])}
            ref={fileInput}
            type="file"
          />
          <label
            className={`upload-button${uploadingName ? " is-disabled" : ""}`}
            htmlFor="pdf-upload"
            aria-disabled={uploadingName !== null}
          >
            {uploadingName ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Plus size={17} />
            )}
            {uploadingName ? "Processing PDF" : "Add a document"}
          </label>
          <div className="privacy-note">
            <LockKeyhole size={13} />
            <span>Stored locally on this device</span>
          </div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <button
            aria-label="Open document sidebar"
            className="icon-button mobile-menu"
            onClick={() => setSidebarOpen(true)}
            type="button"
          >
            <Menu size={19} />
          </button>
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-slash">/</span>
            <strong>Document assistant</strong>
          </div>
          <div
            aria-label={`Local service status: ${generationStatus}`}
            className={`service-pill${health?.status === "unavailable" ? " service-pill--error" : ""}`}
            role="status"
          >
            <span className="service-dot" />
            <span className="service-label">{generationStatus}</span>
          </div>
        </header>

        <div className="conversation-layout">
          <div aria-live="polite" className="conversation-scroll">
            {messages.length === 0 ? (
              <section className="welcome">
                <div className="welcome-symbol">
                  <Sparkles size={21} />
                </div>
                <p className="eyebrow">YOUR DOCUMENTS, IN CONTEXT</p>
                <h1>
                  The answers are
                  <br />
                  <span>in your documents.</span>
                </h1>
                <p className="welcome-description">
                  Select a document, ask a question, and get an answer grounded
                  in its contents—with page-level sources you can verify.
                </p>
                <div className="grounding-card">
                  <div className="grounding-icon">
                    <LockKeyhole size={16} />
                  </div>
                  <div>
                    <strong>Grounded in what you select</strong>
                    <p>
                      Retrieval searches only selected PDFs. The answer model
                      sees relevant passages, not your whole library.
                    </p>
                  </div>
                </div>
                {selectedDocuments.length > 0 && (
                  <div className="starter-block">
                    <span className="starter-label">A place to start</span>
                    <div className="starter-list">
                      {STARTERS.map((starter) => (
                        <button
                          className="starter-chip"
                          key={starter}
                          onClick={() => {
                            setQuestion(starter);
                            composer.current?.focus();
                          }}
                          type="button"
                        >
                          {starter}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            ) : (
              <div className="message-list">
                {messages.map((message) =>
                  message.role === "user" ? (
                    <article className="user-message" key={message.id}>
                      <p>{message.text}</p>
                    </article>
                  ) : (
                    <AssistantMessage
                      documentsById={documentsById}
                      key={message.id}
                      message={message}
                    />
                  ),
                )}
                {asking && (
                  <div className="thinking-row" role="status">
                    <div className="assistant-avatar">
                      <Sparkles size={15} />
                    </div>
                    <span className="thinking-dots">
                      <i />
                      <i />
                      <i />
                    </span>
                    <span>Searching selected documents…</span>
                  </div>
                )}
              </div>
            )}
            {error && (
              <div className="inline-error" role="alert">
                <CircleAlert size={16} />
                <span>{error}</span>
                <button
                  aria-label="Dismiss error"
                  className="icon-button"
                  onClick={() => setError(null)}
                  type="button"
                >
                  <X size={15} />
                </button>
              </div>
            )}
          </div>

          <footer className="composer-wrap">
            <div className="selected-context" aria-live="polite">
              <span className="context-icon">
                <BookOpen size={13} />
              </span>
              {selectedDocuments.length > 0 ? (
                <>
                  <strong>{selectedDocuments.length}</strong>{" "}
                  {selectedDocuments.length === 1 ? "document" : "documents"}{" "}
                  selected
                  <span className="context-separator">·</span>
                  {selectedDocuments
                    .slice(0, 2)
                    .map((document) => document.filename)
                    .join(", ")}
                  {selectedDocuments.length > 2
                    ? ` +${selectedDocuments.length - 2}`
                    : ""}
                </>
              ) : (
                <span>Select at least one document from your library</span>
              )}
            </div>
            <form
              className="composer"
              onSubmit={(event) => void submitQuestion(event)}
            >
              <textarea
                aria-label="Ask a question about your selected documents"
                disabled={selectedIds.size === 0 || asking}
                maxLength={2000}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleComposerKeyDown}
                placeholder={
                  selectedIds.size === 0
                    ? "Select documents to start a conversation…"
                    : "Ask anything about your documents…"
                }
                ref={composer}
                rows={1}
                value={question}
              />
              <button
                aria-label={asking ? "Waiting for answer" : "Send question"}
                className="send-button"
                disabled={
                  selectedIds.size === 0 || !question.trim() || asking
                }
                type="submit"
              >
                {asking ? (
                  <LoaderCircle className="spin" size={17} />
                ) : (
                  <ArrowUp size={18} />
                )}
              </button>
            </form>
            <p className="composer-footnote">
              Answers can be wrong. Open a source to verify important details.
            </p>
          </footer>
        </div>
        <div className="screen-reader-only" aria-live="polite">
          {health?.status === "ready"
            ? "Local document service ready."
            : health?.status === "unavailable"
              ? "Local document service unavailable."
              : ""}
        </div>
      </section>
    </main>
  );
}

function AssistantMessage({
  message,
  documentsById,
}: {
  message: Extract<ChatMessage, { role: "assistant" }>;
  documentsById: Map<string, DocumentRecord>;
}) {
  const response = message.response;
  const isAnswered = response?.status === "answered";
  return (
    <article className="assistant-message">
      <div className="assistant-avatar" aria-hidden="true">
        <Sparkles size={15} />
      </div>
      <div className="assistant-content">
        <div className="assistant-heading">
          <strong>Lattice</strong>
          <span>DOCUMENT ASSISTANT</span>
        </div>
        {message.error ? (
          <div className="answer-error" role="alert">
            <CircleAlert size={16} />
            <p>{message.error}</p>
          </div>
        ) : response ? (
          <>
            <div
              className={`answer-status answer-status--${response.status}`}
              role="status"
            >
              <span className="answer-status-dot" />
              {response.status === "answered"
                ? "Grounded answer"
                : response.status === "calibration_required"
                  ? "Answering not calibrated"
                  : "Not enough evidence"}
            </div>
            <p className="answer-text">
              {renderAnswer(response.answer, response.citations, documentsById)}
            </p>
            {!isAnswered && response.abstention_reason && (
              <p className="abstention-detail">
                {response.status === "calibration_required"
                  ? "The answerability check is not configured for this service. No generated answer was returned."
                  : "The selected documents did not provide enough calibrated evidence to answer safely."}
              </p>
            )}
            {response.citations.length > 0 && (
              <div className="sources">
                <div className="sources-heading">
                  <span>SOURCES</span>
                  <span className="sources-count">
                    {response.citations.length}
                  </span>
                </div>
                <div className="source-list">
                  {response.citations.map((citation) => {
                    const document = documentsById.get(citation.document_id);
                    return (
                      <CitationReference
                        citation={citation}
                        document={document}
                        key={`${citation.document_id}:${citation.chunk_id}`}
                      />
                    );
                  })}
                </div>
              </div>
            )}
          </>
        ) : null}
      </div>
    </article>
  );
}

function CitationReference({
  citation,
  document,
}: {
  citation: AskResponse["citations"][number];
  document: DocumentRecord | undefined;
}) {
  const label = document?.filename ?? "Selected document";
  const source = (
    <>
      <span className="source-number">[{citation.citation}]</span>
      <span className="source-document">
        <FileText size={15} />
        <span>{label}</span>
      </span>
      <span className="source-page">Page {citation.page}</span>
      <code>{citation.chunk_id}</code>
      {document?.source_available && <span className="source-open">Open</span>}
    </>
  );
  const content = document?.source_available ? (
    <a
      aria-label={`Open ${label}, page ${citation.page}`}
      className="source-row source-row--link"
      href={`/api/documents/${encodeURIComponent(citation.document_id)}/file#page=${citation.page}`}
      rel="noreferrer"
      target="_blank"
    >
      {source}
    </a>
  ) : (
    <div className="source-row" aria-label={`${label}, page ${citation.page}`}>
      {source}
    </div>
  );
  return content;
}

function renderAnswer(
  answer: string,
  citations: AskResponse["citations"],
  documentsById: Map<string, DocumentRecord>,
) {
  return answer.split(/(\[\d+\])/g).map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match) return <span key={index}>{part}</span>;
    const citationNumber = Number(match[1]);
    const citation = citations.find(
      (item) => item.citation === citationNumber,
    );
    const document = citation
      ? documentsById.get(citation.document_id)
      : undefined;
    if (!citation || !document?.source_available) {
      return (
        <span className="citation-marker" key={index}>
          {part}
        </span>
      );
    }
    return (
      <a
        aria-label={`Open source ${citationNumber}, ${document.filename}, page ${citation.page}`}
        className="inline-citation"
        href={`/api/documents/${encodeURIComponent(citation.document_id)}/file#page=${citation.page}`}
        key={index}
        rel="noreferrer"
        target="_blank"
      >
        {part}
      </a>
    );
  });
}

function messageForError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return "Something went wrong. Please try again.";
}

function readPdfSignature(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        resolve(new TextDecoder().decode(reader.result));
      } else {
        reject(new Error("PDF signature could not be read"));
      }
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file.slice(0, 5));
  });
}
