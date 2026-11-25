export interface DocumentRecord {
  document_id: string;
  filename: string;
  uploaded_at: string;
  page_count: number;
  chunk_count: number;
  source_available: boolean;
}

export interface Citation {
  citation: number;
  document_id: string;
  page: number;
  chunk_id: string;
}

export interface RetrievalAssessment {
  sufficient: boolean;
  empirical_precision: number | null;
  calibration_id: string;
  reason: string | null;
}

export interface AskResponse {
  answer: string;
  status: "answered" | "abstained" | "calibration_required";
  citations: Citation[];
  abstention_reason: string | null;
  retrieval_assessment: RetrievalAssessment | null;
}

export interface HealthResponse {
  status: "ready" | "unavailable";
  embedding_model: "loaded" | "unavailable";
  generation_model: "lazy" | "unavailable";
  document_count: number;
  error_code?: string | null;
}

export type ChatMessage =
  | {
      id: string;
      role: "user";
      text: string;
    }
  | {
      id: string;
      role: "assistant";
      response?: AskResponse;
      error?: string;
    };
