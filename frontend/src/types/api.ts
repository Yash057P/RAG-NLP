/**
 * TypeScript mirrors of the Pydantic schemas in `backend/models/schemas.py`.
 * Keep both in sync when the API changes.
 */

export type StageStatus =
  | "pending"
  | "running"
  | "success"
  | "error"
  | "cached"
  | "skipped";

export type DocumentStatus = "pending" | "processing" | "ready" | "error";

export interface PipelineStage {
  key: string;
  label: string;
  description: string;
  status: StageStatus;
  duration_ms: number | null;
  detail: string | null;
  metrics: Record<string, string | number | boolean | null>;
}

export interface PipelineTrace {
  id: string;
  operation: "ingestion" | "query" | "rebuild";
  status: "running" | "success" | "error";
  stages: PipelineStage[];
  total_duration_ms: number;
  started_at: string;
}

export interface DocumentRecord {
  id: string;
  filename: string;
  extension: string;
  size_bytes: number;
  size_display: string;
  status: DocumentStatus;
  uploaded_at: string;
  updated_at: string;
  chunk_count: number;
  page_count: number | null;
  char_count: number;
  word_count: number;
  fingerprint: string;
  content_type: string | null;
  stored_filename: string;
  error: string | null;
  processing_ms: number;
  embedding_model: string;
  stages: PipelineStage[];
}

export interface FailedUpload {
  filename: string;
  reason: string;
  code: string;
}

export interface UploadResponse {
  created: DocumentRecord[];
  failed: FailedUpload[];
  pipeline: PipelineTrace | null;
  message: string;
}

export interface DocumentListResponse {
  documents: DocumentRecord[];
  total: number;
}

export interface DeleteResponse {
  deleted_id: string;
  filename: string;
  chunks_removed: number;
  message: string;
}

export interface RebuildResponse {
  pipeline: PipelineTrace;
  documents_processed: number;
  total_chunks: number;
  message: string;
}

export interface RetrievedChunk {
  id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  text: string;
  snippet: string;
  score: number;
  distance: number | null;
  page: number | null;
  char_start: number | null;
  char_end: number | null;
  token_estimate: number;
  is_reference: boolean;
}

export interface RetrievalResult {
  query: string;
  chunks: RetrievedChunk[];
  top_k: number;
  total_candidates: number;
  top_score: number;
  average_score: number;
  grounding_coverage: number;
  embedding_ms: number;
  search_ms: number;
  total_ms: number;
  embedding_model: string;
  vector_store_backend: string;
}

export interface Source {
  document_id: string;
  filename: string;
  chunk_index: number;
  page: number | null;
  snippet: string;
  score: number;
}

export interface LLMInfo {
  provider: string;
  model: string;
  available: boolean;
  fallback_used: boolean;
  detail: string | null;
}

export interface AnswerResponse {
  id: string;
  question: string;
  answer: string;
  grounded: boolean;
  abstained: boolean;
  confidence: number;
  confidence_label: string;
  confidence_breakdown: Record<string, number>;
  sources: Source[];
  retrieved_chunks: RetrievedChunk[];
  retrieval: RetrievalResult;
  llm: LLMInfo;
  prompt: string;
  pipeline: PipelineTrace | null;
  response_time_ms: number;
  created_at: string;
}

export interface HistoryEntry {
  id: string;
  question: string;
  answer: string;
  confidence: number;
  confidence_label: string;
  sources: Source[];
  response_time_ms: number;
  created_at: string;
  llm_model: string;
  top_score: number;
}

export interface HistoryResponse {
  entries: HistoryEntry[];
  total: number;
}

export interface Stats {
  total_documents: number;
  documents_ready: number;
  documents_error: number;
  total_chunks: number;
  total_size_bytes: number;
  total_size_display: string;
  embedding_model: string;
  embedding_provider: string;
  embedding_dimension: number;
  vector_store_status: string;
  vector_store_backend: string;
  collection_name: string;
  questions_asked: number;
  average_response_time_ms: number;
  fastest_response_ms: number | null;
  slowest_response_ms: number | null;
  last_question_at: string | null;
  last_indexed_at: string | null;
  llm_provider: string;
  llm_model: string;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
}

export interface ComponentHealth {
  name: string;
  status: "ok" | "degraded" | "error" | "unknown";
  detail: string;
}

export interface PipelineStageDefinition {
  key: string;
  label: string;
  description: string;
  icon: string;
}

export interface SystemStatus {
  status: "ok" | "degraded" | "error";
  app_name: string;
  version: string;
  environment: string;
  llm: ComponentHealth;
  embeddings: ComponentHealth;
  vector_store: ComponentHealth;
  pipeline_stages: PipelineStageDefinition[];
  example_questions: string[];
  config: {
    app_name: string;
    version: string;
    environment: string;
    api_prefix: string;
    chunking: { chunk_size: number; chunk_overlap: number };
    retrieval: {
      top_k: number;
      min_similarity: number;
      grounding_threshold: number;
    };
    upload: { max_size_mb: number; allowed_extensions: string[] };
  };
}

export interface MessageResponse {
  message: string;
  success: boolean;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  details: Record<string, unknown>;
}
