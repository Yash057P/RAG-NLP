/**
 * Typed fetch wrapper for the FastAPI backend.
 *
 * The dev server proxies /api to the backend (see vite.config.ts). In
 * production set VITE_API_URL to the deployed API origin.
 */

import type {
  AnswerResponse,
  DeleteResponse,
  DocumentListResponse,
  DocumentRecord,
  HistoryResponse,
  MessageResponse,
  RebuildResponse,
  RetrievedChunk,
  Stats,
  SystemStatus,
  UploadResponse,
} from "@/types/api";

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "";
const API = `${BASE_URL}/api`;

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: Record<string, unknown>;

  constructor(
    message: string,
    code = "error",
    status = 500,
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, init);
  } catch (cause) {
    throw new ApiError(
      "Cannot reach the backend. Is the FastAPI server running on port 8000?",
      "network_error",
      0,
      { cause: String(cause) },
    );
  }

  if (!response.ok) {
    let code = `http_${response.status}`;
    let message = `Request failed with status ${response.status}`;
    let details: Record<string, unknown> = {};
    try {
      const body = await response.json();
      if (body?.error) {
        code = body.error.code ?? code;
        message = body.error.message ?? message;
        details = body.error.details ?? {};
      } else if (typeof body?.detail === "string") {
        message = body.detail;
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(message, code, response.status, details);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ------------------------------------------------------------------ system ---
export const getSystemStatus = () => request<SystemStatus>("/system/status");

export const resetSystem = () =>
  request<MessageResponse>("/system/reset", { method: "POST" });

// --------------------------------------------------------------- documents ---
export const listDocuments = () => request<DocumentListResponse>("/documents");

export const getDocument = (id: string) => request<DocumentRecord>(`/documents/${id}`);

export const getDocumentChunks = (id: string, limit = 5) =>
  request<RetrievedChunk[]>(`/documents/${id}/chunks?limit=${limit}`);

export const deleteDocument = (id: string) =>
  request<DeleteResponse>(`/documents/${id}`, { method: "DELETE" });

export const rebuildIndex = () =>
  request<RebuildResponse>("/documents/rebuild", { method: "POST" });

export function uploadDocuments(
  files: File[],
  onProgress?: (percent: number) => void,
): Promise<UploadResponse> {
  // XMLHttpRequest is used instead of fetch so the upload can report progress.
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/documents`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(100);
        try {
          resolve(JSON.parse(xhr.responseText) as UploadResponse);
        } catch (error) {
          reject(new ApiError("Malformed response from the server", "parse_error"));
        }
        return;
      }
      let message = `Upload failed with status ${xhr.status}`;
      let code = `http_${xhr.status}`;
      try {
        const body = JSON.parse(xhr.responseText);
        message = body?.error?.message ?? message;
        code = body?.error?.code ?? code;
      } catch {
        /* ignore */
      }
      reject(new ApiError(message, code, xhr.status));
    };
    xhr.onerror = () =>
      reject(
        new ApiError(
          "Cannot reach the backend. Is the FastAPI server running on port 8000?",
          "network_error",
        ),
      );
    xhr.send(form);
  });
}

// -------------------------------------------------------------------- chat ---
export const askQuestion = (question: string, topK?: number) =>
  request<AnswerResponse>("/chat/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(topK ? { question, top_k: topK } : { question }),
  });

export const getHistory = () => request<HistoryResponse>("/chat/history");

export const clearHistory = () =>
  request<MessageResponse>("/chat/history", { method: "DELETE" });

// ------------------------------------------------------------------- stats ---
export const getStats = () => request<Stats>("/stats");
