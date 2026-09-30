import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  deleteDocument as deleteDocumentRequest,
  getDocumentChunks,
  listDocuments,
  rebuildIndex,
  uploadDocuments,
} from "@/lib/api";
import type { DocumentRecord, RetrievedChunk, UploadResponse } from "@/types/api";

export interface UploadState {
  uploading: boolean;
  progress: number;
  filenames: string[];
}

const IDLE_UPLOAD: UploadState = { uploading: false, progress: 0, filenames: [] };

/** Document registry CRUD: list, multi-file upload, delete, rebuild, chunk peek. */
export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [uploadState, setUploadState] = useState<UploadState>(IDLE_UPLOAD);
  const [rebuilding, setRebuilding] = useState(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const response = await listDocuments();
      if (!mounted.current) return;
      setDocuments(response.documents);
      setError(null);
    } catch (cause) {
      if (!mounted.current) return;
      setError(
        cause instanceof ApiError
          ? cause
          : new ApiError(cause instanceof Error ? cause.message : String(cause)),
      );
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const uploadFiles = useCallback(
    async (files: File[]): Promise<UploadResponse | null> => {
      if (!files.length) return null;
      setUploadState({ uploading: true, progress: 0, filenames: files.map((file) => file.name) });
      try {
        const response = await uploadDocuments(files, (progress) => {
          if (!mounted.current) return;
          setUploadState((current) => ({ ...current, progress }));
        });
        if (mounted.current) {
          setDocuments((current) => mergeById(response.created, current));
        }
        return response;
      } catch (cause) {
        throw cause instanceof ApiError
          ? cause
          : new ApiError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        if (mounted.current) setUploadState(IDLE_UPLOAD);
      }
    },
    [],
  );

  const remove = useCallback(async (id: string) => {
    const response = await deleteDocumentRequest(id);
    if (mounted.current) {
      setDocuments((current) => current.filter((doc) => doc.id !== id));
    }
    return response;
  }, []);

  const rebuild = useCallback(async () => {
    setRebuilding(true);
    try {
      const response = await rebuildIndex();
      if (mounted.current) await refresh();
      return response;
    } finally {
      if (mounted.current) setRebuilding(false);
    }
  }, [refresh]);

  const loadChunks = useCallback(
    async (id: string, limit = 8): Promise<RetrievedChunk[]> => getDocumentChunks(id, limit),
    [],
  );

  return {
    documents,
    loading,
    error,
    upload: uploadState,
    rebuilding,
    refresh,
    uploadFiles,
    deleteDocument: remove,
    rebuildIndex: rebuild,
    loadChunks,
  };
}

/** Keep server order (newest first) but avoid duplicating re-indexed documents. */
function mergeById(incoming: DocumentRecord[], current: DocumentRecord[]): DocumentRecord[] {
  const byId = new Map(current.map((doc) => [doc.id, doc]));
  incoming.forEach((doc) => byId.set(doc.id, doc));
  return Array.from(byId.values()).sort((a, b) => b.uploaded_at.localeCompare(a.uploaded_at));
}
