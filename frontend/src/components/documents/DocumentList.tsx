import { useCallback, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  FileText,
  Loader2,
  MoreVertical,
  RefreshCw,
  ScanText,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";

import { EmptyState } from "@/components/common/EmptyState";
import { SectionHeader } from "@/components/common/SectionHeader";
import { StatusDot } from "@/components/common/StatusDot";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ApiError } from "@/lib/api";
import { formatBytes, formatDateTime, formatDuration, relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { DocumentRecord, RetrievedChunk } from "@/types/api";

interface DocumentListProps {
  documents: DocumentRecord[];
  loading: boolean;
  rebuilding: boolean;
  onDelete: (id: string) => Promise<unknown>;
  onRebuild: () => Promise<unknown>;
  onLoadChunks: (id: string, limit?: number) => Promise<RetrievedChunk[]>;
}

const STATUS_META = {
  ready: { label: "Indexed", variant: "success" as const, icon: CheckCircle2, dot: "ok" as const },
  processing: { label: "Processing", variant: "warning" as const, icon: Loader2, dot: "busy" as const },
  pending: { label: "Queued", variant: "secondary" as const, icon: Loader2, dot: "busy" as const },
  error: { label: "Failed", variant: "destructive" as const, icon: AlertCircle, dot: "warn" as const },
};

export function DocumentList({
  documents,
  loading,
  rebuilding,
  onDelete,
  onRebuild,
  onLoadChunks,
}: DocumentListProps) {
  const totalChunks = documents.reduce((sum, doc) => sum + doc.chunk_count, 0);

  const handleRebuild = useCallback(async () => {
    try {
      const response = (await onRebuild()) as { documents_processed: number; total_chunks: number };
      toast.success("Vector store rebuilt", {
        description: `${response.documents_processed} document(s) → ${response.total_chunks} chunks re-embedded.`,
      });
    } catch (cause) {
      toast.error("Rebuild failed", {
        description: cause instanceof ApiError ? cause.message : "Unexpected error.",
      });
    }
  }, [onRebuild]);

  const handleDelete = useCallback(
    async (doc: DocumentRecord) => {
      try {
        const response = (await onDelete(doc.id)) as { chunks_removed: number };
        toast.success(`Deleted ${doc.filename}`, {
          description: `${response.chunks_removed} vector(s) removed from the store.`,
        });
      } catch (cause) {
        toast.error("Delete failed", {
          description: cause instanceof ApiError ? cause.message : "Unexpected error.",
        });
      }
    },
    [onDelete],
  );

  return (
    <div className="space-y-2.5">
      <SectionHeader
        icon={<FileText className="h-3.5 w-3.5" />}
        title="Documents"
        hint={
          documents.length
            ? `${documents.length} file${documents.length > 1 ? "s" : ""} · ${totalChunks} chunks`
            : "Nothing indexed yet"
        }
        actions={
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                size="icon-sm"
                variant="ghost"
                onClick={handleRebuild}
                disabled={rebuilding || !documents.length}
                aria-label="Rebuild vector store"
              >
                <RefreshCw className={cn("h-3.5 w-3.5", rebuilding && "animate-spin")} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Rebuild vector store from the uploads folder</TooltipContent>
          </Tooltip>
        }
      />

      {loading ? (
        <div className="space-y-2">
          {[0, 1, 2].map((index) => (
            <Skeleton key={index} className="h-[74px] w-full rounded-xl" />
          ))}
        </div>
      ) : documents.length === 0 ? (
        <EmptyState
          icon={<Database className="h-4 w-4" />}
          title="No documents yet"
          description="Upload a PDF, DOCX or TXT file to build the knowledge base. Every file is chunked, embedded and stored in ChromaDB."
        />
      ) : (
        <ul className="space-y-2">
          {documents.map((doc) => (
            <DocumentRow
              key={doc.id}
              document={doc}
              onDelete={() => handleDelete(doc)}
              onInspect={onLoadChunks}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- row ------

function DocumentRow({
  document: doc,
  onDelete,
  onInspect,
}: {
  document: DocumentRecord;
  onDelete: () => void;
  onInspect: (id: string, limit?: number) => Promise<RetrievedChunk[]>;
}) {
  const [chunksOpen, setChunksOpen] = useState(false);
  const [chunks, setChunks] = useState<RetrievedChunk[] | null>(null);
  const [chunksLoading, setChunksLoading] = useState(false);

  const meta = STATUS_META[doc.status];
  const StatusIcon = meta.icon;

  const openChunks = useCallback(async () => {
    setChunksOpen(true);
    setChunksLoading(true);
    try {
      setChunks(await onInspect(doc.id, 12));
    } catch (cause) {
      toast.error("Could not load chunks", {
        description: cause instanceof ApiError ? cause.message : "Unexpected error.",
      });
      setChunks([]);
    } finally {
      setChunksLoading(false);
    }
  }, [doc.id, onInspect]);

  return (
    <li className="group glass glass-hover animate-fade-in rounded-xl p-2.5">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-border/70 bg-background/60 text-[10px] font-bold uppercase text-primary">
          {doc.extension.replace(".", "") || "?"}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <p className="truncate text-[13px] font-medium" title={doc.filename}>
              {doc.filename}
            </p>
          </div>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground">
            <span>{formatBytes(doc.size_bytes)}</span>
            <span className="text-border">•</span>
            <span>{doc.word_count.toLocaleString()} words</span>
            {doc.page_count ? (
              <>
                <span className="text-border">•</span>
                <span>{doc.page_count} pages</span>
              </>
            ) : null}
          </p>

          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <Badge variant={meta.variant}>
              <StatusDot status={meta.dot} className="h-1.5 w-1.5" />
              <StatusIcon className={cn("h-3 w-3", doc.status === "processing" && "animate-spin")} />
              {meta.label}
            </Badge>
            <Badge variant="mono">
              <ScanText className="h-3 w-3" />
              {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
            </Badge>
            {doc.processing_ms ? (
              <Badge variant="outline" className="font-mono">
                {formatDuration(doc.processing_ms)}
              </Badge>
            ) : null}
          </div>

          {doc.status === "error" && doc.error ? (
            <p className="mt-1.5 rounded-md border border-destructive/30 bg-destructive/5 px-2 py-1 text-[11px] text-destructive">
              {doc.error}
            </p>
          ) : null}
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              className="shrink-0 opacity-60 transition-opacity group-hover:opacity-100"
              aria-label={`Actions for ${doc.filename}`}
            >
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-52">
            <DropdownMenuItem onSelect={() => void openChunks()}>
              <ScanText className="h-3.5 w-3.5" />
              Inspect chunks
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={onDelete}
              className="text-destructive focus:text-destructive"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete document
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <p className="mt-1.5 text-[10.5px] text-muted-foreground/70">
        {doc.embedding_model} · uploaded {relativeTime(doc.uploaded_at)} · {formatDateTime(doc.uploaded_at)}
      </p>

      <Dialog open={chunksOpen} onOpenChange={setChunksOpen}>
        <DialogContent className="max-h-[80vh] max-w-2xl overflow-hidden">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <ScanText className="h-4 w-4 text-primary" />
              {doc.filename}
            </DialogTitle>
            <DialogDescription>
              {doc.chunk_count} chunks · {formatBytes(doc.size_bytes)} · {doc.embedding_model}
            </DialogDescription>
          </DialogHeader>
          <div className="scrollbar-thin max-h-[58vh] space-y-2 overflow-y-auto pr-1">
            {chunksLoading
              ? [0, 1, 2].map((index) => <Skeleton key={index} className="h-20 w-full rounded-lg" />)
              : chunks?.map((chunk) => (
                  <div key={chunk.id} className="rounded-lg border border-border/60 bg-background/40 p-2.5">
                    <div className="mb-1 flex items-center gap-1.5">
                      <Badge variant="mono">#{chunk.chunk_index}</Badge>
                      {chunk.page ? <Badge variant="outline">page {chunk.page}</Badge> : null}
                      <span className="font-mono text-[10px] text-muted-foreground">
                        {chunk.char_start ?? 0}–{chunk.char_end ?? 0} · ~{chunk.token_estimate} tokens
                      </span>
                    </div>
                    <p className="whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-foreground/80">
                      {chunk.text}
                    </p>
                  </div>
                ))}
          </div>
        </DialogContent>
      </Dialog>
    </li>
  );
}
