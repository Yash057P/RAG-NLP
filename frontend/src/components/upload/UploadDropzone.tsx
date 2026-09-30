import { useCallback, useMemo, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { AlertTriangle, FileUp, Loader2, RotateCcw, UploadCloud, X } from "lucide-react";
import { toast } from "sonner";

import { StatusDot } from "@/components/common/StatusDot";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { SystemStatus, UploadResponse } from "@/types/api";

import type { UploadState } from "@/hooks/useDocuments";

interface UploadDropzoneProps {
  upload: UploadState;
  onUpload: (files: File[]) => Promise<UploadResponse | null>;
  onUploaded?: (response: UploadResponse) => void;
  systemStatus: SystemStatus | null;
  onReset?: () => void;
  resetPending?: boolean;
}

/** Multi-file drag & drop zone with client-side validation and XHR progress. */
export function UploadDropzone({
  upload,
  onUpload,
  onUploaded,
  systemStatus,
  onReset,
  resetPending = false,
}: UploadDropzoneProps) {
  const [rejected, setRejected] = useState<string[]>([]);

  const allowed = useMemo(
    () => systemStatus?.config.upload.allowed_extensions ?? [".pdf", ".docx", ".txt"],
    [systemStatus],
  );
  const maxSizeMb = systemStatus?.config.upload.max_size_mb ?? 25;
  const extensions = useMemo(() => allowed.map((ext) => ext.replace(".", "")), [allowed]);

  const handleFiles = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      setRejected([]);
      try {
        const response = await onUpload(files);
        if (response) {
          const created = response.created.length;
          const failed = response.failed.length;
          if (created) {
            const chunks = response.created.reduce((sum, doc) => sum + doc.chunk_count, 0);
            toast.success(
              `Indexed ${created} document${created > 1 ? "s" : ""}`,
              { description: `${chunks} chunk${chunks === 1 ? "" : "s"} embedded into the vector store.` },
            );
          }
          if (failed) {
            toast.error(`${failed} file${failed > 1 ? "s" : ""} rejected`, {
              description: response.failed.map((item) => `${item.filename}: ${item.reason}`).join(" · "),
            });
          }
          onUploaded?.(response);
        }
      } catch (cause) {
        const message =
          cause instanceof ApiError ? cause.message : "Unexpected error while uploading files.";
        toast.error("Upload failed", { description: message });
      }
    },
    [onUpload, onUploaded],
  );

  const onDrop = useCallback(
    (accepted: File[], fileRejections: FileRejection[]) => {
      const messages = fileRejections.map(
        (rejection) => `${rejection.file.name}: ${rejection.errors[0]?.message ?? "rejected"}`,
      );
      setRejected(messages);
      if (messages.length) {
        toast.warning("Some files were skipped", { description: messages.join(" · ") });
      }
      void handleFiles(accepted);
    },
    [handleFiles],
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    noClick: true,
    noKeyboard: true,
    multiple: true,
    accept: Object.fromEntries(extensions.map((ext) => [`.${ext}`, []])),
    maxSize: maxSizeMb * 1024 * 1024,
    disabled: upload.uploading,
  });

  return (
    <div className="space-y-2.5">
      <div
        {...getRootProps()}
        className={cn(
          "group relative overflow-hidden rounded-2xl border border-dashed p-4 text-center transition-all duration-200",
          isDragActive
            ? "border-primary bg-primary/10 ring-glow"
            : "border-border/80 bg-background/40 hover:border-primary/50 hover:bg-background/60",
          upload.uploading && "pointer-events-none opacity-70",
        )}
      >
        {/* getInputProps() owns the input ref (react-dropzone needs it for
            open()) and already hides the element visually. */}
        <input {...getInputProps()} />

        <div className="pointer-events-none absolute inset-0 bg-grid-slate opacity-[0.35]" />

        <div className="relative flex flex-col items-center gap-2">
          <span
            className={cn(
              "grid h-11 w-11 place-items-center rounded-2xl border border-border/70 bg-background/70 text-primary transition-transform duration-200",
              isDragActive && "scale-110 animate-pulse-ring",
            )}
          >
            {upload.uploading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <UploadCloud className="h-5 w-5" />
            )}
          </span>

          {upload.uploading ? (
            <div className="w-full space-y-1.5">
              <p className="text-xs font-medium">
                Uploading {upload.filenames.length} file{upload.filenames.length > 1 ? "s" : ""}…
              </p>
              <Progress value={upload.progress} className="h-1" />
              <p className="font-mono text-[10.5px] text-muted-foreground">
                {upload.progress}% · extracting, cleaning, chunking & embedding
              </p>
            </div>
          ) : (
            <>
              <p className="text-xs font-medium">
                {isDragActive ? "Drop the files to start indexing" : "Drag & drop documents here"}
              </p>
              <p className="text-[11px] text-muted-foreground">
                PDF, DOCX or TXT · up to {maxSizeMb} MB each
              </p>
              <div className="mt-0.5 flex items-center gap-1.5">
                <Button type="button" size="xs" variant="gradient" onClick={open}>
                  <FileUp className="h-3.5 w-3.5" />
                  Browse files
                </Button>
                {rejected.length > 0 ? (
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    onClick={() => setRejected([])}
                    className="text-muted-foreground"
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                ) : null}
              </div>
            </>
          )}
        </div>
      </div>

      {rejected.length ? (
        <ul className="space-y-1 rounded-lg border border-destructive/30 bg-destructive/5 p-2 text-[11px] text-destructive">
          {rejected.map((message) => (
            <li key={message} className="flex items-start gap-1.5">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              <span>{message}</span>
            </li>
          ))}
        </ul>
      ) : null}

      <div className="flex items-center justify-between gap-2 rounded-lg border border-border/60 bg-background/30 px-2.5 py-1.5">
        <span className="flex min-w-0 items-center gap-1.5 text-[11px] text-muted-foreground">
          <StatusDot status={systemStatus?.status === "ok" ? "ok" : "warn"} />
          <span className="truncate">
            {systemStatus
              ? `${systemStatus.embeddings.name.split("(")[0]?.trim()} · ${systemStatus.vector_store.name.split("(")[0]?.trim()}`
              : "Connecting to backend…"}
          </span>
        </span>
        {onReset ? (
          <Button
            type="button"
            size="xs"
            variant="ghost"
            onClick={onReset}
            disabled={resetPending}
            className="shrink-0 text-muted-foreground hover:text-destructive"
          >
            {resetPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
            Reset
          </Button>
        ) : null}
      </div>
    </div>
  );
}
