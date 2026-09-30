import {
  BookOpen,
  CheckCircle2,
  Database,
  FileInput,
  Loader2,
  Scissors,
  Search,
  Sparkles,
  SplitSquareHorizontal,
  TriangleAlert,
  Wand2,
  XCircle,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatDuration } from "@/lib/format";
import { cn } from "@/lib/utils";
import type {
  PipelineStage,
  PipelineStageDefinition,
  PipelineTrace,
  StageStatus,
} from "@/types/api";

const STAGE_ICONS: Record<string, LucideIcon> = {
  ingest: FileInput,
  extraction: BookOpen,
  cleaning: Wand2,
  chunking: SplitSquareHorizontal,
  embedding: Scissors,
  vector_store: Database,
  retrieval: Search,
  generation: Sparkles,
};

const STATUS_META: Record<
  StageStatus,
  { icon: LucideIcon; className: string; label: string }
> = {
  pending: { icon: Loader2, className: "text-muted-foreground/50", label: "Idle" },
  running: { icon: Loader2, className: "text-primary", label: "Running" },
  success: { icon: CheckCircle2, className: "text-emerald-400", label: "Done" },
  error: { icon: XCircle, className: "text-destructive", label: "Failed" },
  cached: { icon: CheckCircle2, className: "text-cyan-400", label: "Cached" },
  skipped: { icon: Loader2, className: "text-muted-foreground/40", label: "Skipped" },
};

interface PipelineFlowProps {
  definitions: PipelineStageDefinition[];
  trace: PipelineTrace | null;
  loading?: boolean;
}

/** The 8-stage RAG pipeline: ingest → … → generation. */
export function PipelineFlow({ definitions, trace, loading = false }: PipelineFlowProps) {
  if (loading && definitions.length === 0) {
    return (
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        {Array.from({ length: 8 }).map((_, index) => (
          <Skeleton key={index} className="h-[72px] w-full rounded-xl" />
        ))}
      </div>
    );
  }

  const byKey = new Map<string, PipelineStage>((trace?.stages ?? []).map((s) => [s.key, s]));
  const operationLabel =
    trace === null
      ? "No pipeline run yet"
      : `${trace.operation} · ${trace.status} · ${formatDuration(trace.total_duration_ms)}`;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-1.5">
        <p className="text-[10.5px] font-medium uppercase tracking-wide text-muted-foreground">
          Pipeline flow
        </p>
        {trace ? (
          <Badge
            variant={trace.status === "success" ? "success" : trace.status === "error" ? "destructive" : "warning"}
            className="font-mono"
          >
            {operationLabel}
          </Badge>
        ) : null}
      </div>

      <ol className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        {definitions.map((definition, index) => {
          const Icon = STAGE_ICONS[definition.key] ?? Sparkles;
          const stage = byKey.get(definition.key);
          const status: StageStatus = stage?.status ?? "pending";
          const meta = STATUS_META[status];
          const StatusIcon = meta.icon;
          const isRunning = status === "running";

          const node = (
            <li
              className={cn(
                "glass glass-hover relative animate-fade-in rounded-xl p-2 transition-all",
                isRunning && "border-primary/50 ring-glow",
                status === "error" && "border-destructive/40",
              )}
              style={{ animationDelay: `${index * 40}ms` }}
            >
              <div className="flex items-center gap-1.5">
                <span className="grid h-5 w-5 place-items-center rounded-md bg-primary/12 text-primary">
                  <Icon className="h-3 w-3" />
                </span>
                <span className="ml-auto">
                  <StatusIcon
                    className={cn("h-3 w-3", meta.className, isRunning && "animate-spin")}
                  />
                </span>
              </div>
              <p className="mt-1.5 text-[11.5px] font-medium leading-tight">{definition.label}</p>
              <p className="mt-0.5 font-mono text-[9.5px] text-muted-foreground/80">
                {stage?.duration_ms != null ? formatDuration(stage.duration_ms) : meta.label}
              </p>
              {index < definitions.length - 1 ? (
                <span className="pointer-events-none absolute -right-[5px] top-1/2 hidden h-px w-1.5 bg-border sm:block" />
              ) : null}
            </li>
          );

          const detail = stage?.detail ?? definition.description;
          return (
            <Tooltip key={definition.key}>
              <TooltipTrigger asChild>{node}</TooltipTrigger>
              <TooltipContent side="top" className="max-w-[26ch] whitespace-normal">
                <p className="font-medium">{definition.label}</p>
                <p className="text-muted-foreground">{detail}</p>
                {stage && Object.keys(stage.metrics).length ? (
                  <p className="font-mono text-[10.5px] text-muted-foreground">
                    {Object.entries(stage.metrics)
                      .map(([key, value]) => `${key}=${String(value)}`)
                      .join(" · ")}
                  </p>
                ) : null}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </ol>

      {trace?.status === "error" ? (
        <p className="flex items-start gap-1.5 rounded-lg border border-destructive/30 bg-destructive/5 px-2.5 py-2 text-[11px] text-destructive">
          <TriangleAlert className="mt-0.5 h-3 w-3 shrink-0" />
          {trace.stages.find((stage) => stage.status === "error")?.detail ?? "A pipeline stage failed."}
        </p>
      ) : null}
    </div>
  );
}
