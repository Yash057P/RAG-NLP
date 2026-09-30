import type { ReactNode } from "react";
import {
  Activity,
  BrainCircuit,
  Clock,
  Database,
  FileStack,
  Gauge,
  MessageSquare,
  Ruler,
  Timer,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatBytes, formatDateTime, formatDuration, formatNumber, relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ComponentHealth, Stats } from "@/types/api";

interface StatsCardsProps {
  stats: Stats | null;
  vectorStoreHealth?: ComponentHealth | null;
  embeddingDetail?: string | null;
  loading?: boolean;
}

/** The six analytics cards required by the spec plus configuration extras. */
export function StatsCards({ stats, vectorStoreHealth, embeddingDetail, loading }: StatsCardsProps) {
  if (loading && !stats) {
    return (
      <div className="grid grid-cols-2 gap-1.5">
        {[0, 1, 2, 3, 4, 5].map((index) => (
          <Skeleton key={index} className="h-[68px] w-full rounded-xl" />
        ))}
      </div>
    );
  }

  if (!stats) {
    return (
      <p className="rounded-lg border border-destructive/30 bg-destructive/5 px-2.5 py-2 text-[11px] text-destructive">
        Analytics unavailable — the backend did not answer /api/stats.
      </p>
    );
  }

  const storeReady = vectorStoreHealth?.status === "ok" || vectorStoreHealth?.status === "degraded";

  return (
    <div className="grid grid-cols-2 gap-1.5">
      <StatCard
        icon={<FileStack className="h-3.5 w-3.5" />}
        label="Documents"
        value={formatNumber(stats.total_documents)}
        hint={`${stats.documents_ready} ready · ${formatBytes(stats.total_size_bytes)}`}
        tone="primary"
      />
      <StatCard
        icon={<Database className="h-3.5 w-3.5" />}
        label="Chunks"
        value={formatNumber(stats.total_chunks)}
        hint={`${stats.embedding_dimension}-dim vectors`}
        tone="accent"
      />
      <StatCard
        icon={<BrainCircuit className="h-3.5 w-3.5" />}
        label="Embedding model"
        value={shortModel(stats.embedding_model)}
        hint={stats.embedding_provider}
        tooltip={embeddingDetail ?? stats.embedding_model}
        mono
        tone="primary"
      />
      <StatCard
        icon={<Activity className="h-3.5 w-3.5" />}
        label="Vector store"
        value={storeReady ? "Connected" : "Unavailable"}
        hint={`${stats.vector_store_backend} · ${stats.vector_store_status}`}
        tooltip={vectorStoreHealth?.detail}
        tone={storeReady ? "success" : "destructive"}
      />
      <StatCard
        icon={<MessageSquare className="h-3.5 w-3.5" />}
        label="Questions asked"
        value={formatNumber(stats.questions_asked)}
        hint={stats.last_question_at ? `last ${relativeTime(stats.last_question_at)}` : "no questions yet"}
        tone="accent"
      />
      <StatCard
        icon={<Timer className="h-3.5 w-3.5" />}
        label="Avg response"
        value={
          stats.average_response_time_ms ? formatDuration(stats.average_response_time_ms) : "-"
        }
        hint={
          stats.slowest_response_ms
            ? `max ${formatDuration(stats.slowest_response_ms)}`
            : "no samples"
        }
        tone="primary"
      />

      {/* configuration strip ------------------------------------------- */}
      <div className="col-span-2 flex flex-wrap items-center gap-1.5 rounded-xl border border-border/50 bg-background/30 px-2.5 py-2 text-[10.5px] text-muted-foreground">
        <Badge variant="mono" className="gap-1">
          <Ruler className="h-3 w-3" />
          {stats.chunk_size}/{stats.chunk_overlap} chars
        </Badge>
        <Badge variant="mono" className="gap-1">
          <Gauge className="h-3 w-3" />top-{stats.top_k}
        </Badge>
        <Badge variant="mono" className="gap-1">
          <BrainCircuit className="h-3 w-3" />
          {stats.llm_provider}
        </Badge>
        <Badge variant="mono" className="gap-1">
          <Clock className="h-3 w-3" />
          {stats.llm_model}
        </Badge>
        {stats.last_indexed_at ? (
          <span className="ml-auto">indexed {formatDateTime(stats.last_indexed_at)}</span>
        ) : null}
      </div>
    </div>
  );
}

type Tone = "primary" | "accent" | "success" | "destructive";

const TONE_STYLES: Record<Tone, string> = {
  primary: "text-primary bg-primary/12",
  accent: "text-accent bg-accent/12",
  success: "text-emerald-400 bg-emerald-500/12",
  destructive: "text-destructive bg-destructive/12",
};

function StatCard({
  icon,
  label,
  value,
  hint,
  tooltip,
  tone = "primary",
  mono = false,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  hint?: string;
  tooltip?: string;
  tone?: Tone;
  mono?: boolean;
}) {
  const card = (
    <div className="glass glass-hover min-w-0 rounded-xl p-2">
      <div className="flex items-center gap-1.5">
        <span className={cn("grid h-5 w-5 place-items-center rounded-md", TONE_STYLES[tone])}>
          {icon}
        </span>
        <span className="truncate text-[10.5px] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
      </div>
      <p
        className={cn(
          "mt-1.5 truncate text-[15px] font-semibold leading-none tracking-tight",
          mono && "font-mono text-[12px]",
        )}
        title={value}
      >
        {value}
      </p>
      {hint ? <p className="mt-1 truncate text-[10px] text-muted-foreground/80">{hint}</p> : null}
    </div>
  );

  if (!tooltip) return card;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{card}</TooltipTrigger>
      <TooltipContent side="bottom">{tooltip}</TooltipContent>
    </Tooltip>
  );
}

function shortModel(model: string): string {
  const trimmed = model.includes("/") ? (model.split("/").pop() ?? model) : model;
  return trimmed.length > 22 ? `${trimmed.slice(0, 21)}…` : trimmed;
}
