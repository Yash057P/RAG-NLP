import { Layers, Search, Timer } from "lucide-react";

import { EmptyState } from "@/components/common/EmptyState";
import { SectionHeader } from "@/components/common/SectionHeader";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { formatDuration, formatNumber, scoreBarColor, scoreColor, splitHighlight } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AnswerResponse, RetrievedChunk } from "@/types/api";

interface RetrievedChunksProps {
  answer: AnswerResponse | null;
  query: string;
}

/** Expandable list of the top-K chunks with their cosine similarity scores. */
export function RetrievedChunks({ answer, query }: RetrievedChunksProps) {
  const chunks = answer?.retrieved_chunks ?? [];
  const retrieval = answer?.retrieval;

  return (
    <div className="space-y-2.5">
      <SectionHeader
        icon={<Layers className="h-3.5 w-3.5" />}
        title="Retrieved chunks"
        hint={
          retrieval
            ? `Top ${retrieval.top_k} of ${formatNumber(retrieval.total_candidates)} candidates · ${formatDuration(retrieval.total_ms)}`
            : "Shown after you ask a question"
        }
      />

      {!retrieval || !chunks.length ? (
        <EmptyState
          icon={<Search className="h-4 w-4" />}
          title="No retrieval yet"
          description="The chunks returned by the vector search for the latest question appear here, ranked by cosine similarity."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-1.5 text-[10.5px]">
            <MiniStat label="Embed" value={formatDuration(retrieval.embedding_ms)} />
            <MiniStat label="Search" value={formatDuration(retrieval.search_ms)} />
          </div>

          <Accordion
            type="multiple"
            defaultValue={[chunks[0]?.id ?? ""]}
            className="space-y-1.5"
          >
            {chunks.map((chunk, index) => (
              <ChunkItem key={chunk.id} chunk={chunk} rank={index + 1} query={query} />
            ))}
          </Accordion>

          <p className="rounded-lg border border-border/50 bg-background/30 px-2.5 py-2 text-[10.5px] leading-relaxed text-muted-foreground">
            Vector search: <span className="font-mono">{retrieval.vector_store_backend}</span> ·{" "}
            <span className="font-mono">{retrieval.embedding_model}</span>
          </p>
        </>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border/50 bg-background/30 px-2 py-1.5">
      <p className="text-[9.5px] uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="font-mono text-[11.5px] tabular-nums text-foreground/85">{value}</p>
    </div>
  );
}

function ChunkItem({ chunk, rank, query }: { chunk: RetrievedChunk; rank: number; query: string }) {
  const percent = Math.round(Math.max(0, Math.min(1, chunk.score)) * 100);

  return (
    <AccordionItem value={chunk.id}>
      <AccordionTrigger className="gap-2.5">
        <span className="grid h-5 w-5 shrink-0 place-items-center rounded-md bg-muted text-[10px] font-bold text-muted-foreground">
          {rank}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <span className="truncate text-[12px] font-medium">{chunk.filename}</span>
            <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
              #{chunk.chunk_index}
            </span>
            {chunk.page ? (
              <Badge variant="outline" className="shrink-0 px-1.5 py-0 text-[9.5px]">
                p.{chunk.page}
              </Badge>
            ) : null}
            {chunk.is_reference ? (
              <Badge
                variant="outline"
                title="Citation list - ranked below prose because a paper's own title repeats the topic terms, but it answers nothing."
                className="shrink-0 px-1.5 py-0 text-[9.5px] text-amber-600 dark:text-amber-400"
              >
                bibliography
              </Badge>
            ) : null}
          </span>
          <span className="mt-1 flex items-center gap-2">
            <span className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
              <span
                className={cn("block h-full rounded-full transition-all duration-500", scoreBarColor(chunk.score))}
                style={{ width: `${Math.max(percent, 3)}%` }}
              />
            </span>
            <span className={cn("shrink-0 font-mono text-[10.5px] tabular-nums", scoreColor(chunk.score))}>
              {chunk.score.toFixed(4)}
            </span>
          </span>
        </span>
      </AccordionTrigger>

      <AccordionContent>
        <HighlightedText text={chunk.text} query={query} />
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <Badge variant="mono" className="gap-1">
            <Timer className="h-3 w-3" />~{chunk.token_estimate} tokens
          </Badge>
          {chunk.char_start !== null && chunk.char_end !== null ? (
            <Badge variant="outline" className="font-mono">
              chars {chunk.char_start}–{chunk.char_end}
            </Badge>
          ) : null}
          {chunk.distance !== null ? (
            <Badge variant="outline" className="font-mono">
              d={chunk.distance.toFixed(4)}
            </Badge>
          ) : null}
        </div>
      </AccordionContent>
    </AccordionItem>
  );
}

/** Renders the chunk with the query terms highlighted. */
export function HighlightedText({ text, query }: { text: string; query: string }) {
  return (
    <p className="whitespace-pre-wrap rounded-lg border border-border/50 bg-background/50 p-2 text-[11.5px] leading-relaxed text-foreground/80">
      {splitHighlight(text, query).map((part, index) =>
        part.hit ? (
          <mark
            key={index}
            className="rounded bg-primary/25 px-0.5 text-foreground [box-shadow:0_0_0_1px_hsl(var(--primary)/0.35)]"
          >
            {part.text}
          </mark>
        ) : (
          <span key={index}>{part.text}</span>
        ),
      )}
    </p>
  );
}
