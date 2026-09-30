import { BookMarked, ExternalLink } from "lucide-react";

import { EmptyState } from "@/components/common/EmptyState";
import { SectionHeader } from "@/components/common/SectionHeader";
import { Badge } from "@/components/ui/badge";
import { scoreColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Source } from "@/types/api";

interface SourceCitationsProps {
  sources: Source[];
  onHoverSource?: (source: Source | null) => void;
}

/** Numbered citation list shown next to (and inside) the generated answer. */
export function SourceCitations({ sources, onHoverSource }: SourceCitationsProps) {
  return (
    <div className="space-y-2.5">
      <SectionHeader
        icon={<BookMarked className="h-3.5 w-3.5" />}
        title="Source citations"
        hint={
          sources.length
            ? `${sources.length} unique chunk${sources.length > 1 ? "s" : ""} cited`
            : "Cited for the latest answer"
        }
      />

      {!sources.length ? (
        <EmptyState
          icon={<BookMarked className="h-4 w-4" />}
          title="No citations"
          description="When retrieval returns relevant chunks, each one is listed here with its file, chunk number, page and similarity score."
        />
      ) : (
        <ol className="space-y-1.5">
          {sources.map((source, index) => (
            <li
              key={`${source.document_id}-${source.chunk_index}`}
              onMouseEnter={() => onHoverSource?.(source)}
              onMouseLeave={() => onHoverSource?.(null)}
              className="group rounded-xl border border-border/60 bg-background/40 p-2.5 transition-colors hover:border-primary/35 hover:bg-background/60"
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5 grid h-4.5 w-4.5 shrink-0 place-items-center rounded bg-primary/15 text-[9px] font-bold text-primary">
                  {index + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-1.5 text-[12px] font-medium">
                    <span className="truncate">{source.filename}</span>
                    <ExternalLink className="h-3 w-3 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-70" />
                  </p>
                  <p className="mt-0.5 flex flex-wrap items-center gap-1.5">
                    <Badge variant="mono" className="px-1.5 py-0 text-[9.5px]">
                      chunk #{source.chunk_index}
                    </Badge>
                    {source.page ? (
                      <Badge variant="outline" className="px-1.5 py-0 text-[9.5px]">
                        page {source.page}
                      </Badge>
                    ) : null}
                    <span className={cn("font-mono text-[10px] tabular-nums", scoreColor(source.score))}>
                      {source.score.toFixed(4)}
                    </span>
                  </p>
                  <p className="mt-1.5 line-clamp-3 text-[11px] leading-snug text-muted-foreground">
                    “{source.snippet}”
                  </p>
                </div>
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
