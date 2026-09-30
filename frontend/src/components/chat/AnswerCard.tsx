import { useState } from "react";
import {
  BadgeCheck,
  Clock,
  Copy,
  FileDown,
  Quote,
  ScrollText,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import { toast } from "sonner";

import { ConfidenceMeter } from "@/components/common/ConfidenceMeter";
import { MarkdownAnswer } from "@/components/common/MarkdownAnswer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { downloadAnswerAsPdf } from "@/lib/export";
import { formatDuration, formatTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { AnswerResponse, Source } from "@/types/api";

import type { AssistantMessage } from "./types";

interface AnswerCardProps {
  message: AssistantMessage;
  active?: boolean;
  onSelect?: (answer: AnswerResponse) => void;
}

export function AnswerCard({ message, active = false, onSelect }: AnswerCardProps) {
  const { answer } = message;
  const [promptOpen, setPromptOpen] = useState(false);
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    setExporting(true);
    try {
      await downloadAnswerAsPdf(answer);
      toast.success("PDF downloaded", { description: "Answer, metrics and sources exported." });
    } catch {
      toast.error("Could not generate the PDF");
    } finally {
      setExporting(false);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(answer.answer);
      toast.success("Answer copied to clipboard");
    } catch {
      toast.error("Clipboard access denied by the browser");
    }
  };

  return (
    <article
      className={cn(
        "group animate-fade-in rounded-2xl border bg-background/50 p-3.5 transition-colors",
        active ? "border-primary/40 bg-background/70" : "border-border/70",
      )}
      onClick={() => onSelect?.(answer)}
    >
      {/* header ---------------------------------------------------------- */}
      <div className="mb-2.5 flex flex-wrap items-center gap-1.5">
        <Badge variant={answer.abstained ? "warning" : answer.grounded ? "success" : "destructive"}>
          {answer.abstained ? (
            <TriangleAlert className="h-3 w-3" />
          ) : answer.grounded ? (
            <BadgeCheck className="h-3 w-3" />
          ) : (
            <TriangleAlert className="h-3 w-3" />
          )}
          {answer.abstained ? "Not in the documents" : answer.grounded ? "Grounded" : "Ungrounded"}
        </Badge>
        <Badge variant="mono" className="gap-1">
          <Sparkles className="h-3 w-3" />
          {answer.llm.model}
        </Badge>
        <Badge variant="outline" className="gap-1 font-mono">
          <Clock className="h-3 w-3" />
          {formatDuration(answer.response_time_ms)}
        </Badge>
        <span className="ml-auto font-mono text-[10px] text-muted-foreground/70">
          {formatTime(message.created_at)}
        </span>
      </div>

      {/* answer ---------------------------------------------------------- */}
      <MarkdownAnswer content={answer.answer} />

      {/* confidence ------------------------------------------------------ */}
      <div className="mt-3 rounded-xl border border-border/60 bg-background/40 p-2.5">
        <ConfidenceMeter
          confidence={answer.confidence}
          label={answer.confidence_label}
          breakdown={answer.confidence_breakdown}
          grounded={answer.grounded}
        />
        <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10.5px] text-muted-foreground sm:grid-cols-4">
          <Meta label="Top similarity" value={answer.retrieval.top_score.toFixed(4)} />
          <Meta label="Avg similarity" value={answer.retrieval.average_score.toFixed(4)} />
          <Meta label="Chunks" value={String(answer.retrieved_chunks.length)} />
          <Meta label="Coverage" value={`${Math.round(answer.retrieval.grounding_coverage * 100)}%`} />
        </div>
      </div>

      {/* sources --------------------------------------------------------- */}
      {answer.sources.length ? (
        <div className="mt-2.5">
          <p className="mb-1.5 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            <Quote className="h-3 w-3" />
            Sources ({answer.sources.length})
          </p>
          <ul className="space-y-1">
            {answer.sources.map((source, index) => (
              <li key={`${source.document_id}-${source.chunk_index}`}>
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelect?.(answer);
                  }}
                  className="flex w-full items-start gap-2 rounded-lg border border-border/50 bg-background/30 px-2 py-1.5 text-left transition-colors hover:border-primary/40 hover:bg-accent/5"
                >
                  <span className="mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded bg-primary/15 text-[9px] font-bold text-primary">
                    {index + 1}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex flex-wrap items-center gap-x-1.5 text-[11.5px] font-medium">
                      <span className="truncate">{source.filename}</span>
                      <span className="font-mono text-[10px] text-muted-foreground">
                        chunk #{source.chunk_index}
                        {source.page ? ` · p.${source.page}` : ""}
                      </span>
                      <span className="ml-auto font-mono text-[10px] text-muted-foreground">
                        {source.score.toFixed(3)}
                      </span>
                    </span>
                    <span className="mt-0.5 line-clamp-2 block text-[11px] leading-snug text-muted-foreground">
                      “{source.snippet}”
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {/* actions --------------------------------------------------------- */}
      <div className="mt-2.5 flex flex-wrap items-center gap-1.5 opacity-70 transition-opacity group-hover:opacity-100">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button type="button" size="xs" variant="ghost" onClick={handleExport} disabled={exporting}>
              <FileDown className="h-3.5 w-3.5" />
              {exporting ? "Preparing…" : "Download PDF"}
            </Button>
          </TooltipTrigger>
          <TooltipContent>Export answer, metrics and sources as a PDF</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button type="button" size="xs" variant="ghost" onClick={handleCopy}>
              <Copy className="h-3.5 w-3.5" />
              Copy
            </Button>
          </TooltipTrigger>
          <TooltipContent>Copy the raw answer text</TooltipContent>
        </Tooltip>
        <Button type="button" size="xs" variant="ghost" onClick={() => setPromptOpen(true)}>
          <ScrollText className="h-3.5 w-3.5" />
          View prompt
        </Button>
      </div>

      <PromptDialog
        open={promptOpen}
        onOpenChange={setPromptOpen}
        answer={answer}
        sources={answer.sources}
      />
    </article>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-center justify-between gap-1">
      <span className="truncate">{label}</span>
      <span className="font-mono tabular-nums text-foreground/80">{value}</span>
    </span>
  );
}

function PromptDialog({
  open,
  onOpenChange,
  answer,
  sources,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  answer: AnswerResponse;
  sources: Source[];
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[80vh] max-w-2xl overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ScrollText className="h-4 w-4 text-primary" />
            Prompt sent to {answer.llm.model}
          </DialogTitle>
          <DialogDescription>
            The exact string the model received — system instructions plus the {sources.length}{" "}
            retrieved chunk(s) as context.
          </DialogDescription>
        </DialogHeader>
        <pre className="scrollbar-thin max-h-[58vh] overflow-auto whitespace-pre-wrap rounded-lg border border-border/60 bg-background/60 p-3 font-mono text-[11.5px] leading-relaxed text-foreground/85">
          {answer.prompt}
        </pre>
      </DialogContent>
    </Dialog>
  );
}
