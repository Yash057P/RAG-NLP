import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface ConfidenceMeterProps {
  confidence: number;
  label: string;
  breakdown?: Record<string, number>;
  grounded?: boolean;
  className?: string;
  compact?: boolean;
}

const WEIGHTS: Record<string, string> = {
  similarity: "45% top similarity",
  coverage: "35% term coverage",
  margin: "20% score margin",
};

function barClass(confidence: number, grounded = true): string {
  if (!grounded) return "bg-gradient-to-r from-amber-500 to-orange-500";
  if (confidence >= 0.7) return "bg-gradient-to-r from-emerald-400 to-teal-400";
  if (confidence >= 0.45) return "bg-gradient-to-r from-sky-400 to-cyan-400";
  if (confidence >= 0.25) return "bg-gradient-to-r from-amber-400 to-orange-400";
  return "bg-gradient-to-r from-rose-500 to-red-500";
}

function textClass(confidence: number, grounded = true): string {
  if (!grounded) return "text-amber-400";
  if (confidence >= 0.7) return "text-emerald-400";
  if (confidence >= 0.45) return "text-sky-400";
  if (confidence >= 0.25) return "text-amber-400";
  return "text-rose-400";
}

/**
 * Confidence = 0.45·similarity + 0.35·coverage + 0.20·margin (see
 * backend/services/rag_service.py). Hovering the meter exposes the split.
 */
export function ConfidenceMeter({
  confidence,
  label,
  breakdown,
  grounded = true,
  className,
  compact = false,
}: ConfidenceMeterProps) {
  const percent = Math.round(Math.max(0, Math.min(1, confidence)) * 100);

  const meter = (
    <div className={cn("w-full", className)}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className={cn("font-semibold tabular-nums", compact ? "text-sm" : "text-base", textClass(confidence, grounded))}>
          {percent}%
        </span>
        <span className={cn("truncate text-muted-foreground", compact ? "text-[10.5px]" : "text-[11px]")}>
          {label}
        </span>
      </div>
      <Progress
        value={percent}
        className={cn(compact ? "h-1" : "h-1.5")}
        indicatorClassName={barClass(confidence, grounded)}
        aria-label={`Confidence ${percent}% - ${label}`}
      />
    </div>
  );

  if (!breakdown || !Object.keys(breakdown).length) return meter;

  return (
    <Tooltip>
      <TooltipTrigger asChild>{meter}</TooltipTrigger>
      <TooltipContent side="top" className="space-y-1">
        <p className="font-medium">Confidence breakdown</p>
        {Object.entries(breakdown).map(([key, value]) => (
          <p key={key} className="flex items-center justify-between gap-4 text-muted-foreground">
            <span>{WEIGHTS[key] ?? key}</span>
            <span className="font-mono tabular-nums text-foreground">
              {Math.round(value * 100)}%
            </span>
          </p>
        ))}
      </TooltipContent>
    </Tooltip>
  );
}
