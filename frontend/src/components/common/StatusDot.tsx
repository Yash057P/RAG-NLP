import { cn } from "@/lib/utils";

type DotStatus = "ok" | "busy" | "warn" | "idle";

const DOT_STYLES: Record<DotStatus, string> = {
  ok: "bg-emerald-400 shadow-[0_0_8px_1px_rgba(52,211,153,0.55)]",
  busy: "bg-primary animate-blink",
  warn: "bg-amber-400",
  idle: "bg-muted-foreground/50",
};

export function StatusDot({ status, className }: { status: DotStatus; className?: string }) {
  return <span className={cn("inline-block h-2 w-2 shrink-0 rounded-full", DOT_STYLES[status], className)} />;
}
