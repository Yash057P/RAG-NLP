import { GraduationCap, Moon, Sun } from "lucide-react";

import { StatusDot } from "@/components/common/StatusDot";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Stats, SystemStatus } from "@/types/api";

import type { Theme } from "@/hooks/useTheme";

interface AppHeaderProps {
  theme: Theme;
  onToggleTheme: () => void;
  systemStatus: SystemStatus | null;
  stats: Stats | null;
  backendError?: string | null;
}

export function AppHeader({
  theme,
  onToggleTheme,
  systemStatus,
  stats,
  backendError,
}: AppHeaderProps) {
  const online = !backendError && systemStatus !== null;

  return (
    <header className="sticky top-0 z-30 border-b border-border/60 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex h-14 w-full max-w-[1800px] items-center gap-3 px-4">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-primary to-accent text-white shadow-lg shadow-primary/25">
            <GraduationCap className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold leading-tight tracking-tight">
              RAG <span className="text-gradient">NLP Assistant</span>
            </h1>
            <p className="truncate text-[10.5px] leading-tight text-muted-foreground">
              Retrieval-Augmented Generation lab · grounded answers with citations
            </p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-1.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant={online ? "success" : "destructive"}
                className="hidden gap-1.5 md:inline-flex"
              >
                <StatusDot status={online ? "ok" : "warn"} />
                {online ? `API v${systemStatus?.version}` : "API offline"}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              {online
                ? `${systemStatus?.llm.name} · ${systemStatus?.embeddings.name} · ${systemStatus?.vector_store.name}`
                : (backendError ?? "Cannot reach http://127.0.0.1:8000 — start the FastAPI server.")}
            </TooltipContent>
          </Tooltip>

          {stats ? (
            <Badge variant="mono" className="hidden lg:inline-flex">
              {stats.vector_store_backend} · {stats.total_chunks} chunks
            </Badge>
          ) : null}

          <Separator orientation="vertical" className="mx-0.5 hidden h-6 sm:block" />

          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="outline" className="hidden font-mono capitalize sm:inline-flex">
                {systemStatus?.environment ?? "…"}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>FastAPI environment reported by /api/system/status</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                size="icon-sm"
                variant="ghost"
                onClick={onToggleTheme}
                aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              >
                {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>{theme === "dark" ? "Light mode" : "Dark mode"}</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </header>
  );
}
