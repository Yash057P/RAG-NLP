import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CornerDownLeft,
  Eraser,
  MessagesSquare,
  Send,
  Sparkles,
  Timer,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { exportHistory } from "@/lib/export";
import { cn } from "@/lib/utils";
import type { AnswerResponse, HistoryEntry, SystemStatus } from "@/types/api";

import { AnswerCard } from "./AnswerCard";
import { NoCorpusNotice, QuestionBubble, ThinkingBubble, WelcomePanel } from "./MessageBubble";
import { isAssistant, type ChatMessage } from "./types";

interface ChatInterfaceProps {
  messages: ChatMessage[];
  asking: boolean;
  pendingQuestion: string | null;
  activeAnswerId: string | null;
  hasDocuments: boolean;
  systemStatus: SystemStatus | null;
  stats: { questionsAsked: number; averageResponseTimeMs: number } | null;
  history: HistoryEntry[];
  onAsk: (question: string, topK: number) => void;
  onClear: () => void;
  onSelectAnswer: (answer: AnswerResponse) => void;
}

const TOP_K_OPTIONS = [3, 5, 8] as const;

export function ChatInterface({
  messages,
  asking,
  pendingQuestion,
  activeAnswerId,
  hasDocuments,
  systemStatus,
  stats,
  history,
  onAsk,
  onClear,
  onSelectAnswer,
}: ChatInterfaceProps) {
  const [draft, setDraft] = useState("");
  const [topK, setTopK] = useState<number>(5);
  const viewportRef = useRef<HTMLDivElement | null>(null);

  const examples = useMemo(
    () => systemStatus?.example_questions ?? [],
    [systemStatus],
  );
  const defaultTopK = systemStatus?.config.retrieval.top_k ?? 5;
  const maxLength = 500;

  useEffect(() => {
    if (TOP_K_OPTIONS.includes(defaultTopK as (typeof TOP_K_OPTIONS)[number])) {
      setTopK(defaultTopK);
    }
  }, [defaultTopK]);

  // Keep the transcript pinned to the newest message.
  useEffect(() => {
    const node = viewportRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [messages.length, asking]);

  const submit = useCallback(
    (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || asking) return;
      onAsk(trimmed, topK);
      setDraft("");
    },
    [asking, onAsk, topK],
  );

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit(draft);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* transcript ------------------------------------------------------- */}
      <ScrollArea className="min-h-0 flex-1" viewportRef={viewportRef}>
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-5">
          {messages.length === 0 && !asking ? (
            hasDocuments ? (
              <WelcomePanel examples={examples} onSelect={(question) => submit(question)} />
            ) : (
              <NoCorpusNotice />
            )
          ) : null}

          {messages.map((message) =>
            isAssistant(message) ? (
              <AnswerCard
                key={message.id}
                message={message}
                active={message.id === activeAnswerId}
                onSelect={(answer) => onSelectAnswer(answer)}
              />
            ) : (
              <QuestionBubble key={message.id} message={message} />
            ),
          )}

          {asking ? <ThinkingBubble question={pendingQuestion ?? undefined} /> : null}
        </div>
      </ScrollArea>

      {/* composer -------------------------------------------------------- */}
      <div className="shrink-0 border-t border-border/60 bg-background/40 backdrop-blur-xl">
        <div className="mx-auto w-full max-w-3xl px-4 py-3">
          <div className="glass rounded-2xl p-2 transition-colors focus-within:border-primary/40">
            <Textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value.slice(0, maxLength))}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents…  (Enter to send, Shift+Enter for a new line)"
              className="min-h-[64px] resize-none border-0 bg-transparent px-2 py-1.5 text-[13.5px] shadow-none focus-visible:ring-0"
              disabled={asking}
              aria-label="Question"
            />

            <div className="flex flex-wrap items-center gap-2 border-t border-border/50 px-1 pt-2">
              <div className="flex items-center gap-1">
                <span className="text-[10.5px] font-medium uppercase tracking-wide text-muted-foreground">
                  Top-K
                </span>
                {TOP_K_OPTIONS.map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => setTopK(option)}
                    disabled={asking}
                    className={cn(
                      "rounded-md px-1.5 py-0.5 font-mono text-[10.5px] transition-colors disabled:opacity-50",
                      topK === option
                        ? "bg-primary/20 text-primary"
                        : "text-muted-foreground hover:bg-accent/10",
                    )}
                  >
                    {option}
                  </button>
                ))}
              </div>

              {stats && stats.questionsAsked > 0 ? (
                <span className="hidden items-center gap-1 text-[10.5px] text-muted-foreground sm:flex">
                  <Timer className="h-3 w-3" />
                  {stats.questionsAsked} asked · {Math.round(stats.averageResponseTimeMs)} ms avg
                </span>
              ) : null}

              <div className="ml-auto flex items-center gap-1.5">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span>
                      <Button
                        type="button"
                        size="xs"
                        variant="ghost"
                        onClick={() => exportHistory(history, "markdown")}
                        disabled={!history.length}
                      >
                        <MessagesSquare className="h-3.5 w-3.5" />
                        Export
                      </Button>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>Download the chat history as Markdown</TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger asChild>
                    <span>
                      <Button
                        type="button"
                        size="xs"
                        variant="ghost"
                        onClick={onClear}
                        disabled={!messages.length || asking}
                        className="text-muted-foreground hover:text-destructive"
                      >
                        <Eraser className="h-3.5 w-3.5" />
                        Clear
                      </Button>
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>Clear the conversation and the server-side history</TooltipContent>
                </Tooltip>

                <Button
                  type="button"
                  size="sm"
                  variant="gradient"
                  onClick={() => submit(draft)}
                  loading={asking}
                  disabled={!draft.trim()}
                  className="h-8"
                >
                  {asking ? <Sparkles className="h-3.5 w-3.5" /> : <Send className="h-3.5 w-3.5" />}
                  Ask
                </Button>
              </div>
            </div>
          </div>

          <p className="mt-1.5 flex items-center justify-center gap-1 text-[10.5px] text-muted-foreground/70">
            <CornerDownLeft className="h-3 w-3" />
            Answers are generated strictly from the {topK} retrieved chunks — never from outside
            knowledge.
          </p>
        </div>
      </div>
    </div>
  );
}
