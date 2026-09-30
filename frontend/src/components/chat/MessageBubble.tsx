import { BookOpenCheck, Bot, HelpCircle, Loader2, ShieldAlert, Sparkles, User } from "lucide-react";

import { cn } from "@/lib/utils";

import type { UserMessage } from "./types";

export function QuestionBubble({ message }: { message: UserMessage }) {
  return (
    <div className="flex animate-fade-in items-start justify-end gap-2.5">
      <div className="max-w-[85%] rounded-2xl rounded-tr-sm border border-primary/25 bg-primary/10 px-3.5 py-2.5">
        <p className="text-[13.5px] leading-relaxed text-foreground">{message.question}</p>
      </div>
      <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-primary/15 text-primary">
        <User className="h-3.5 w-3.5" />
      </span>
    </div>
  );
}

/** Placeholder shown while the RAG pipeline is running for a question. */
export function ThinkingBubble({ question }: { question?: string }) {
  const steps = ["Embedding query", "Searching ChromaDB", "Building prompt", "Generating answer"];

  return (
    <div className="flex animate-fade-in items-start gap-2.5">
      <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-accent/15 text-accent">
        <Bot className="h-3.5 w-3.5" />
      </span>
      <div className="max-w-[88%] rounded-2xl rounded-tl-sm border border-border/70 bg-background/50 px-3.5 py-3">
        <div className="flex items-center gap-2">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
          <p className="text-[13px] font-medium">Retrieving context and generating…</p>
        </div>
        {question ? (
          <p className="mt-1.5 truncate text-[11.5px] italic text-muted-foreground">“{question}”</p>
        ) : null}
        <ul className="mt-2 space-y-1">
          {steps.map((step) => (
            <li key={step} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <Sparkles className="h-3 w-3 animate-blink" />
              {step}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** Shown when the corpus is empty: the chat cannot work without documents. */
export function NoCorpusNotice() {
  return (
    <div className="glass animate-fade-in rounded-2xl border-dashed p-5 text-center">
      <span className="mx-auto mb-2 grid h-10 w-10 place-items-center rounded-full bg-amber-500/15 text-amber-400">
        <ShieldAlert className="h-4 w-4" />
      </span>
      <p className="text-sm font-medium">No documents indexed yet</p>
      <p className="mx-auto mt-1 max-w-[46ch] text-xs leading-relaxed text-muted-foreground">
        The assistant only answers from your uploads. Add a PDF, DOCX or TXT file on the left, then
        ask your question again.
      </p>
    </div>
  );
}

export function WelcomePanel({ examples, onSelect, disabled }: {
  examples: string[];
  onSelect: (question: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="mx-auto max-w-xl space-y-4 py-6 text-center">
      <span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl border border-primary/30 bg-primary/10 text-primary">
        <BookOpenCheck className="h-5 w-5" />
      </span>
      <div>
        <h2 className="text-lg font-semibold tracking-tight">
          Ask anything about your <span className="text-gradient">documents</span>
        </h2>
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
          Every answer is grounded in the retrieved chunks, cites its sources and carries a
          confidence score. If the corpus has no answer, the model says so.
        </p>
      </div>
      <div className="space-y-1.5 text-left">
        <p className="flex items-center justify-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          <HelpCircle className="h-3 w-3" />
          Try one of these
        </p>
        <div className="grid gap-1.5 sm:grid-cols-2">
          {examples.slice(0, 6).map((question) => (
            <button
              key={question}
              type="button"
              disabled={disabled}
              onClick={() => onSelect(question)}
              className={cn(
                "glass glass-hover rounded-xl px-3 py-2 text-left text-[12px] leading-snug text-foreground/85",
                "disabled:cursor-not-allowed disabled:opacity-50",
              )}
            >
              {question}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
