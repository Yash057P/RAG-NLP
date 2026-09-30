import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

interface MarkdownAnswerProps {
  content: string;
  className?: string;
}

/** Renders the LLM answer; `.prose-rag` in index.css owns the typography. */
export function MarkdownAnswer({ content, className }: MarkdownAnswerProps) {
  return (
    <div className={cn("prose-rag text-foreground/90", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}
