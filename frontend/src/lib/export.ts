/** Client-side export helpers: chat history (JSON / Markdown) and answer PDFs. */

import type { AnswerResponse, HistoryEntry } from "@/types/api";

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revoke on the next tick so Safari has time to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function timestampSlug(): string {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

export function exportHistoryAsJson(entries: HistoryEntry[]): void {
  const payload = {
    exported_at: new Date().toISOString(),
    total: entries.length,
    entries,
  };
  download(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
    `rag-chat-history-${timestampSlug()}.json`,
  );
}

export function exportHistoryAsMarkdown(entries: HistoryEntry[]): void {
  const lines: string[] = [
    "# RAG Chat History",
    "",
    `_Exported ${new Date().toLocaleString()} - ${entries.length} question(s)_`,
    "",
  ];

  entries.forEach((entry, index) => {
    lines.push(`## ${index + 1}. ${entry.question}`);
    lines.push("");
    lines.push(`- **Confidence**: ${(entry.confidence * 100).toFixed(0)}% (${entry.confidence_label})`);
    lines.push(`- **Response time**: ${Math.round(entry.response_time_ms)} ms`);
    lines.push(`- **Top similarity**: ${entry.top_score.toFixed(4)}`);
    lines.push(`- **Model**: ${entry.llm_model || "n/a"}`);
    lines.push(`- **Asked at**: ${new Date(entry.created_at).toLocaleString()}`);
    lines.push("");
    lines.push("**Answer**");
    lines.push("");
    lines.push(entry.answer);
    lines.push("");
    if (entry.sources.length) {
      lines.push("**Sources**");
      lines.push("");
      entry.sources.forEach((source) => {
        const page = source.page ? `, page ${source.page}` : "";
        lines.push(
          `- \`${source.filename}\` - chunk #${source.chunk_index}${page} (similarity ${source.score.toFixed(4)})`,
        );
        lines.push(`  > ${source.snippet}`);
      });
      lines.push("");
    }
    lines.push("---");
    lines.push("");
  });

  download(
    new Blob([lines.join("\n")], { type: "text/markdown" }),
    `rag-chat-history-${timestampSlug()}.md`,
  );
}

export function exportHistory(entries: HistoryEntry[], format: "json" | "markdown"): void {
  if (!entries.length) return;
  if (format === "json") exportHistoryAsJson(entries);
  else exportHistoryAsMarkdown(entries);
}

/** Render a single answer (with its sources) to a PDF and download it. */
export async function downloadAnswerAsPdf(answer: AnswerResponse): Promise<void> {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ unit: "pt", format: "a4" });

  const margin = 56;
  const pageWidth = doc.internal.pageSize.getWidth();
  const contentWidth = pageWidth - margin * 2;
  let cursor = margin;

  const ensureSpace = (needed: number) => {
    if (cursor + needed > doc.internal.pageSize.getHeight() - margin) {
      doc.addPage();
      cursor = margin;
    }
  };

  // --- header ------------------------------------------------------------
  doc.setFillColor(99, 62, 231);
  doc.rect(0, 0, pageWidth, 6, "F");
  cursor = margin + 10;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(17);
  doc.setTextColor(24, 24, 32);
  const titleLines = doc.splitTextToSize("RAG Answer Report", contentWidth) as string[];
  doc.text(titleLines, margin, cursor);
  cursor += titleLines.length * 20 + 4;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(110, 110, 125);
  doc.text(
    `Generated ${new Date().toLocaleString()}  -  model: ${answer.llm.model}  -  ${Math.round(
      answer.response_time_ms,
    )} ms`,
    margin,
    cursor,
  );
  cursor += 22;

  // --- question ----------------------------------------------------------
  const section = (label: string, value: string, bold = false) => {
    ensureSpace(34);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.setTextColor(110, 60, 200);
    doc.text(label.toUpperCase(), margin, cursor);
    cursor += 14;
    doc.setFont("helvetica", bold ? "bold" : "normal");
    doc.setFontSize(bold ? 12 : 11);
    doc.setTextColor(24, 24, 32);
    const lines = doc.splitTextToSize(value, contentWidth) as string[];
    doc.text(lines, margin, cursor);
    cursor += lines.length * (bold ? 16 : 15) + 12;
  };

  section("Question", answer.question, true);
  section("Answer", answer.answer);

  // --- metadata table ----------------------------------------------------
  ensureSpace(96);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(110, 60, 200);
  doc.text("METRICS", margin, cursor);
  cursor += 16;

  const rows: Array<[string, string]> = [
    ["Grounded in documents", answer.grounded ? "Yes" : "No"],
    ["Confidence", `${(answer.confidence * 100).toFixed(0)}% (${answer.confidence_label})`],
    ["Top similarity", answer.retrieval.top_score.toFixed(4)],
    ["Chunks retrieved", String(answer.retrieved_chunks.length)],
    ["Grounding coverage", `${(answer.retrieval.grounding_coverage * 100).toFixed(0)}%`],
    ["Embedding model", answer.retrieval.embedding_model],
    ["Vector store", answer.retrieval.vector_store_backend],
  ];

  doc.setFontSize(9.5);
  rows.forEach(([label, value], index) => {
    if (index % 2 === 0) {
      doc.setFillColor(246, 245, 252);
      doc.rect(margin, cursor - 11, contentWidth, 17, "F");
    }
    doc.setFont("helvetica", "bold");
    doc.setTextColor(90, 90, 105);
    doc.text(label, margin + 6, cursor);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(30, 30, 40);
    doc.text(String(value), margin + 230, cursor);
    cursor += 17;
  });
  cursor += 16;

  // --- sources -----------------------------------------------------------
  if (answer.sources.length) {
    ensureSpace(40);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.setTextColor(110, 60, 200);
    doc.text("SOURCES", margin, cursor);
    cursor += 16;

    answer.sources.forEach((source, index) => {
      ensureSpace(64);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(10);
      doc.setTextColor(30, 30, 40);
      const page = source.page ? ` - page ${source.page}` : "";
      doc.text(
        `[${index + 1}] ${source.filename} - chunk #${source.chunk_index}${page} - similarity ${source.score.toFixed(4)}`,
        margin,
        cursor,
      );
      cursor += 15;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(9.5);
      doc.setTextColor(85, 85, 100);
      const snippet = doc.splitTextToSize(`"${source.snippet}"`, contentWidth) as string[];
      doc.text(snippet, margin + 10, cursor);
      cursor += snippet.length * 12 + 10;
    });
  }

  // --- footer ------------------------------------------------------------
  const pageCount = doc.getNumberOfPages();
  for (let page = 1; page <= pageCount; page += 1) {
    doc.setPage(page);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 160);
    doc.text(
      "Generated by the RAG-NLP Assistant - answers are grounded in the uploaded documents only.",
      margin,
      doc.internal.pageSize.getHeight() - 28,
    );
    doc.text(
      `Page ${page} of ${pageCount}`,
      pageWidth - margin - 40,
      doc.internal.pageSize.getHeight() - 28,
    );
  }

  const slug = answer.question
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40);
  doc.save(`rag-answer-${slug || "question"}.pdf`);
}
