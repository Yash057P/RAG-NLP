import { useCallback, useEffect, useMemo, useState } from "react";
import { Info, Layers, MessageSquare, UploadCloud } from "lucide-react";
import { toast } from "sonner";

import { AppHeader } from "@/components/layout/AppHeader";
import { Panel } from "@/components/layout/Panel";
import { ChatInterface } from "@/components/chat/ChatInterface";
import { DocumentList } from "@/components/documents/DocumentList";
import { PipelineFlow } from "@/components/pipeline/PipelineFlow";
import { RetrievedChunks } from "@/components/retrieval/RetrievedChunks";
import { SourceCitations } from "@/components/retrieval/SourceCitations";
import { StatsCards } from "@/components/stats/StatsCards";
import { UploadDropzone } from "@/components/upload/UploadDropzone";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useDocuments } from "@/hooks/useDocuments";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { useSystemData } from "@/hooks/useSystemData";
import { useTheme } from "@/hooks/useTheme";
import { ApiError, askQuestion, clearHistory, getHistory, resetSystem } from "@/lib/api";
import type { AnswerResponse, HistoryEntry, PipelineTrace, UploadResponse } from "@/types/api";
import type { AssistantMessage, ChatMessage, UserMessage } from "@/components/chat/types";

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const isDesktop = useMediaQuery("(min-width: 1280px)");

  const { stats, status, error: systemError, loading: systemLoading, refresh: refreshSystem } =
    useSystemData();
  const {
    documents,
    loading: documentsLoading,
    upload,
    rebuilding,
    uploadFiles,
    deleteDocument,
    rebuildIndex,
    loadChunks,
    refresh: refreshDocuments,
  } = useDocuments();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [asking, setAsking] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [activeAnswer, setActiveAnswer] = useState<AnswerResponse | null>(null);
  const [lastTrace, setLastTrace] = useState<PipelineTrace | null>(null);

  const readyDocuments = useMemo(
    () => documents.filter((doc) => doc.status === "ready"),
    [documents],
  );

  // --- history (used by the export button) --------------------------------
  const refreshHistory = useCallback(async () => {
    try {
      const response = await getHistory();
      setHistory(response.entries);
    } catch {
      /* the export button simply stays disabled */
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  // --- ask ----------------------------------------------------------------
  const handleAsk = useCallback(
    (question: string, topK: number) => {
      if (asking) return;
      const now = new Date().toISOString();
      const userMessage: UserMessage = {
        id: `q-${Date.now()}`,
        role: "user",
        question,
        created_at: now,
      };
      setMessages((current) => [...current, userMessage]);
      setPendingQuestion(question);
      setAsking(true);

      void (async () => {
        try {
          const answer = await askQuestion(question, topK);
          const assistantMessage: AssistantMessage = {
            id: answer.id,
            role: "assistant",
            answer,
            created_at: answer.created_at ?? now,
          };
          setMessages((current) => [...current, assistantMessage]);
          setActiveAnswer(answer);
          if (answer.pipeline) setLastTrace(answer.pipeline);
          if (answer.abstained) {
            toast.warning("No answer found in the documents", {
              description: "Retrieval did not find supporting context, so the model abstained.",
            });
          }
        } catch (cause) {
          const message =
            cause instanceof ApiError ? cause.message : "Unexpected error while asking.";
          toast.error("Request failed", { description: message });
          setMessages((current) => current.filter((item) => item.id !== userMessage.id));
        } finally {
          setAsking(false);
          setPendingQuestion(null);
          void refreshSystem();
          void refreshHistory();
        }
      })();
    },
    [asking, refreshHistory, refreshSystem],
  );

  // --- clear conversation -------------------------------------------------
  const handleClear = useCallback(async () => {
    setMessages([]);
    setActiveAnswer(null);
    try {
      await clearHistory();
      toast.success("Conversation cleared");
      void refreshSystem();
    } catch (cause) {
      toast.error("Could not clear the history", {
        description: cause instanceof ApiError ? cause.message : undefined,
      });
    }
    void refreshHistory();
  }, [refreshSystem]);

  // --- documents ----------------------------------------------------------
  const handleUploaded = useCallback(
    (response: UploadResponse) => {
      if (response.pipeline) setLastTrace(response.pipeline);
      void refreshSystem();
      void refreshDocuments();
    },
    [refreshDocuments, refreshSystem],
  );

  const handleReset = useCallback(async () => {
    try {
      const response = await resetSystem();
      toast.success("System reset", { description: response.message });
      setMessages([]);
      setActiveAnswer(null);
      await Promise.all([refreshDocuments(), refreshSystem()]);
      void refreshHistory();
    } catch (cause) {
      toast.error("Reset failed", {
        description: cause instanceof ApiError ? cause.message : "Unexpected error.",
      });
    }
  }, [refreshDocuments, refreshHistory, refreshSystem]);

  // --- panels -------------------------------------------------------------
  const leftPanel = (
    <div className="space-y-4">
      <UploadDropzone
        upload={upload}
        onUpload={uploadFiles}
        onUploaded={handleUploaded}
        systemStatus={status}
        onReset={handleReset}
      />
      <DocumentList
        documents={documents}
        loading={documentsLoading}
        rebuilding={rebuilding}
        onDelete={deleteDocument}
        onRebuild={rebuildIndex}
        onLoadChunks={loadChunks}
      />
      <StatsCards
        stats={stats}
        vectorStoreHealth={status?.vector_store}
        embeddingDetail={status?.embeddings.detail}
        loading={systemLoading}
      />
    </div>
  );

  const chatPanel = (
    <ChatInterface
      messages={messages}
      asking={asking}
      pendingQuestion={pendingQuestion}
      activeAnswerId={activeAnswer?.id ?? null}
      hasDocuments={readyDocuments.length > 0}
      systemStatus={status}
      stats={
        stats
          ? {
              questionsAsked: stats.questions_asked,
              averageResponseTimeMs: stats.average_response_time_ms,
            }
          : null
      }
      history={history}
      onAsk={handleAsk}
      onClear={handleClear}
      onSelectAnswer={setActiveAnswer}
    />
  );

  const rightPanel = (
    <div className="space-y-4">
      <RetrievedChunks answer={activeAnswer} query={activeAnswer?.question ?? ""} />
      <SourceCitations sources={activeAnswer?.sources ?? []} />
    </div>
  );

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex h-screen flex-col overflow-hidden">
        <AppHeader
          theme={theme}
          onToggleTheme={toggleTheme}
          systemStatus={status}
          stats={stats}
          backendError={systemError?.message}
        />

        {isDesktop ? (
          <main className="mx-auto grid min-h-0 w-full max-w-[1800px] flex-1 grid-cols-[minmax(300px,340px)_minmax(0,1fr)_minmax(300px,360px)] gap-3 p-3">
            <div className="min-h-0 overflow-y-auto scrollbar-thin pr-0.5">
              <div className="space-y-4">{leftPanel}</div>
              <div className="mt-4 rounded-2xl border border-border/50 bg-background/30 p-3">
                <PipelineFlow
                  definitions={status?.pipeline_stages ?? []}
                  trace={lastTrace}
                  loading={systemLoading}
                />
              </div>
            </div>

            <Panel className="min-h-0" bodyClassName="flex flex-col overflow-hidden p-0">
              {chatPanel}
            </Panel>

            <div className="min-h-0 overflow-y-auto scrollbar-thin pr-0.5">{rightPanel}</div>
          </main>
        ) : (
          <main className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col p-3">
            <Tabs defaultValue="chat" className="flex min-h-0 flex-1 flex-col gap-3">
              <TabsList className="w-full">
                <TabsTrigger value="upload" className="flex-1">
                  <UploadCloud className="h-3.5 w-3.5" />
                  Upload
                </TabsTrigger>
                <TabsTrigger value="chat" className="flex-1">
                  <MessageSquare className="h-3.5 w-3.5" />
                  Chat
                </TabsTrigger>
                <TabsTrigger value="retrieval" className="flex-1">
                  <Layers className="h-3.5 w-3.5" />
                  Retrieval
                </TabsTrigger>
              </TabsList>

              <TabsContent value="upload" className="min-h-0 flex-1 space-y-4 overflow-y-auto scrollbar-thin">
                {leftPanel}
                <div className="rounded-2xl border border-border/50 bg-background/30 p-3">
                  <PipelineFlow
                    definitions={status?.pipeline_stages ?? []}
                    trace={lastTrace}
                    loading={systemLoading}
                  />
                </div>
              </TabsContent>

              <TabsContent value="chat" className="min-h-0 flex-1">
                <Panel className="h-full" bodyClassName="flex flex-col overflow-hidden p-0">
                  {chatPanel}
                </Panel>
              </TabsContent>

              <TabsContent value="retrieval" className="min-h-0 flex-1 space-y-4 overflow-y-auto scrollbar-thin">
                <div className="rounded-2xl border border-border/50 bg-background/30 p-3">
                  <p className="mb-1.5 flex items-center gap-1.5 text-[10.5px] font-medium uppercase tracking-wide text-muted-foreground">
                    <Info className="h-3 w-3" />
                    Retrieval inspector
                  </p>
                  {rightPanel}
                </div>
              </TabsContent>
            </Tabs>
          </main>
        )}
      </div>
    </TooltipProvider>
  );
}
