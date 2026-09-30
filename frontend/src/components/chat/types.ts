import type { AnswerResponse } from "@/types/api";

export interface UserMessage {
  id: string;
  role: "user";
  question: string;
  created_at: string;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  answer: AnswerResponse;
  created_at: string;
}

export type ChatMessage = UserMessage | AssistantMessage;

export function isAssistant(message: ChatMessage): message is AssistantMessage {
  return message.role === "assistant";
}
