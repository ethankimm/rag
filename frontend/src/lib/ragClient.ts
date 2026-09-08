export interface ChatHistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  message: string;
  history: ChatHistoryMessage[];
}

export interface ChatResponse {
  answer: string;
  found: boolean;
  confidence: number;
  sources: string[];
}

const invalidResponseMessage =
  "The knowledge service returned an unexpected response.";

function isChatResponse(value: unknown): value is ChatResponse {
  if (typeof value !== "object" || value === null) return false;

  const response = value as Partial<ChatResponse>;
  return (
    typeof response.answer === "string" &&
    typeof response.found === "boolean" &&
    typeof response.confidence === "number" &&
    Array.isArray(response.sources) &&
    response.sources.every((source) => typeof source === "string")
  );
}

export function parseChatResponse(value: unknown): ChatResponse {
  // Keep untrusted API data behind a single runtime validation boundary.
  if (!isChatResponse(value)) throw new Error(invalidResponseMessage);
  return value;
}

export async function sendChatMessage(
  apiBaseUrl: string,
  request: ChatRequest,
  signal?: AbortSignal,
  fetchImplementation: typeof fetch = fetch,
): Promise<ChatResponse> {
  const endpoint = `${apiBaseUrl.replace(/\/$/, "")}/chat`;
  const response = await fetchImplementation(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) {
    throw new Error(`The knowledge service returned ${response.status}.`);
  }
  return parseChatResponse(await response.json());
}
