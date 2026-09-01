import {
  AlertCircle,
  ArrowUp,
  BookOpen,
  Bot,
  FileText,
  MessageSquarePlus,
  Sparkles,
  User,
} from "lucide-react";
import {
  type ChangeEvent,
  type KeyboardEvent,
  type SyntheticEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  sendChatMessage,
  type ChatHistoryMessage,
} from "../lib/ragClient";

const apiBaseUrl = (import.meta.env.PUBLIC_RAG_API_URL ?? "/api").replace(
  /\/$/,
  "",
);

const suggestions = [
  "What is the difference between NLP and an LLM?",
  "How does a Transformer architecture work?",
  "How do I fine-tune a pretrained model?",
];
const MAX_HISTORY_MESSAGES = 12;

type ApiStatus = "ready" | "working" | "error";

const statusTextByStatus: Record<ApiStatus, string> = {
  ready: "Knowledge base connected",
  working: "Searching knowledge base",
  error: "Connection needs attention",
};

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  confidence?: number;
  sources?: string[];
  error?: boolean;
}

function messageId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

function sourceLabel(source: string) {
  const parts = source.split(/[\\/]/).filter(Boolean);
  return parts.slice(-2).join("/") || source;
}

function buildChatHistory(messages: ChatMessage[]): ChatHistoryMessage[] {
  // Connection-error messages are UI feedback, not conversational context.
  return messages
    .filter((message) => !message.error)
    .slice(-MAX_HISTORY_MESSAGES)
    .map(({ role, content }) => ({ role, content }));
}

export default function ChatApp() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<ApiStatus>("ready");
  const abortController = useRef<AbortController | null>(null);
  const endOfMessages = useRef<HTMLDivElement | null>(null);
  const textarea = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    endOfMessages.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  function resizeComposer(element: HTMLTextAreaElement) {
    element.style.height = "0";
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }

  function resetComposer() {
    setPrompt("");
    if (textarea.current) textarea.current.style.height = "auto";
  }

  function startNewChat() {
    abortController.current?.abort();
    abortController.current = null;
    setMessages([]);
    setIsLoading(false);
    setApiStatus("ready");
    resetComposer();
    textarea.current?.focus();
  }

  async function sendMessage() {
    const message = prompt.trim();
    if (!message || isLoading) return;

    const history = buildChatHistory(messages);

    setMessages((current) => [
      ...current,
      { id: messageId(), role: "user", content: message },
    ]);
    resetComposer();
    setIsLoading(true);
    setApiStatus("working");

    const controller = new AbortController();
    abortController.current = controller;

    try {
      const result = await sendChatMessage(
        apiBaseUrl,
        { message, history },
        controller.signal,
      );

      setMessages((current) => [
        ...current,
        {
          id: messageId(),
          role: "assistant",
          content: result.answer,
          confidence: result.confidence,
          sources: result.found ? result.sources : [],
        },
      ]);
      setApiStatus("ready");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setMessages((current) => [
        ...current,
        {
          id: messageId(),
          role: "assistant",
          content:
            "I couldn’t reach the knowledge service. Check that the API is running, then try again.",
          error: true,
        },
      ]);
      setApiStatus("error");
    } finally {
      // Starting a new chat clears the active controller before this completes.
      if (abortController.current === controller) {
        abortController.current = null;
        setIsLoading(false);
      }
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  function handlePromptChange(event: ChangeEvent<HTMLTextAreaElement>) {
    setPrompt(event.target.value);
    resizeComposer(event.target);
  }

  function handleSubmit(event: SyntheticEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage();
  }

  const statusText = statusTextByStatus[apiStatus];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <BookOpen size={18} />
          </span>
          <span>LLM Course Chat</span>
        </div>

        <button className="new-chat" type="button" onClick={startNewChat}>
          <MessageSquarePlus size={17} />
          New conversation
        </button>

        {messages.length > 0 && (
          <div className="current-chat">
            <span>Current conversation</span>
            <p>{messages.find((message) => message.role === "user")?.content}</p>
          </div>
        )}

        <div className={`sidebar-note status-${apiStatus}`}>
          <span className="status-dot" />
          <span>{statusText}</span>
        </div>
      </aside>

      <main className="chat-surface">
        <header className="topbar">
          <div>
            <span className="eyebrow">Hugging Face LLM Course</span>
            <h1>Ask the course guides</h1>
          </div>
          <span className="grounded-label">Grounded answers</span>
        </header>

        {messages.length === 0 ? (
          <section className="welcome" aria-labelledby="welcome-title">
            <span className="welcome-icon">
              <Sparkles size={22} />
            </span>
            <h2 id="welcome-title">What can I help you find?</h2>
            <p>
              Ask about Transformers, tokenizers, fine-tuning, datasets, demos,
              and the other topics covered by the Hugging Face LLM Course.
            </p>

            <div className="suggestions">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => setPrompt(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </section>
        ) : (
          <section
            className="messages"
            aria-label="Conversation"
            aria-live="polite"
          >
            {messages.map((message) => (
              <article
                className={`message message-${message.role}`}
                key={message.id}
              >
                <span className="message-avatar" aria-hidden="true">
                  {message.role === "assistant" ? (
                    <Bot size={18} />
                  ) : (
                    <User size={17} />
                  )}
                </span>
                <div className="message-body">
                  <span className="message-author">
                    {message.role === "assistant" ? "LLM Course Chat" : "You"}
                  </span>
                  <p className={message.error ? "error-message" : undefined}>
                    {message.error && <AlertCircle size={16} />}
                    {message.content}
                  </p>
                  {message.sources && message.sources.length > 0 && (
                    <div className="answer-evidence">
                      <span className="confidence">
                        {Math.round((message.confidence ?? 0) * 100)}% match
                      </span>
                      <div className="sources" aria-label="Answer sources">
                        {message.sources.map((source) => (
                          <span
                            className="source-chip"
                            title={source}
                            key={source}
                          >
                            <FileText size={13} />
                            {sourceLabel(source)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </article>
            ))}

            {isLoading && (
              <article
                className="message message-assistant"
                aria-label="Searching"
              >
                <span className="message-avatar" aria-hidden="true">
                  <Bot size={18} />
                </span>
                <div className="message-body">
                  <span className="message-author">LLM Course Chat</span>
                  <span className="typing-dots">
                    <i />
                    <i />
                    <i />
                  </span>
                </div>
              </article>
            )}
            <div ref={endOfMessages} />
          </section>
        )}

        <div className="composer-wrap">
          <form className="composer" onSubmit={handleSubmit}>
            <label className="sr-only" htmlFor="chat-prompt">
              Ask a question
            </label>
            <textarea
              ref={textarea}
              id="chat-prompt"
              rows={1}
              placeholder="Ask a question about the LLM Course…"
              value={prompt}
              onChange={handlePromptChange}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
            <button
              className="send-button"
              type="submit"
              aria-label="Send message"
              disabled={!prompt.trim() || isLoading}
            >
              <ArrowUp size={19} strokeWidth={2.4} />
            </button>
          </form>
          <p className="composer-hint">
            Low-variability answers are returned only above the confidence
            threshold.
          </p>
        </div>
      </main>
    </div>
  );
}
