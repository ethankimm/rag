import { describe, expect, it, vi } from "vitest";

import { parseChatResponse, sendChatMessage } from "./ragClient";

const validResponse = {
  answer: "Grounded answer",
  found: true,
  confidence: 0.91,
  sources: ["chapter1/1.md"],
};

describe("parseChatResponse", () => {
  it("accepts the complete backend response contract", () => {
    expect(parseChatResponse(validResponse)).toEqual(validResponse);
  });

  it("rejects missing or incorrectly typed fields", () => {
    expect(() => parseChatResponse({ answer: "incomplete" })).toThrow(
      "unexpected response",
    );
    expect(() =>
      parseChatResponse({ ...validResponse, sources: [42] }),
    ).toThrow("unexpected response");
  });
});

describe("sendChatMessage", () => {
  it("posts message history and parses the response", async () => {
    const fetchImplementation = vi.fn(async () =>
      new Response(JSON.stringify(validResponse), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    ) as unknown as typeof fetch;
    const request = {
      message: "How do they work?",
      history: [{ role: "user" as const, content: "Explain Transformers" }],
    };

    await expect(
      sendChatMessage("http://localhost:8000/", request, undefined, fetchImplementation),
    ).resolves.toEqual(validResponse);
    expect(fetchImplementation).toHaveBeenCalledWith(
      "http://localhost:8000/chat",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(request),
      }),
    );
  });

  it("reports non-success HTTP responses", async () => {
    const fetchImplementation = vi.fn(async () =>
      new Response("failure", { status: 503 }),
    ) as unknown as typeof fetch;

    await expect(
      sendChatMessage(
        "http://localhost:8000",
        { message: "question", history: [] },
        undefined,
        fetchImplementation,
      ),
    ).rejects.toThrow("503");
  });
});
