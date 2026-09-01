import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import ChatApp from "./ChatApp";

function response(answer: string, source = "chapter1/1.md") {
  return new Response(
    JSON.stringify({
      answer,
      found: true,
      confidence: 0.9,
      sources: [source],
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ChatApp", () => {
  it("submits a first turn and renders grounded evidence", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () =>
      response("Transformers use attention."),
    );
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ChatApp />);

    await user.type(
      screen.getByLabelText("Ask a question"),
      "How do Transformers work?",
    );
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByText("Transformers use attention.")).toBeVisible();
    expect(screen.getByText("chapter1/1.md")).toBeVisible();
    const payload = JSON.parse(fetchMock.mock.calls[0][1]?.body as string);
    expect(payload).toEqual({ message: "How do Transformers work?", history: [] });
  });

  it("sends successful prior turns with a contextual follow-up", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(response("They use attention."))
      .mockResolvedValueOnce(
        response("Tokenizers convert text to tokens.", "chapter2/4.md"),
      );
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ChatApp />);
    const composer = screen.getByLabelText("Ask a question");

    await user.type(composer, "Explain Transformers");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    await screen.findByText("They use attention.");
    await user.type(composer, "What about tokenizers?");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    await screen.findByText("Tokenizers convert text to tokens.");

    const payload = JSON.parse(fetchMock.mock.calls[1][1]?.body as string);
    expect(payload).toEqual({
      message: "What about tokenizers?",
      history: [
        { role: "user", content: "Explain Transformers" },
        { role: "assistant", content: "They use attention." },
      ],
    });
  });

  it("shows loading, reports failures, and excludes errors from history", async () => {
    let resolveRequest: ((value: Response) => void) | undefined;
    const pending = new Promise<Response>((resolve) => {
      resolveRequest = resolve;
    });
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockReturnValueOnce(pending)
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(response("Recovered"));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ChatApp />);
    const composer = screen.getByLabelText("Ask a question");

    await user.type(composer, "First question");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    expect(screen.getByLabelText("Searching")).toBeVisible();
    expect(composer).toBeDisabled();
    resolveRequest?.(response("First answer"));
    await screen.findByText("First answer");

    await user.type(composer, "Second question");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    expect(await screen.findByText(/couldn’t reach/)).toBeVisible();
    await user.type(composer, "Third question");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    await screen.findByText("Recovered");

    const payload = JSON.parse(fetchMock.mock.calls[2][1]?.body as string);
    expect(payload.history).not.toContainEqual(
      expect.objectContaining({ content: expect.stringMatching(/couldn’t reach/) }),
    );
  });

  it("clears browser-memory history with New conversation", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => response("Answer"));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<ChatApp />);

    await user.type(screen.getByLabelText("Ask a question"), "Question");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    await screen.findByText("Answer");
    await user.click(screen.getByRole("button", { name: "New conversation" }));

    await waitFor(() => {
      expect(screen.queryByText("Answer")).not.toBeInTheDocument();
    });
    expect(screen.getByText("What can I help you find?")).toBeVisible();
  });
});
