import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { ContextExhausted, compactMessages, compaction } from "./compaction";
import type { TurnRequest } from "../types";

const big = (n: number) => "x".repeat(n);

function toolResult(id: string, name: string, chars: number): ModelMessage {
  return {
    role: "tool",
    content: [{ type: "tool-result", toolCallId: id, toolName: name, output: { type: "text", value: big(chars) } }],
  } as ModelMessage;
}

function assistantCall(id: string, name: string): ModelMessage {
  return {
    role: "assistant",
    content: [{ type: "tool-call", toolCallId: id, toolName: name, input: {} }],
  } as ModelMessage;
}

/** A transcript of `pairs` tool round-trips, oldest first. */
function transcript(pairs: number, chars = 5_000): ModelMessage[] {
  const out: ModelMessage[] = [
    { role: "system", content: "you are an agent" },
    { role: "user", content: "run your purpose" },
  ];
  for (let i = 0; i < pairs; i++) {
    out.push(assistantCall(`c${i}`, "run_shell"), toolResult(`c${i}`, "run_shell", chars));
  }
  return out;
}

const request = (window: number): TurnRequest => ({
  scope: { tenantId: "t" },
  sessionId: "S1",
  conversationId: null,
  agent: "a",
  system: "",
  query: "",
  history: [],
  fresh: [],
  window,
  toolTokens: 0,
});

test("old tool payloads go, ids and structure stay", () => {
  const msgs = transcript(20);
  const { compacted, stubbed, freedChars } = compactMessages(msgs);

  expect(compacted.length).toBe(msgs.length);
  expect(stubbed).toBeGreaterThan(0);
  expect(freedChars).toBeGreaterThan(0);

  // Every tool call still has its matching result, with the same id and name:
  // that pairing is what the provider validates.
  for (let i = 0; i < compacted.length; i++) {
    const m = compacted[i] as { role: string; content: unknown };
    if (m.role !== "tool") continue;
    const part = (m.content as Array<{ toolCallId: string; toolName: string }>)[0]!;
    const call = (compacted[i - 1] as { content: Array<{ toolCallId: string }> }).content[0]!;
    expect(part.toolCallId).toBe(call.toolCallId);
    expect(part.toolName).toBe("run_shell");
  }
});

test("the recent tail is left whole", () => {
  const msgs = transcript(20);
  const { compacted } = compactMessages(msgs);
  const last = compacted[compacted.length - 1] as { content: Array<{ output: { value: string } }> };
  expect(last.content[0]!.output.value).toBe(big(5_000));
});

test("the system and the first user message are never touched", () => {
  const msgs = transcript(20);
  const { compacted } = compactMessages(msgs);
  expect(compacted[0]).toEqual(msgs[0]!);
  expect(compacted[1]).toEqual(msgs[1]!);
});

test("a stub says the call ran, so the model does not re-run it to find out", () => {
  const { compacted } = compactMessages(transcript(20));
  const stub = (compacted[3] as { content: Array<{ output: { value: string } }> }).content[0]!.output.value;
  expect(stub).toContain("compacted");
  expect(stub).toContain("returned normally");
});

test("the stub text is the caller's when it gives one", () => {
  const { compacted } = compactMessages(transcript(20), { stub: ({ chars }) => `gone: ${chars}` });
  const stub = (compacted[3] as { content: Array<{ output: { value: string } }> }).content[0]!.output.value;
  expect(stub).toBe("gone: 5000");
});

test("small results are left alone: the stub would cost more", () => {
  const { stubbed } = compactMessages(transcript(20, 50));
  expect(stubbed).toBe(0);
});

test("below the threshold nothing happens at all", () => {
  const compact = compaction({ onCompact: () => { throw new Error("must not compact"); } });
  const out = compact.prepareStep!({ steps: [{ usage: { inputTokens: 100_000 } }], messages: transcript(20) }, request(1_000_000));
  expect(out).toBeUndefined();
});

test("past the threshold it compacts and reports what it freed", () => {
  const reports: unknown[] = [];
  const compact = compaction({ onCompact: (r) => reports.push(r) });
  const msgs = transcript(20);
  const out = compact.prepareStep!({ steps: [{ usage: { inputTokens: 900_000 } }], messages: msgs }, request(1_000_000));

  expect(out).toBeDefined();
  expect(out!.messages.length).toBe(msgs.length);
  expect(reports).toHaveLength(1);
  expect(reports[0]).toMatchObject({ sessionId: "S1", usedTokens: 900_000, window: 1_000_000 });
  expect((reports[0] as { stubbed: number }).stubbed).toBeGreaterThan(0);
});

test("with nothing old left to drop it ends the turn instead of overflowing", () => {
  const compact = compaction();
  // Two pairs: everything is inside the keep-recent window, nothing to stub.
  expect(() =>
    compact.prepareStep!({ steps: [{ usage: { inputTokens: 900_000 } }], messages: transcript(2) }, request(1_000_000)),
  ).toThrow(ContextExhausted);
});
