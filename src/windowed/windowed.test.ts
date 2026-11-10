import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import type { StoredMessage, TurnRequest } from "../types";
import type { TrimReport } from "./trim";
import { windowContext } from "./windowed";

const turn = (chars: number): ModelMessage[] => [
  { role: "user", content: "u".repeat(chars) },
  { role: "assistant", content: "a".repeat(chars) },
];
const call = (id: string, tool: string): ModelMessage => ({ role: "assistant", content: [{ type: "tool-call", toolCallId: id, toolName: tool, input: {} }] }) as ModelMessage;
const result = (id: string, tool: string, value: string): ModelMessage => ({ role: "tool", content: [{ type: "tool-result", toolCallId: id, toolName: tool, output: { type: "text", value } }] }) as ModelMessage;

function thread(): StoredMessage[] {
  const messages: ModelMessage[] = [];
  for (let i = 0; i < 12; i++) {
    messages.push({ role: "user", content: `q${i}` }, call(`c${i}`, "get_page"), result(`c${i}`, "get_page", "x".repeat(3_000)), { role: "assistant", content: "ok" });
  }
  return messages.map((message, i) => ({ idx: i + 1, message }));
}

const req = (history: StoredMessage[]): TurnRequest => ({
  scope: { tenantId: "t" }, sessionId: "S", conversationId: null, agent: "assistant",
  system: "", query: "", history, fresh: turn(10), window: 5_000, toolTokens: 0,
});

test("the windowed strategy prunes old tool outputs, which relieves the pressure before trimming", async () => {
  const ctx = windowContext({ prune: { maxChars: 2_000, ttl: { default: 2 } }, trim: { floor: () => 999 } });
  const { history, report } = await ctx.prepareMessages!(req(thread()));
  const r = report as unknown as TrimReport;
  // Pruning turned the old tool outputs into cards, so the thread is small and
  // trimming has nothing left to drop. The two compose: prune first.
  const anyPruned = history.some((m) => Array.isArray(m.message.content) && m.message.content.some((p) => {
    const out = (p as { output?: { value?: unknown } }).output?.value;
    return typeof out === "string" && out.includes("pruned");
  }));
  expect(anyPruned).toBe(true);
  expect(r.dropped).toBe(0);
  // Trimming still ran: the report carries the budget it measured.
  expect(r.fifo).toBeLessThanOrEqual(r.space);
});

test("the floor holds what the memory has not absorbed", async () => {
  const ctx = windowContext({ trim: { floor: () => 4 } }); // only idx 1..4 may go
  const { report } = await ctx.prepareMessages!(req(thread()));
  expect((report as unknown as TrimReport).dropped).toBe(4);
  expect((report as unknown as TrimReport).pressure).toBe(true);
});
