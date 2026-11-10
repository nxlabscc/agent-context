import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import type { TurnRequest } from "../types";
import { linearContext } from "./linear";

const open = { scope: { tenantId: "t" }, sessionId: "S", agent: "assistant", systemPrompt: "persona" };
const req = (window: number): TurnRequest => ({
  scope: { tenantId: "t" }, sessionId: "S", conversationId: null, agent: "assistant",
  system: "", query: "", history: [], fresh: [], window, toolTokens: 0,
});
const toolResult = (chars: number): ModelMessage =>
  ({ role: "tool", content: [{ type: "tool-result", toolCallId: "c", toolName: "run", output: { type: "text", value: "x".repeat(chars) } }] }) as ModelMessage;

test("the linear strategy puts the open blocks in the prompt and compacts under pressure", async () => {
  const ctx = linearContext({ open: () => ["<clock>12:00</clock>"], compaction: { threshold: 0.5 } });
  expect(await ctx.atOpen!(open)).toEqual(["<clock>12:00</clock>"]);

  const tail: ModelMessage[] = Array.from({ length: 20 }, () => toolResult(5_000));
  const old: ModelMessage[] = [toolResult(5_000)];
  const out = ctx.prepareStep!({ steps: [{ usage: { inputTokens: 80 } }], messages: [...old, ...tail] }, req(100));
  expect(out).toBeDefined();
});

test("with no compaction configured the strategy only sets open blocks", async () => {
  const ctx = linearContext({ open: () => ["a"] });
  expect(await ctx.atOpen!(open)).toEqual(["a"]);
  expect(ctx.prepareStep?.({ steps: [], messages: [] }, req(100))).toBeUndefined();
});
