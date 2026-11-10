import { expect, test } from "bun:test";
import type { OpenRequest, TurnOutcome, TurnRequest } from "./types";
import { reactor } from "./reactor";

const open: OpenRequest = { scope: { tenantId: "t" }, sessionId: "S", agent: "a", systemPrompt: "p" };
const turn: TurnRequest = {
  scope: { tenantId: "t" }, sessionId: "S", conversationId: null, agent: "a",
  system: "p", query: "q", history: [{ idx: 1, message: { role: "user", content: "old" } }], fresh: [], window: 1000, toolTokens: 0,
};
const outcome: TurnOutcome = { messages: [], history: [] };

test("a reactor answers only the moments it was given", async () => {
  const ctx = reactor({ atOpen: () => ["clock"] });
  expect(ctx.atOpen).toBeDefined();
  expect(ctx.beforeTurn).toBeUndefined();
  expect(ctx.prepareMessages).toBeUndefined();
  expect(ctx.prepareStep).toBeUndefined();
  expect(ctx.afterTurn).toBeUndefined();
  expect(await ctx.atOpen!(open)).toEqual(["clock"]);
});

test("a synchronous reaction is normalized to the async method the interface declares", async () => {
  const ctx = reactor({
    atOpen: () => ["a"],
    beforeTurn: () => [{ content: "m", persist: true }],
    prepareMessages: (req) => ({ history: req.history, report: { seen: req.history.length } }),
  });
  expect(await ctx.atOpen!(open)).toEqual(["a"]);
  expect(await ctx.beforeTurn!(turn)).toEqual([{ content: "m", persist: true }]);
  expect(await ctx.prepareMessages!(turn)).toEqual({ history: turn.history, report: { seen: 1 } });
});

test("prepareStep stays synchronous, and afterTurn runs", async () => {
  const ran: string[] = [];
  const ctx = reactor({
    prepareStep: (state) => ({ messages: state.messages.slice(-1) }),
    afterTurn: async () => { ran.push("after"); },
  });
  expect(ctx.prepareStep!({ steps: [], messages: [{ role: "user", content: "x" }, { role: "assistant", content: "y" }] }, turn)).toEqual({
    messages: [{ role: "assistant", content: "y" }],
  });
  await ctx.afterTurn!(turn, outcome);
  expect(ran).toEqual(["after"]);
});
