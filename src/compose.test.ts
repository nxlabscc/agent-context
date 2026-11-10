import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { reactor } from "./reactor";
import { compose } from "./compose";
import type { ContextBuilder, OpenRequest, TurnRequest } from "./types";

const open: OpenRequest = { scope: { tenantId: "t" }, sessionId: "S", agent: "a", systemPrompt: "persona" };
const turn: TurnRequest = {
  scope: { tenantId: "t" },
  sessionId: "S",
  conversationId: null,
  agent: "a",
  system: "persona",
  query: "hello",
  history: [{ idx: 1, message: { role: "user", content: "old" } }, { idx: 2, message: { role: "assistant", content: "older" } }],
  fresh: [{ role: "user", content: "hello" }],
  window: 1000,
  toolTokens: 0,
};

test("blocks concatenate in order", async () => {
  const ctx = compose(
    reactor({ atOpen: () => ["clock"] }),
    reactor({ atOpen: async (req) => [`state of ${req.agent}`] }),
    reactor({ beforeTurn: () => [{ content: "memory", persist: true }] }),
    reactor({ beforeTurn: (req) => [{ content: `voice for ${req.query}`, persist: false }] }),
  );
  expect(await ctx.atOpen!(open)).toEqual(["clock", "state of a"]);
  expect(await ctx.beforeTurn!(turn)).toEqual([
    { content: "memory", persist: true },
    { content: "voice for hello", persist: false },
  ]);
});

test("prepareMessages chains: each step sees what the one before left, reports merge", async () => {
  const dropFirst: ContextBuilder = {
    async prepareMessages(req) {
      return { history: req.history.slice(1), report: { dropped: 1 } };
    },
  };
  const count: ContextBuilder = {
    async prepareMessages(req) {
      return { history: req.history, report: { seen: req.history.length } };
    },
  };
  const { history, report } = await compose(dropFirst, count).prepareMessages!(turn);
  expect(history.map((m) => m.idx)).toEqual([2]);
  expect(report).toEqual({ dropped: 1, seen: 1 });
});

test("prepareStep chains over the messages and is undefined when no step changed anything", () => {
  const nothing: ContextBuilder = { prepareStep: () => undefined };
  const tail: ContextBuilder = { prepareStep: (s) => ({ messages: s.messages.slice(-1) }) };
  const msgs: ModelMessage[] = [{ role: "user", content: "a" }, { role: "assistant", content: "b" }];
  expect(compose(nothing).prepareStep!({ steps: [], messages: msgs }, turn)).toBeUndefined();
  expect(compose(nothing, tail).prepareStep!({ steps: [], messages: msgs }, turn)).toEqual({
    messages: [{ role: "assistant", content: "b" }],
  });
});

test("afterTurn runs every step in order", async () => {
  const ran: string[] = [];
  const a: ContextBuilder = { afterTurn: async () => { ran.push("a"); } };
  const b: ContextBuilder = { afterTurn: async () => { ran.push("b"); } };
  await compose(a, b).afterTurn!(turn, { messages: [], history: turn.history });
  expect(ran).toEqual(["a", "b"]);
});

test("a builder that answers no moment composes to no change", async () => {
  const ctx = compose({});
  expect(await ctx.atOpen!(open)).toEqual([]);
  expect(await ctx.beforeTurn!(turn)).toEqual([]);
  expect((await ctx.prepareMessages!(turn)).history).toBe(turn.history);
});
