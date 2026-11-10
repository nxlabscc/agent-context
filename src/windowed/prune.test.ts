// Large tool outputs expire by the person's turns; a card that says how to get them back takes their place.
import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { pruneToolOutputs } from "./prune";
import type { StoredMessage, TurnRequest } from "../types";
import type { PruneReport } from "./prune";

const call = (id: string, tool: string, input: unknown): ModelMessage =>
  ({ role: "assistant", content: [{ type: "tool-call", toolCallId: id, toolName: tool, input }] }) as ModelMessage;

const result = (id: string, tool: string, output: unknown): ModelMessage =>
  ({ role: "tool", content: [{ type: "tool-result", toolCallId: id, toolName: tool, output }] }) as ModelMessage;

const user = (text: string): ModelMessage => ({ role: "user", content: text });
const assistant = (text: string): ModelMessage => ({ role: "assistant", content: text });

const stored = (messages: ModelMessage[]): StoredMessage[] => messages.map((message, i) => ({ idx: i + 1, message }));

async function prune(history: StoredMessage[], fresh: ModelMessage[], opts?: Parameters<typeof pruneToolOutputs>[0]) {
  const req: TurnRequest = {
    scope: { tenantId: "t" },
    sessionId: "S",
    conversationId: null,
    agent: "a",
    system: "",
    query: "",
    history,
    fresh,
    window: 100_000,
    toolTokens: 0,
  };
  const { history: out, report } = await pruneToolOutputs(opts).prepareMessages!(req);
  return { out, report: report as unknown as PruneReport };
}

const output = (m: StoredMessage) => (m.message.content as Array<{ output: { type: string; value: unknown } }>)[0]!.output;

/** One exchange where the agent called `tool` and got `value` back, then answered. */
const exchange = (id: string, tool: string, input: unknown, value: unknown, question = "q"): ModelMessage[] => [
  user(question),
  call(id, tool, input),
  result(id, tool, { type: "text", value }),
  assistant("done"),
];

test("a large output that has lived its turns becomes a card in the shape it had", async () => {
  const history = stored([
    ...exchange("c1", "get_page", { id: "p1" }, "x".repeat(3_000)),
    ...exchange("c2", "get_page", { id: "p2" }, "y".repeat(3_000)),
  ]);
  // The first output has one of the person's turns after it in the history and
  // this turn's message on top: two turns, so a TTL of 2 expires it.
  const { out, report } = await prune(history, [user("next")], { ttl: { default: 2 } });

  const first = output(out[2]!);
  expect(first.type).toBe("text");
  const card = JSON.parse(first.value as string) as Record<string, unknown>;
  expect(card.kind).toBe("pruned");
  expect(card.tool).toBe("get_page");
  expect(card.chars).toBe(3_000);
  expect(card.how_to_recover).toBe('get_page({"id":"p1"})');

  // The second output has only this turn after it: it still travels whole.
  expect(output(out[6]!).value).toBe("y".repeat(3_000));
  expect(report.pruned).toBe(1);
  expect(report.charsSaved).toBeGreaterThan(2_000);
});

test("a small output never expires", async () => {
  const history = stored(exchange("c1", "get_page", {}, "short"));
  const { out, report } = await prune(history, [user("next")], { ttl: { default: 0 } });
  expect(output(out[2]!).value).toBe("short");
  expect(report.pruned).toBe(0);
});

test("the TTL is per tool: a tool with 0 expires in the very next turn, the default waits", async () => {
  const history = stored([
    ...exchange("c1", "read_file", { path: "a" }, "x".repeat(3_000)),
    ...exchange("c2", "search", { q: "a" }, "y".repeat(3_000)),
  ]);
  const { out, report } = await prune(history, [user("next")], { ttl: { default: 10, byTool: { read_file: 0 } } });
  expect(JSON.parse(output(out[2]!).value as string).kind).toBe("pruned");
  expect(output(out[6]!).value).toBe("y".repeat(3_000));
  expect(report.pruned).toBe(1);
});

test("a json output gets a json card, and typed parts are pruned one by one", async () => {
  const jsonOut = result("c1", "list", { type: "json", value: { rows: "r".repeat(3_000) } });
  const typed = result("c2", "read_file", {
    type: "content",
    value: [{ type: "text", text: "t".repeat(3_000) }, { type: "media", data: "…", mediaType: "image/png" }],
  });
  const history = stored([
    user("q"),
    call("c1", "list", {}),
    jsonOut,
    call("c2", "read_file", { path: "img" }),
    typed,
    assistant("done"),
  ]);
  const { out } = await prune(history, [user("next")], { ttl: { default: 0 } });

  const json = output(out[2]!);
  expect(json.type).toBe("json");
  expect((json.value as { kind: string }).kind).toBe("pruned");

  const parts = output(out[4]!).value as Array<Record<string, unknown>>;
  expect(JSON.parse(parts[0]!.text as string).kind).toBe("pruned");
  expect(parts[1]).toEqual({ type: "media", data: "…", mediaType: "image/png" });
});

test("the stored history is untouched: pruning returns new messages", async () => {
  const history = stored(exchange("c1", "get_page", {}, "x".repeat(3_000)));
  const before = JSON.stringify(history);
  await prune(history, [user("next")], { ttl: { default: 0 } });
  expect(JSON.stringify(history)).toBe(before);
});
