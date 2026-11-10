// What of the thread travels: pressure, the floor, and a cut at a whole turn.
import { expect, test } from "bun:test";
import type { ModelMessage } from "ai";
import { trimWindow } from "./trim";
import type { StoredMessage, TurnRequest } from "../types";
import type { TrimReport } from "./trim";

/** A turn: what the person says and what the agent answers, of the size asked. */
const turn = (chars: number): ModelMessage[] => [
  { role: "user", content: "u".repeat(chars) },
  { role: "assistant", content: "a".repeat(chars) },
];

/** N turns stored, with idx from 1 (0 is the session's system message). */
function thread(turns: number, chars: number): StoredMessage[] {
  return Array.from({ length: turns }, () => turn(chars))
    .flat()
    .map((message, i) => ({ idx: i + 1, message }));
}

const WINDOW = 10_000; // tokens

type Input = Partial<TurnRequest> & { floor?: number; onPressure?: (r: TrimReport) => void };

async function trim(input: Input) {
  const { floor = 999, onPressure, ...rest } = input;
  const req: TurnRequest = {
    scope: { tenantId: "t" },
    sessionId: "S",
    conversationId: null,
    agent: "a",
    system: "",
    query: "",
    history: [],
    fresh: [],
    window: WINDOW,
    toolTokens: 0,
    ...rest,
  };
  const builder = trimWindow({ floor: () => floor, onPressure: (_, r) => onPressure?.(r) });
  const { history, report } = await builder.prepareMessages!(req);
  return { history, report: report as unknown as TrimReport };
}

test("without pressure nothing is trimmed, even with everything past the floor", async () => {
  const history = thread(2, 100);
  const { history: sent, report } = await trim({ history, fresh: turn(10) });
  expect(report.dropped).toBe(0);
  expect(sent).toHaveLength(history.length);
  expect(report.pressure).toBe(false);
});

test("with pressure the oldest go until the FIFO is down to the target", async () => {
  // 20 turns of 2,000 characters: 80,000 chars ≈ 20,000 tokens, twice the window.
  const history = thread(20, 2_000);
  const { history: sent, report } = await trim({ history, fresh: turn(10) });
  expect(report.dropped).toBeGreaterThan(0);
  expect(report.fifo).toBeLessThanOrEqual(report.space * 0.25);
  expect(report.pressure).toBe(false);
  // What stays is the end of the thread, with its idx.
  expect(sent.at(-1)).toEqual(history.at(-1)!);
});

test("the cut always falls before a message of the person's", async () => {
  const { history: sent } = await trim({ history: thread(20, 2_000), fresh: turn(10) });
  expect(sent[0]!.message.role).toBe("user");
});

test("the floor holds: what is past it is kept however old", async () => {
  const history = thread(20, 2_000);
  const pressures: TrimReport[] = [];
  // Only the first two turns (idx 1..4) may go.
  const { report } = await trim({ history, fresh: turn(10), floor: 4, onPressure: (r) => pressures.push(r) });
  expect(report.dropped).toBe(4);
  // Trimming everything allowed did not reach the target.
  expect(report.fifo).toBeGreaterThan(report.space * 0.25);
  expect(report.pressure).toBe(true);
  expect(pressures).toHaveLength(1);
});

test("with a floor of -1 nothing is trimmed, and pressure is reported", async () => {
  const { report } = await trim({ history: thread(20, 2_000), fresh: turn(10), floor: -1 });
  expect(report.dropped).toBe(0);
  expect(report.pressure).toBe(true);
});

test("this turn's messages are never trimmed, even when they are all there is", async () => {
  const { history: sent, report } = await trim({ history: [], fresh: turn(50_000) });
  expect(report.dropped).toBe(0);
  expect(sent).toHaveLength(0);
});

test("each area has its space, and the FIFO's is what remains", async () => {
  // Two turns of a thousand characters, and a memory block in between as a
  // harness sends them: a system message inside the thread.
  const messages: ModelMessage[] = [...turn(1_000), { role: "system", content: "m".repeat(4_000) }, ...turn(1_000)];
  const history = messages.map((message, i) => ({ idx: i + 1, message }));
  const { report } = await trim({ history, system: "s".repeat(8_000), toolTokens: 300 });
  expect(report.fixed).toBe(2_000); // 8,000 chars / 4
  expect(report.tools).toBe(300);
  expect(report.memories).toBe(Math.ceil(JSON.stringify(messages[2]).length / 4));
  // The FIFO holds neither the system nor the memory: the conversation alone.
  expect(report.space).toBe(WINDOW - report.fixed - report.tools - report.memories);
  expect(report.fifo).toBeLessThan(report.space);
});

test("the system prompt and the tools shrink the FIFO's space, and that is where they press", async () => {
  const history = thread(4, 1_000);
  expect((await trim({ history })).report.dropped).toBe(0);
  expect((await trim({ history, system: "s".repeat(30_000) })).report.dropped).toBeGreaterThan(0);
  expect((await trim({ history, toolTokens: 8_000 })).report.dropped).toBeGreaterThan(0);
});

test("an exhausted FIFO space divides by nothing: it presses and reports", async () => {
  const { report } = await trim({
    history: thread(4, 1_000),
    system: "s".repeat(WINDOW * 4 + 100), // the system alone eats the window
    floor: -1,
  });
  expect(report.space).toBe(0);
  expect(report.pressure).toBe(true);
});

/** One turn a call: a 2,000-char message and a 6,000-char answer, about 2k tokens. */
function growBy(history: StoredMessage[], idx: { n: number }) {
  history.push({ idx: idx.n++, message: { role: "user", content: "u".repeat(2_000) } });
  history.push({ idx: idx.n++, message: { role: "assistant", content: "a".repeat(6_000) } });
}

const nextTurn: ModelMessage[] = [{ role: "user", content: "next" }];

test("hysteresis: after a trim the window climbs back to high before the next cut", async () => {
  // One long-lived builder: the process-local memory of the last cut carries
  // across calls, so the conversation grows from `low` to `high` between trims
  // instead of being re-cut to `low` every turn.
  const builder = trimWindow({ floor: (req) => req.history.at(-1)?.idx ?? -1 });
  const history: StoredMessage[] = [];
  const idx = { n: 1 };
  const firsts: Array<number | undefined> = [];
  for (let t = 1; t <= 10; t++) {
    growBy(history, idx);
    const { history: sent } = await builder.prepareMessages!({
      scope: { tenantId: "t" }, sessionId: "S", conversationId: null, agent: "a", system: "", query: "",
      history: [...history], fresh: nextTurn, window: WINDOW, toolTokens: 0,
    });
    firsts.push(sent[0]?.idx);
  }
  // space*high = 7,000, space*low = 2,500. Untouched until it passes high (idx 1
  // through turn 3), one trim at turn 4 to idx 7, then that cut holds across the
  // climb (turns 5 and 6), the next trim at turn 7, and so on.
  expect(firsts).toEqual([1, 1, 1, 7, 7, 7, 13, 13, 13, 19]);
});

test("a wired cut memory survives a new builder, as an ephemeral run needs", async () => {
  // A fresh builder every turn, as a process that lives one turn: the only thing
  // that carries the last cut over is the wired store, standing in for a mark the
  // harness persists. With it, the hysteresis holds across the restarts.
  const store = new Map<string, number>();
  const make = () =>
    trimWindow({
      floor: (req) => req.history.at(-1)?.idx ?? -1,
      cut: {
        load: (req) => store.get(req.sessionId) ?? null,
        save: (req, i) => void store.set(req.sessionId, i),
      },
    });
  const history: StoredMessage[] = [];
  const idx = { n: 1 };
  const firsts: Array<number | undefined> = [];
  for (let t = 1; t <= 7; t++) {
    growBy(history, idx);
    const { history: sent } = await make().prepareMessages!({
      scope: { tenantId: "t" }, sessionId: "S", conversationId: null, agent: "a", system: "", query: "",
      history: [...history], fresh: nextTurn, window: WINDOW, toolTokens: 0,
    });
    firsts.push(sent[0]?.idx);
  }
  expect(firsts).toEqual([1, 1, 1, 7, 7, 7, 13]);
});
