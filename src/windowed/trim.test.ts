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
