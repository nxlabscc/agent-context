import type { ModelMessage } from "ai";
import type { ContextBuilder, Prepared, StoredMessage, TurnRequest } from "../types";

export type PruneOptions = {
  /** From how many characters a tool output stops travelling whole. Default 2000. */
  maxChars?: number;
  /**
   * How many of the person's turns an output lives whole, by tool and by default.
   * `0` is "already in the next turn". The default covers tools the strategy
   * does not know, MCP tools included. Default 10.
   */
  ttl?: { default?: number; byTool?: Record<string, number> };
  /** The sentence in the card that replaces the output. */
  note?: string;
};

export type PruneReport = {
  /** How many outputs travel as a card this turn. Cumulative over the thread: once expired, expired. */
  pruned: number;
  /** Characters the pruning took out of what is sent. */
  charsSaved: number;
};

/**
 * A tool output, in the three shapes the SDK produces:
 *
 * - `content`: typed parts, when the tool declares `toModelOutput` (an image as an image).
 * - `text`: the tool returned a string (MCP tools).
 * - `json`: the tool returned an object.
 *
 * All three can come large, so the pruning understands all three.
 */
type Output = { type?: string; value?: unknown };

type Part = {
  type?: string;
  toolCallId?: string;
  toolName?: string;
  input?: unknown;
  output?: Output;
};

const partsOf = (m: ModelMessage): Part[] =>
  Array.isArray(m.content) ? (m.content as unknown as Part[]) : [];

/** The call that produced the output, written as it is made again. */
const howToRecover = (tool: string, input: unknown): string =>
  input === undefined ? `${tool}()` : `${tool}(${JSON.stringify(input)})`;

const DEFAULT_NOTE =
  "This output was taken out of the history so it is not re-sent every turn. Ask for it again if you need it.";

/**
 * The card that replaces the text: what was there, how much it took, and the
 * call that brings it back. The arguments travel in the tool call, and they
 * are what makes the card actionable: tool outputs reach no memory, so making
 * the call again is the only way back.
 */
function card(original: string, tool: string, input: unknown, note: string): Record<string, unknown> {
  return {
    kind: "pruned",
    tool,
    chars: original.length,
    note,
    how_to_recover: howToRecover(tool, input),
  };
}

/**
 * The pruned output, or null when nothing in it was large.
 *
 * The card travels in the shape the original had, text where there was text
 * and an object where there was an object: the output's type is part of the
 * contract with the provider.
 */
function pruneOutput(
  output: Output | undefined,
  tool: string,
  input: unknown,
  maxChars: number,
  note: string,
): Output | null {
  if (output === undefined) return null;

  // Typed parts: one by one, because one may be an image, and that is not text.
  if (Array.isArray(output.value)) {
    let changed = false;
    const parts = (output.value as { text?: unknown }[]).map((v) => {
      if (typeof v?.text !== "string" || v.text.length <= maxChars) return v;
      changed = true;
      return { ...v, text: JSON.stringify(card(v.text, tool, input, note)) };
    });
    return changed ? { ...output, value: parts } : null;
  }

  if (typeof output.value === "string") {
    return output.value.length > maxChars
      ? { ...output, value: JSON.stringify(card(output.value, tool, input, note)) }
      : null;
  }

  const text = JSON.stringify(output.value ?? null);
  return text.length > maxChars ? { ...output, value: card(text, tool, input, note) } : null;
}

/**
 * How many of the person's turns come AFTER each message.
 *
 * A turn is a message of the person's: what separates one exchange from the
 * next is that someone speaks again, whatever the agent did in between. Counted
 * from the end, where the present hangs: the last message has zero.
 */
function turnsAfter(messages: ModelMessage[]): number[] {
  const out = new Array<number>(messages.length);
  let turns = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    out[i] = turns;
    if (messages[i]!.role === "user") turns++;
  }
  return out;
}

/**
 * Prune the large tool outputs that have expired, as a `prepareMessages`.
 *
 * A turn re-sends the whole conversation, and what a tool returned stays there
 * for good. Every large output gets an expiry, measured in the person's turns.
 * Once due, a card takes its place. The stored message is untouched: this
 * transforms what is sent.
 *
 * Expiring once is part of the design: the provider's cache is a byte-for-byte
 * prefix match, and an output that goes from whole to card at one moment, and
 * never back, breaks the cache once and never again.
 */
export function pruneToolOutputs(opts: PruneOptions = {}): ContextBuilder {
  const maxChars = opts.maxChars ?? 2_000;
  const defaultTtl = opts.ttl?.default ?? 10;
  const byTool = opts.ttl?.byTool ?? {};
  const note = opts.note ?? DEFAULT_NOTE;
  const ttlOf = (tool: string): number => byTool[tool] ?? defaultTtl;

  return {
    async prepareMessages(req: TurnRequest): Promise<Prepared> {
      const { history, fresh } = req;
      const all = [...history.map((m) => m.message), ...fresh];
      // What each tool was called with: it travels in the call, not in the result.
      const inputByCall = new Map<string, unknown>();
      for (const m of all) {
        for (const p of partsOf(m)) {
          if (p.type === "tool-call" && typeof p.toolCallId === "string") inputByCall.set(p.toolCallId, p.input);
        }
      }
      const turns = turnsAfter(all);
      let pruned = 0;
      let charsSaved = 0;

      const out: StoredMessage[] = history.map((stored, i) => {
        const m = stored.message;
        if (m.role !== "tool") return stored;
        let changed = false;
        const parts = partsOf(m).map((p) => {
          if (p.type !== "tool-result") return p;
          const tool = p.toolName ?? "";
          if (turns[i]! < ttlOf(tool)) return p;
          const before = JSON.stringify(p.output ?? null).length;
          const replaced = pruneOutput(p.output, tool, inputByCall.get(p.toolCallId ?? ""), maxChars, note);
          if (replaced === null) return p;
          changed = true;
          pruned++;
          charsSaved += before - JSON.stringify(replaced).length;
          return { ...p, output: replaced };
        });
        return changed ? { idx: stored.idx, message: { ...m, content: parts } as unknown as ModelMessage } : stored;
      });

      const report: PruneReport = { pruned, charsSaved };
      return { history: out, report };
    },
  };
}
