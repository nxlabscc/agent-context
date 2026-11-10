import type { ModelMessage } from "ai";
import type { ContextBuilder, StepState, TurnRequest } from "../types";

export type CompactionOptions = {
  /** Compact once the measured context reaches this share of the window. Default 0.8. */
  threshold?: number;
  /** Messages at the tail left whole: the model is mid-task and reasons against them. Default 12. */
  keepRecent?: number;
  /** A tool result shorter than this costs less than the stub explaining it. Default 400. */
  minStubChars?: number;
  /** The text that replaces a dropped result. The default says the call ran and what it was. */
  stub?: (info: { chars: number; toolName: string }) => string;
  /** Told each time a step was compacted, with what it freed. */
  onCompact?: (report: CompactionReport) => void;
};

export type CompactionReport = {
  sessionId: string;
  usedTokens: number;
  window: number;
  freedChars: number;
  stubbed: number;
};

/**
 * The turn filled its context with material compaction is not allowed to touch.
 *
 * Thrown instead of letting the next call breach the window, so the turn ends
 * through the harness's ordinary failure path rather than as a provider
 * exception nobody can read.
 */
export class ContextExhausted extends Error {
  constructor(
    readonly sessionId: string,
    readonly usedTokens: number,
    readonly window: number,
  ) {
    super(
      `context exhausted: the turn reached ${usedTokens} of ${window} tokens and everything older ` +
        `had already been compacted, so there was nothing left to free`,
    );
    this.name = "ContextExhausted";
  }
}

const defaultStub = ({ chars, toolName }: { chars: number; toolName: string }): string =>
  `[compacted: ${chars} chars of ${toolName} output dropped to make room. This call ran and ` +
  `returned normally; run it again only if you need the contents, not to find out whether it worked.]`;

function outputChars(output: unknown): number {
  if (output == null) return 0;
  const v = (output as { value?: unknown }).value;
  if (typeof v === "string") return v.length;
  try {
    return JSON.stringify(v ?? output).length;
  } catch {
    return 0;
  }
}

/**
 * Replace the payload of older tool results with a stub, leaving every message,
 * every tool call and every id in place.
 *
 * Mechanical on purpose: no model call, so it cannot invent a summary of what it
 * dropped, costs nothing, and takes the same decision every time for the same
 * transcript. Tool results are where the volume is.
 *
 * The pairing between a tool call and its result is what the provider
 * validates, so the result part keeps its `toolCallId` and `toolName` and only
 * its `output` is swapped. A stubbed step still reads as "you ran this and got
 * an answer", which is what the model needs to not run it again.
 */
export function compactMessages(
  messages: ModelMessage[],
  opts: Pick<CompactionOptions, "keepRecent" | "minStubChars" | "stub"> = {},
): { compacted: ModelMessage[]; freedChars: number; stubbed: number } {
  const keepRecent = opts.keepRecent ?? 12;
  const minStubChars = opts.minStubChars ?? 400;
  const stub = opts.stub ?? defaultStub;
  const cutoff = Math.max(0, messages.length - keepRecent);
  let freedChars = 0;
  let stubbed = 0;

  const compacted = messages.map((msg, i) => {
    if (i >= cutoff || msg.role !== "tool" || !Array.isArray(msg.content)) return msg;
    const content = msg.content.map((part) => {
      const p = part as { type?: string; output?: unknown; toolName?: string };
      if (p.type !== "tool-result") return part;
      const size = outputChars(p.output);
      if (size < minStubChars) return part;
      freedChars += size;
      stubbed++;
      return {
        ...p,
        output: { type: "text", value: stub({ chars: size, toolName: p.toolName ?? "tool" }) },
      };
    });
    return { ...msg, content } as ModelMessage;
  });

  return { compacted, freedChars, stubbed };
}

/**
 * Keep a long turn inside its window, as a `prepareStep`.
 *
 * A turn that overflows loses the whole run, not just the step. This reads the
 * context the provider measured on the previous step rather than estimating
 * it, and compacts before the next one would breach. With nothing old left to
 * stub it throws `ContextExhausted`: what fills the window is the live tail,
 * and cutting that would take away what the model is working on.
 */
export function compaction(opts: CompactionOptions = {}): ContextBuilder {
  const threshold = opts.threshold ?? 0.8;
  return {
    prepareStep(state: StepState, req: TurnRequest): { messages: ModelMessage[] } | undefined {
      const used = state.steps[state.steps.length - 1]?.usage?.inputTokens ?? 0;
      const limit = Math.floor(req.window * threshold);
      if (used < limit) return undefined;

      const { compacted, freedChars, stubbed } = compactMessages(state.messages, opts);
      if (stubbed === 0) throw new ContextExhausted(req.sessionId, used, req.window);

      opts.onCompact?.({ sessionId: req.sessionId, usedTokens: used, window: req.window, freedChars, stubbed });
      return { messages: compacted };
    },
  };
}
