import type { ModelMessage } from "ai";
import { CHARS_PER_TOKEN, estimateTokens } from "../tokens";
import type { ContextBuilder, Prepared, StoredMessage, TurnRequest } from "../types";

export type TrimOptions = {
  /** Trim once the FIFO passes this share of its space. Default 0.7. */
  high?: number;
  /**
   * How far down a trim goes. The two numbers are hysteresis: with one threshold
   * every turn crosses it again, trims again, and the cached prefix breaks every
   * time. Default 0.25.
   */
  low?: number;
  /** Still above this after trimming everything allowed: whoever feeds the floor is behind. Default 0.6. */
  alarm?: number;
  /**
   * Up to which stored `idx` the history may be dropped, `-1` for none.
   *
   * Nothing past the floor is dropped, however old: it is typically the memory
   * extraction mark, so what the memory has not absorbed stays.
   */
  floor: (req: TurnRequest) => number | Promise<number>;
  /** Told when trimming everything allowed was not enough. The place to force whatever moves the floor. */
  onPressure?: (req: TurnRequest, report: TrimReport) => void;
  charsPerToken?: number;
};

/** How the window is shared out, in estimated tokens. */
export type TrimReport = {
  /** The system prompt: the same in turn one and in turn forty. */
  fixed: number;
  /** The tool definitions, as they travel. */
  tools: number;
  /** The system messages inside the thread: memory blocks and the like. */
  memories: number;
  /** The conversation: the only thing this trims. */
  fifo: number;
  /** What is left to the FIFO once the other areas are taken out. `high` and `low` are shares of this. */
  space: number;
  /** How many stored messages were left out. */
  dropped: number;
  /** Trimming everything allowed was not enough. */
  pressure: boolean;
};

/**
 * What each message weighs, by area: `system` messages inside the thread are
 * memory, the rest is conversation. Computed once and accumulated from the end,
 * so any cut can be asked about without recomputing.
 */
function suffixes(messages: ModelMessage[]): { fifo: number[]; memories: number[] } {
  const fifo = new Array<number>(messages.length + 1).fill(0);
  const memories = new Array<number>(messages.length + 1).fill(0);
  for (let i = messages.length - 1; i >= 0; i--) {
    const weight = JSON.stringify(messages[i]).length;
    const isMemory = messages[i]!.role === "system";
    fifo[i] = fifo[i + 1]! + (isMemory ? 0 : weight);
    memories[i] = memories[i + 1]! + (isMemory ? weight : 0);
  }
  return { fifo, memories };
}

/**
 * How far back the thread that is sent reaches, as a `prepareMessages`.
 *
 * The window has areas: the system prompt, the tool definitions, the memory
 * blocks inside the thread, and the conversation. The two that move are the
 * memories and the FIFO. Trimming is a decision about the FIFO's space, which
 * is what remains after the other areas, and it drops the oldest under two
 * conditions that must both hold: there is pressure, and the floor allows it.
 *
 * The cut always falls right before a message of the person's. Between two of
 * those live a tool call and its result, and separating them sends a result
 * that answers no call, which the provider rejects.
 *
 * Messages are transformed on the way out: what is stored stays whole.
 */
export function trimWindow(opts: TrimOptions): ContextBuilder {
  const high = opts.high ?? 0.7;
  const low = opts.low ?? 0.25;
  const alarm = opts.alarm ?? 0.6;
  const cpt = opts.charsPerToken ?? CHARS_PER_TOKEN;

  return {
    async prepareMessages(req: TurnRequest): Promise<Prepared> {
      const { history, fresh, system, window, toolTokens } = req;
      const all = [...history.map((m) => m.message), ...fresh];
      const suf = suffixes(all);
      const fixed = estimateTokens(system.length, cpt);

      /** The share-out if the cut fell at `from`. The FIFO's space depends on how much memory stays. */
      const shareAt = (from: number): Omit<TrimReport, "dropped" | "pressure"> => {
        const memories = estimateTokens(suf.memories[from]!, cpt);
        return {
          fixed,
          tools: toolTokens,
          memories,
          fifo: estimateTokens(suf.fifo[from]!, cpt),
          space: Math.max(0, window - fixed - toolTokens - memories),
        };
      };

      const whole = shareAt(0);
      if (whole.fifo <= whole.space * high) {
        return { history, report: { ...whole, dropped: 0, pressure: false } };
      }

      const floor = await opts.floor(req);
      let cut = 0;
      for (let p = 1; p <= history.length; p++) {
        if (all[p]?.role !== "user") continue;
        // Stored idx are in order, so the last message that would be left out decides.
        if (history[p - 1]!.idx > floor) break;
        cut = p;
        const share = shareAt(p);
        if (share.fifo <= share.space * low) break;
      }

      const share = shareAt(cut);
      const report: TrimReport = {
        ...share,
        dropped: cut,
        pressure: share.fifo > share.space * alarm,
      };
      if (report.pressure) opts.onPressure?.(req, report);
      return { history: cut === 0 ? history : history.slice(cut), report };
    },
  };
}
