import type { ModelMessage } from "ai";
import { CHARS_PER_TOKEN, estimateTokens } from "../tokens";
import type { ContextBuilder, Prepared, StoredMessage, TurnRequest } from "../types";

/**
 * Where the last cut is remembered, so the window grows from `low` back up to
 * `high` between trims instead of being re-cut to `low` every turn.
 *
 * The stored history keeps growing —trimming only shapes what travels— so the
 * builder needs memory of where it cut last to measure pressure from there. The
 * value is one `idx`: the oldest stored message that still traveled. Left out,
 * the builder remembers it in a process-local map keyed by session: enough for a
 * single long-lived process, and reset on restart or across instances. An
 * ephemeral or multi-instance run wires `load`/`save` to a durable mark (the
 * harness has a place for it), and then the window grows from `low` to `high`
 * across restarts too.
 */
export type CutMemory = {
  /** The remembered cut for this session: the oldest `idx` that last traveled, or `null` the first time. */
  load: (req: TurnRequest) => number | null | Promise<number | null>;
  /** Called when the cut moves, with the oldest `idx` that travels now. */
  save: (req: TurnRequest, idx: number) => void | Promise<void>;
};

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
  /**
   * Where the last cut is remembered, so trimming happens once per climb from
   * `low` to `high` rather than every turn. Left out, a process-local map stands
   * in. See {@link CutMemory}.
   */
  cut?: CutMemory;
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
 * Pressure is measured from the last cut, not from the whole stored thread: the
 * store keeps every message, so a fresh measure from idx 0 would stay over
 * `high` forever once crossed, and re-cut to `low` every turn. Holding the cut
 * and measuring the span since it lets the conversation climb from `low` back to
 * `high` with the provider's prefix intact, and trims once at the top. The last
 * cut lives in {@link CutMemory}.
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

  // The fallback for when no durable memory is wired: the last cut per session,
  // in this process. A long-lived process gets hysteresis for free; an ephemeral
  // one wires `opts.cut` instead.
  const local = new Map<string, number>();
  const key = (req: TurnRequest) => `${req.scope.tenantId}\u0000${req.sessionId}`;
  const loadCut = async (req: TurnRequest): Promise<number | null> =>
    opts.cut ? await opts.cut.load(req) : local.get(key(req)) ?? null;
  const saveCut = async (req: TurnRequest, idx: number): Promise<void> => {
    if (opts.cut) await opts.cut.save(req, idx);
    else local.set(key(req), idx);
  };

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

      // Where the thread was cut last. Measuring pressure from here, not from 0,
      // is what lets the window climb back to `high` before trimming again.
      const lastIdx = await loadCut(req);
      let start = 0;
      if (lastIdx !== null) {
        while (start < history.length && history[start]!.idx < lastIdx) start++;
      }

      // Under `high` since the last cut: hold it, so the prefix the provider
      // cached from last turn stays whole and the conversation keeps growing.
      const held = shareAt(start);
      if (held.fifo <= held.space * high) {
        return { history: start === 0 ? history : history.slice(start), report: { ...held, dropped: start, pressure: false } };
      }

      // Over `high`: drop the oldest down to `low`, from the held cut forward and
      // never past the floor.
      const floor = await opts.floor(req);
      let cut = start;
      for (let p = start + 1; p <= history.length; p++) {
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
      if (cut !== start) {
        // The oldest idx that still travels is the new cut. If the whole stored
        // thread went, remember the one past its end so next turn holds there.
        const idx = history[cut]?.idx ?? (history[history.length - 1]?.idx ?? -1) + 1;
        await saveCut(req, idx);
      }
      return { history: cut === 0 ? history : history.slice(cut), report };
    },
  };
}
