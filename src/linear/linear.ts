import type { ContextBuilder, OpenRequest } from "../types";
import { compose } from "../compose";
import { reactor } from "../reactor";
import { compaction, type CompactionOptions } from "./compaction";

export type LinearContextOptions = {
  /** Blocks for the system prompt at session open: a state file, a clock, a workspace shape. */
  open?: (req: OpenRequest) => string[] | Promise<string[]>;
  /** How a long turn is kept inside the window. Omit to not compact. */
  compaction?: CompactionOptions;
};

/**
 * The linear strategy: the context is a fixed set of blocks decided
 * when the session opens, and a long turn is kept inside the window by compacting
 * older tool results in place. Nothing is recalled per turn.
 *
 * It composes a reactor for the open blocks with `compaction` (the reaction at
 * each step). A project that wants only one of the two uses that piece directly.
 */
export function linearContext(opts: LinearContextOptions = {}): ContextBuilder {
  const parts: ContextBuilder[] = [];
  if (opts.open) parts.push(reactor({ atOpen: opts.open }));
  if (opts.compaction) parts.push(compaction(opts.compaction));
  return compose(...parts);
}
