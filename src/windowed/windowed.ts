import type { ContextBuilder } from "../types";
import { compose } from "../compose";
import { pruneToolOutputs, type PruneOptions } from "./prune";
import { trimWindow, type TrimOptions } from "./trim";

export type WindowContextOptions = {
  /** Expire large tool outputs so they stop travelling every turn. Omit to not prune. */
  prune?: PruneOptions;
  /** Drop the oldest of the thread under pressure, down to a floor. Required: it is the strategy. */
  trim: TrimOptions;
};

/**
 * The windowed strategy: the thread is kept inside the window turn by
 * turn. Large tool outputs expire into a card after a few turns (pruning), and
 * when the conversation still presses, the oldest of it is dropped down to a
 * floor (trimming). What the memory has not absorbed is the floor, so trimming
 * never drops what is not safe to lose.
 *
 * It composes `pruneToolOutputs` with `trimWindow`. The two run in that order,
 * both as `prepareMessages` reactions. A project uses either piece directly to
 * mix its own strategy.
 */
export function windowContext(opts: WindowContextOptions): ContextBuilder {
  const parts: ContextBuilder[] = [];
  if (opts.prune) parts.push(pruneToolOutputs(opts.prune));
  parts.push(trimWindow(opts.trim));
  return compose(...parts);
}
