// @nxlabscc/agent-context — the base: the abstraction and the mechanism.
//
// This entry has the `ContextBuilder` interface, its types, `compose` to chain
// builders, the per-moment registration helpers, and the token utilities any
// reaction needs to measure. It has no concrete strategy.
//
// The concrete strategies are the subpackages:
//   @nxlabscc/agent-context/linear    — a fixed context + compaction
//   @nxlabscc/agent-context/windowed  — pruning + trimming the thread

export type {
  ContextBuilder,
  OpenRequest,
  Prepared,
  Scope,
  StepState,
  StoredMessage,
  TurnBlock,
  TurnOutcome,
  TurnRequest,
} from "./types";

export { compose } from "./compose";
export { reactor } from "./reactor";
export type { Reactions } from "./reactor";
export { CHARS_PER_TOKEN, estimateTokens, tokensOfTools } from "./tokens";
