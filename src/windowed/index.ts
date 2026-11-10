// The windowed context strategy. The thread is kept inside the window
// turn by turn: large tool outputs expire into a card (pruning), and the oldest
// of the conversation is dropped under pressure down to a floor (trimming).

export { pruneToolOutputs } from "./prune";
export type { PruneOptions, PruneReport } from "./prune";
export { trimWindow } from "./trim";
export type { TrimOptions, TrimReport } from "./trim";
export { windowContext } from "./windowed";
export type { WindowContextOptions } from "./windowed";
