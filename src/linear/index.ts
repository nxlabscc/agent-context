// The linear context strategy. The context is a fixed set of
// blocks set at session open; a long turn stays inside the window by compacting
// older tool results in place.

export { ContextExhausted, compactMessages, compaction } from "./compaction";
export type { CompactionOptions, CompactionReport } from "./compaction";
export { linearContext } from "./linear";
export type { LinearContextOptions } from "./linear";
