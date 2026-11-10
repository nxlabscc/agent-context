// The context of an LLM agent's turn, as a strategy the harness calls.
//
// A harness owns the turn: it opens the session, persists the transcript, calls
// the model, runs the tools. What the model gets to read beyond the agent's
// persona is this package's: which blocks ride in the system prompt, which
// blocks come with each turn, what of the stored history travels, and how a
// long turn stays inside the window. Each of those is a moment the harness
// calls; a strategy answers the moments it cares about and leaves the rest
// undefined.
//
// Everything here is plain data the harness already has: strings, the SDK's
// `ModelMessage`, and a stored message's position in its session. The package
// knows no store, no agent definition, no provider.

import type { ModelMessage } from "ai";

/** Who the session belongs to. A single-tenant deployment passes a fixed one. */
export type Scope = { tenantId: string };

/** A message as the harness stored it: its position in the session, and the message. */
export type StoredMessage = { idx: number; message: ModelMessage };

/**
 * A block a strategy adds to a turn, placed after the user's message.
 *
 * `persist: true` stores it with the session so later turns re-send it with the
 * history; `persist: false` sends it this turn only, for what must not be
 * inherited (a voice instruction, say).
 */
export type TurnBlock = { content: string; persist: boolean };

export type OpenRequest = {
  scope: Scope;
  sessionId: string;
  agent: string;
  /** The agent's prompt as the harness composed it, before any block of this strategy. */
  systemPrompt: string;
};

export type TurnRequest = {
  scope: Scope;
  sessionId: string;
  conversationId: string | null;
  agent: string;
  /** The session's system prompt, frozen when it opened. */
  system: string;
  /** What was said this turn, as text, for whoever searches with it. */
  query: string;
  /** The stored history, the system message excluded, oldest first. */
  history: StoredMessage[];
  /** This turn's messages, never trimmed: the user's message and the blocks after it. */
  fresh: ModelMessage[];
  /** Tokens the model accepts. */
  window: number;
  /** Tokens the tool definitions take, as they travel (see `tokensOfTools`). */
  toolTokens: number;
};

export type Prepared = {
  /** The history that travels, in place of `TurnRequest.history`. */
  history: StoredMessage[];
  /** What the step did, in numbers worth logging. Keys are the step's own. */
  report: Record<string, number | boolean>;
};

/** What the SDK hands `prepareStep` before each step of a multi-step turn. */
export type StepState = {
  steps: ReadonlyArray<{ usage?: { inputTokens?: number } }>;
  messages: ModelMessage[];
};

export type TurnOutcome = {
  /** The messages the model produced this turn, as the SDK returned them. */
  messages: ModelMessage[];
  /** The stored history once this turn is persisted, the system message excluded. */
  history: StoredMessage[];
};

/**
 * A context strategy. Every moment is optional; `compose` chains several.
 *
 * - `atOpen`: blocks appended to the agent's prompt when the session opens. The
 *   harness persists the result as the session's system message and never
 *   rebuilds it: it is the prefix the provider caches.
 * - `beforeTurn`: blocks for this turn, after the user's message, so the prefix
 *   stays intact.
 * - `prepareMessages`: what of the stored history is sent. Pruning and trimming
 *   live here. The message is transformed on the way out; what is stored stays.
 * - `prepareStep`: inside a multi-step turn, before each step. Compaction lives
 *   here.
 * - `afterTurn`: once the turn is persisted. Extraction and bookkeeping live here.
 */
export interface ContextBuilder {
  atOpen?(req: OpenRequest): Promise<string[]>;
  beforeTurn?(req: TurnRequest): Promise<TurnBlock[]>;
  prepareMessages?(req: TurnRequest): Promise<Prepared>;
  prepareStep?(state: StepState, req: TurnRequest): { messages: ModelMessage[] } | undefined;
  afterTurn?(req: TurnRequest, outcome: TurnOutcome): Promise<void>;
}
