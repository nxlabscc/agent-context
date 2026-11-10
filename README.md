# @nxlabscc/agent-context

What an LLM agent's context holds beyond its persona, as a strategy a harness calls. For Bun, in
TypeScript source.

A harness owns the turn: it opens the session, persists the transcript, calls the model, runs the
tools. This package owns what the model gets to read beyond the agent's prompt, at five moments the
harness calls. Compaction, pruning and trimming are reactions to those moments; a concrete strategy
is a set of reactions, composed.

The package is in two layers:

- **The base** (`@nxlabscc/agent-context`) is the abstraction and the mechanism: the `ContextBuilder`
  interface and its types, `compose` to chain builders, the per-moment registration helpers, and the
  token utilities any reaction needs to measure. No concrete strategy.
- **Two style subpackages** carry the concrete strategies: `@nxlabscc/agent-context/linear` (a fixed
  context plus compaction) and `@nxlabscc/agent-context/windowed` (pruning plus trimming
  the thread). A project imports the style it wants, or the individual reactions to mix its
  own.

Everything crosses the boundary as plain data the harness already has: strings, the `ai` SDK's
`ModelMessage`, and a stored message's position in its session.

## Install

```json
"dependencies": {
  "@nxlabscc/agent-context": "github:nxlabscc/agent-context#v0.1.0"
}
```

Bun runs the `.ts` files as they are; there is no build step. `ai` is a peer dependency.

## The interface

```ts
interface ContextBuilder {
  atOpen?(req: OpenRequest): Promise<string[]>;
  beforeTurn?(req: TurnRequest): Promise<TurnBlock[]>;
  prepareMessages?(req: TurnRequest): Promise<{ history: StoredMessage[]; report: Record<string, number | boolean> }>;
  prepareStep?(state: StepState, req: TurnRequest): { messages: ModelMessage[] } | undefined;
  afterTurn?(req: TurnRequest, outcome: TurnOutcome): Promise<void>;
}
```

| moment | when the harness calls it | what the answer does |
|---|---|---|
| `atOpen` | once, when the session opens | blocks appended to the agent's prompt; persisted as the system message and never rebuilt, since it is the prefix the provider caches |
| `beforeTurn` | every turn, before the model call | blocks placed after the user's message, so the prefix stays intact; `persist: false` for a block this call must not pass on |
| `prepareMessages` | every turn, after `beforeTurn` | which of the stored messages travel, and in what shape; what is stored stays whole |
| `prepareStep` | inside a multi-step turn, before each step | the SDK's `prepareStep`: the messages of the next step |
| `afterTurn` | once the turn is persisted | bookkeeping: extraction, marks |

`TurnRequest` carries the scope, the session, the frozen system prompt, the turn's text (`query`),
the stored `history` (each message with its `idx`), this turn's `fresh` messages (never trimmed),
the model's `window` in tokens and what the tool definitions weigh (`toolTokens`, from
`tokensOfTools`).

## The base: abstraction and mechanism

```ts
import { compose, reactor, tokensOfTools, estimateTokens } from "@nxlabscc/agent-context";
```

The runtime owns the moments and fires them turn by turn; a context is a reactor that answers the
ones it cares about and never drives the loop. A `ContextBuilder` is a plain object with one optional
method per moment (`atOpen`, `beforeTurn`, `prepareMessages`, `prepareStep`, `afterTurn`), so a
reactor is just that object. `reactor({ ... })` builds one from the reactions you pass, normalizing a
synchronous reaction to the async method the interface declares and leaving out the moments you did
not give it. `compose(...builders)` chains several reactors into one. `tokensOfTools(tools)` and
`estimateTokens(chars)` measure the budget, four characters per token.

```ts
const clock = reactor({ atOpen: () => ["<clock>12:00</clock>"] });
const voice = reactor({ beforeTurn: () => [{ content: voiceBlock, persist: false }] });
const context = compose(clock, voice);   // handed whole to createRuntime({ context })
```

## `@nxlabscc/agent-context/linear` — the linear strategy

```ts
import { linearContext, compaction, compactMessages, ContextExhausted } from "@nxlabscc/agent-context/linear";
```

The context is a fixed set of blocks set at session open; a long turn stays inside the window by
compacting older tool results in place. `linearContext({ open, compaction })` composes the two.

- **`compaction({ threshold = 0.8, keepRecent = 12, minStubChars = 400, stub?, onCompact? })`** — a
  `prepareStep` for long turns. Reads the context the provider measured on the previous step; past
  the threshold it replaces the payload of tool results older than the last `keepRecent` messages
  with a stub, leaving every message, call and id in place. With nothing left to stub it throws
  `ContextExhausted`, so the harness ends the turn through its own failure path instead of the
  provider's. `compactMessages(messages, opts)` is the pure step.

## `@nxlabscc/agent-context/windowed` — the windowed strategy

```ts
import { windowContext, pruneToolOutputs, trimWindow } from "@nxlabscc/agent-context/windowed";
```

The thread is kept inside the window turn by turn: large tool outputs expire into a card (pruning),
and the oldest of the conversation is dropped under pressure down to a floor (trimming).
`windowContext({ prune, trim })` composes the two, prune first.

- **`pruneToolOutputs({ maxChars = 2000, ttl: { default = 10, byTool }, note? })`** — a
  `prepareMessages` for long conversations. A tool output past `maxChars` lives whole for a number
  of the person's turns and then travels as a card: `{ kind: "pruned", tool, chars, note,
  how_to_recover }`, where `how_to_recover` is the call written as it is made again. The card takes
  the shape the output had (text, json, or one typed part at a time; media parts are never touched),
  and an output expires once, so the provider's prefix cache breaks once.
- **`trimWindow({ floor, high = 0.7, low = 0.25, alarm = 0.6, onPressure? })`** — a
  `prepareMessages` that decides how far back the thread reaches. The window has areas: the system
  prompt, the tool definitions, the `system` messages inside the thread (memory blocks), and the
  conversation; `high` and `low` are shares of what the conversation has left. It drops the oldest
  messages when two conditions hold: the conversation passed `high`, and the message's `idx` is at
  or below `floor(req)` (typically the memory extraction mark). It cuts right before a message of
  the person's, never between a tool call and its result, and goes down to `low`. Still above
  `alarm` after dropping everything allowed, it calls `onPressure`, the place to force whatever
  moves the floor. The report says what each area weighs.
A harness composes its own strategy from the pieces: `compose(reactor({ atOpen: stateAndClock }),
compaction())` is the linear one by hand, and `windowContext({ prune, trim })` is the windowed one
ready-made. A project's memory recall is one more reactor it composes in.

## Develop

Nothing is installed on the host: Bun runs inside the shared NxLabs toolchain image.

```sh
docker run --rm -v "$PWD":/workspace -w /workspace ghcr.io/nxlabscc/bun-builder:1.4 \
  bash -c "bun install --frozen-lockfile && bun run typecheck && bun run test"
```

`bun install` without `--frozen-lockfile` is how `bun.lock` gets created or updated, locally.

## License

Apache-2.0
