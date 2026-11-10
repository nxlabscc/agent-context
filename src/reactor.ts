import type { ModelMessage } from "ai";
import type {
  ContextBuilder,
  OpenRequest,
  Prepared,
  StepState,
  TurnBlock,
  TurnOutcome,
  TurnRequest,
} from "./types";

// A context is a reactor: it does not drive the turn, it reacts to the moments
// the runtime fires. The runtime owns the moments (open, turn, prepare messages,
// each step, after); the context only answers the ones it cares about. Whoever
// instantiates both connects them: `createRuntime({ context })`.
//
// A `ContextBuilder` is already a plain object with one optional method per
// moment, so a reactor is just that object. `reactor` builds one from the
// reactions you pass, normalizing a synchronous reaction to the async method
// the interface declares and leaving out the moments you did not give it.
// `compose` chains several reactors into one.

export type Reactions = {
  atOpen?: (req: OpenRequest) => string[] | Promise<string[]>;
  beforeTurn?: (req: TurnRequest) => TurnBlock[] | Promise<TurnBlock[]>;
  prepareMessages?: (req: TurnRequest) => Prepared | Promise<Prepared>;
  prepareStep?: (state: StepState, req: TurnRequest) => { messages: ModelMessage[] } | undefined;
  afterTurn?: (req: TurnRequest, outcome: TurnOutcome) => void | Promise<void>;
};

/** A context builder that reacts to the moments you give it, and to no others. */
export function reactor(reactions: Reactions): ContextBuilder {
  const out: ContextBuilder = {};
  if (reactions.atOpen) out.atOpen = async (req) => reactions.atOpen!(req);
  if (reactions.beforeTurn) out.beforeTurn = async (req) => reactions.beforeTurn!(req);
  if (reactions.prepareMessages) out.prepareMessages = async (req) => reactions.prepareMessages!(req);
  if (reactions.prepareStep) out.prepareStep = reactions.prepareStep;
  if (reactions.afterTurn) {
    const afterTurn = reactions.afterTurn;
    out.afterTurn = async (req, outcome) => { await afterTurn(req, outcome); };
  }
  return out;
}
