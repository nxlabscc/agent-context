import type { ContextBuilder, OpenRequest, Prepared, StepState, TurnBlock, TurnOutcome, TurnRequest } from "./types";
import type { ModelMessage } from "ai";

/**
 * One strategy out of several, in order.
 *
 * Blocks concatenate. `prepareMessages` chains: each step receives the history
 * the one before it left, and the reports merge (a later step's key wins).
 * `prepareStep` chains the same way over the step's messages. `afterTurn` runs
 * every step, in order, and stops at the first that throws.
 */
export function compose(...builders: ContextBuilder[]): ContextBuilder {
  return {
    async atOpen(req: OpenRequest): Promise<string[]> {
      const out: string[] = [];
      for (const b of builders) if (b.atOpen) out.push(...(await b.atOpen(req)));
      return out;
    },

    async beforeTurn(req: TurnRequest): Promise<TurnBlock[]> {
      const out: TurnBlock[] = [];
      for (const b of builders) if (b.beforeTurn) out.push(...(await b.beforeTurn(req)));
      return out;
    },

    async prepareMessages(req: TurnRequest): Promise<Prepared> {
      let history = req.history;
      const report: Prepared["report"] = {};
      for (const b of builders) {
        if (!b.prepareMessages) continue;
        const prepared = await b.prepareMessages({ ...req, history });
        history = prepared.history;
        Object.assign(report, prepared.report);
      }
      return { history, report };
    },

    prepareStep(state: StepState, req: TurnRequest): { messages: ModelMessage[] } | undefined {
      let messages = state.messages;
      let changed = false;
      for (const b of builders) {
        if (!b.prepareStep) continue;
        const result = b.prepareStep({ ...state, messages }, req);
        if (result === undefined) continue;
        messages = result.messages;
        changed = true;
      }
      return changed ? { messages } : undefined;
    },

    async afterTurn(req: TurnRequest, outcome: TurnOutcome): Promise<void> {
      for (const b of builders) if (b.afterTurn) await b.afterTurn(req, outcome);
    },
  };
}
