import { asSchema, type ToolSet } from "ai";

/**
 * Four characters per token.
 *
 * An estimate, and known to be one: the real count is the provider's and comes
 * back after the turn, too late to decide. What is measured is the JSON of what
 * is about to be sent, tool-call structure included, since that is what travels.
 */
export const CHARS_PER_TOKEN = 4;

export const estimateTokens = (chars: number, charsPerToken = CHARS_PER_TOKEN): number =>
  Math.ceil(chars / charsPerToken);

/**
 * What the tool definitions weigh.
 *
 * They are neither in the system prompt nor in the messages: they travel in
 * their own field of the request, and they are an area of the budget like any
 * other. They are measured as they travel: `asSchema` is the conversion the SDK
 * makes before sending them, so a zod schema and one from an MCP server count
 * the same.
 */
export function tokensOfTools(tools: ToolSet | undefined, charsPerToken = CHARS_PER_TOKEN): number {
  if (tools === undefined) return 0;
  let chars = 0;
  for (const [name, t] of Object.entries(tools)) {
    const def = t as { description?: string; inputSchema?: unknown };
    let schema: unknown = null;
    try {
      schema = asSchema(def.inputSchema as never).jsonSchema;
    } catch {
      // A schema that cannot be resolved counts for what is known of it. Under-
      // estimating an area beats refusing to send the turn.
    }
    chars += JSON.stringify({ name, description: def.description ?? "", schema }).length;
  }
  return estimateTokens(chars, charsPerToken);
}
