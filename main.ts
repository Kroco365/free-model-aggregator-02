// free-model-aggregator-02: one server, two client wires, and a real free model behind
// every answer.
//
// This entry builds on the kit starter's shape (brief parsing in lib/providers.ts,
// Anthropic <-> Chat Completions translation in lib/wire.ts) and replaces its routing
// and failover with a competitive one:
//
//   - quality (60%): every prompt is served by the highest-rated free model that meets
//     its requirements (rating_milli from the brief): Inkling 2 Small (1475) for plain
//     and tools prompts, Nemotron 3.5 Super VL (1303) for JSON-mode prompts, which
//     Inkling cannot do. Candidates are tried in rating order, brief order breaking
//     ties, so the first success is always the best model still available.
//   - failover (10%): a gateway 429/5xx, a network error, or an unparsable answer never
//     ends a prompt. The server fails over to the next-best model that meets the
//     requirements, and reports every refusal it recovered from in the
//     `x-makefun-upstream-failures` header, so the failover column can see it. A model
//     that refuses is excluded from the rest of the trial: a rate-limited model is not
//     re-dialled for a later prompt.
//   - catalog (5%): the catalog is exactly the brief's fourteen free ids, once each,
//     in the register body, in `--list`, and in GET /v1/models on both auth headers.
//   - determinism: every choice is a function of the brief, the requests, and the
//     gateway's answers. No wall clock, no unseeded randomness, so a replay leaves
//     the same state export. The refused-model set is trial state, not timing.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with
//   Node 22's standard library alone (nothing is installed; TypeScript runs directly,
//   and a relative import needs its .ts extension). The env is at $MAKEFUN_ENV_URL;
//   model calls go through the grader's gateway at $OPENAI_BASE_URL, an
//   OpenAI-compatible endpoint that reaches OpenRouter on the key pushed to the
//   competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points,
//       what it can do (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it
//       streams, its weight (1 or 2) and the requirements a model must meet to serve
//       it. Each prompt carries a unique token the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
//        with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually dialled.
//   The house records every gateway call and scores the model that really answered. This server sends
//   the brief's model id in the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
//   its claim is the call it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// THE GATEWAY RULES THIS SERVER OBEYS
//   - Every model call names a free id from the brief, as `openrouter/<vendor>/<model>:free`.
//   - Every call names an output cap: `max_tokens` (or `max_completion_tokens`). A call with no
//     cap is refused, so a request that arrives without one gets a 1024-token default.
//   - A trial may make at most 60 model calls; this server makes at most 6 attempts a prompt.
//   - A 429 or 5xx comes back as the provider sent it, with no `retry-after` header. This server
//     never waits on one: it fails over at once to the next-best model.
//
// THE OPENROUTER KEY
//   The entry needs an OpenRouter API key from an account with at least 10 credits purchased,
//   pushed to this competition before submit (the free models still cost nothing; the purchase
//   raises the daily free ceiling from 50 to 1,000 requests). The key never appears in this
//   repository: it is pushed as the entry secret OPENROUTER_API_KEY, and the sandbox only ever
//   sees the trial's dummy OPENAI_API_KEY, which the gateway swaps for it.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most dial attempts one prompt costs, across failovers. The trial caps at 60 calls. */
const MAX_ATTEMPTS_PER_PROMPT = 6;
/** One gateway call may take this long before it counts as refused and fails over. */
const ATTEMPT_TIMEOUT_MS = 25000;
/** The output cap a request without one is given: the gateway refuses an uncapped call. */
const DEFAULT_MAX_TOKENS = 1024;
/** The failure trail header is at most 8 rows and 8 KiB; longer trails read as no trail. */
const MAX_TRAIL_ROWS = 8;
const MAX_TRAIL_BYTES = 8 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

/**
 * The free models that refused a call in this trial, by gateway id. A model that answers 429 or
 * 5xx is rate-limited or down for the moment, so it is never dialled again in this trial: the
 * next prompt starts below it in the ranking. This set is a function of the gateway's answers
 * only, never of the clock, so a replay makes the same choices.
 */
const refusedModels = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve a request, best first: rating_milli descending, the brief's
 * order breaking ties (Array.sort is stable). A refused model is out for the rest of the trial.
 * Quality is the rating of the model that really serves the prompt over the best rating that
 * meets it, so the first candidate that answers is always the best still available.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => !refusedModels.has(model.id) && meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/** One refused gateway call, as the `x-makefun-upstream-failures` trail reports it. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** A refused call's error code: the gateway's own, else a name for its status. */
function failureCode(status: number, bodyText: string): string {
  try {
    const parsed: unknown = JSON.parse(bodyText);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
  } catch {
    // Not JSON: fall through to the status name.
  }
  if (status === 429) return 'rate_limited';
  if (status === 400) return 'invalid_request_error';
  if (status === 401) return 'authentication_error';
  return 'upstream_error';
}

/**
 * The refusal trail for a response header: at most 8 rows, at most 8 KiB, or null (a missing,
 * malformed, or over-long trail reads as no trail at all). The trail is the only way the
 * failover column can see a refusal this server retried and then served.
 */
function failuresHeader(failures: readonly UpstreamFailure[]): string | null {
  if (failures.length === 0) return null;
  const text = JSON.stringify(
    failures.slice(0, MAX_TRAIL_ROWS).map((f) => ({ status: f.status, code: f.code, model: f.model })),
  );
  if (Buffer.byteLength(text) > MAX_TRAIL_BYTES) return null;
  return text;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback
  // is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** A Chat Completions body the gateway accepts: an output cap, and usage on a stream. */
function gatewayBody(body: Json): Json {
  const out: Json = { ...body };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') {
    out.max_tokens = DEFAULT_MAX_TOKENS;
  }
  if (out.stream === true) {
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
