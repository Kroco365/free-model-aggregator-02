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
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  extraHeaders: Record<string, string> = {},
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
  });
}

/**
 * One prompt dialled through the gateway: the best free model that meets it, then the next
 * best on every refusal, until one answers or the candidates run out. A refusal is never an
 * answer: it is recorded in the trail, its model is shelved for the trial, and the prompt
 * moves on. `model` is null only when no candidate answered, and the failures still travel
 * with it, so even a lost prompt reports the refusals behind it.
 */
interface ServedCall {
  /** The id this entry dialled: the `model` every answer and chunk reports; null when none answered. */
  model: string | null;
  /** Every refusal this prompt met, in the order the calls refused. */
  failures: UpstreamFailure[];
  /** The gateway's answer: a stream when the client asked for one, else its whole body text. */
  upstream: Response | null;
  text: string | null;
}

async function dialWithFailover(body: Json, requires: Requirement): Promise<ServedCall> {
  const failures: UpstreamFailure[] = [];
  const candidates = rankedCandidates(requires).slice(0, MAX_ATTEMPTS_PER_PROMPT);
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, gatewayBody(body));
    } catch (err) {
      // Unreachable, timed out, or aborted: the call never reached the provider, so it is a
      // refusal of this model and the prompt moves to the next-best one.
      log(`gateway call to ${candidate.id} failed: ${(err as Error).message}`);
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      refusedModels.add(candidate.id);
      continue;
    }
    if (body.stream === true) {
      if (upstream.ok && upstream.body !== null) return { model: candidate.id, failures, upstream, text: null };
      const text = await upstream.text().catch(() => '');
      failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
      refusedModels.add(candidate.id);
      log(`gateway refused ${candidate.id} with HTTP ${upstream.status}`);
      continue;
    }
    const text = await upstream.text().catch(() => '');
    if (!upstream.ok) {
      failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
      refusedModels.add(candidate.id);
      log(`gateway refused ${candidate.id} with HTTP ${upstream.status}`);
      continue;
    }
    try {
      const parsed: unknown = JSON.parse(text);
      if (!isRecord(parsed)) throw new Error('the gateway answer is not one JSON object');
    } catch {
      failures.push({ status: 502, code: 'invalid_upstream_body', model: candidate.id });
      refusedModels.add(candidate.id);
      log(`gateway answer from ${candidate.id} is not JSON`);
      continue;
    }
    return { model: candidate.id, failures, upstream: null, text };
  }
    return { model: null, failures, upstream: null, text: null };
}

/** A prompt no candidate answered, as a 502 in the client's wire, carrying the refusal trail. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: UpstreamFailure[]): void {
  const header = failuresHeader(failures);
  const extra = header === null ? {} : { 'x-makefun-upstream-failures': header };
  const detail = failures.map((f) => `${f.model ?? 'gateway'} HTTP ${f.status} (${f.code})`).join('; ');
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `no free model answered: ${detail}`, extra);
  } else {
    sendError(res, wire, 502, 'upstream_error', `no free model answered: ${detail}`, extra);
  }
}

/** The refusal trail as response headers, for a served prompt that recovered from refusals. */
function trailHeaders(failures: readonly UpstreamFailure[]): Record<string, string> {
  const header = failuresHeader(failures);
  return header === null ? {} : { 'x-makefun-upstream-failures': header };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const served = await dialWithFailover(body, requires);
  if (served.model === null) return refused(res, 'openai', served.failures);
  if (served.upstream !== null) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trailHeaders(served.failures),
    });
    try {
      for await (const frame of openAIStreamWithModel(served.upstream.body, served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = JSON.parse(served.text ?? '{}');
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, served.model), trailHeaders(served.failures));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const translated = anthropicToOpenAI(body, 'auto');
  const served = await dialWithFailover(translated, requires);
  if (served.model === null) return refused(res, 'anthropic', served.failures);
  if (served.upstream !== null) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trailHeaders(served.failures),
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(served.upstream.body, served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = JSON.parse(served.text ?? '{}');
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, served.model), trailHeaders(served.failures));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
