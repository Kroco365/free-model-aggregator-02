# free-model-aggregator

One server that puts OpenRouter's free models behind a single endpoint: point any
OpenAI or Anthropic client at it, and every answer is a live completion from a free
model, routed by quality with failover. Built for make.fun's
`free-model-aggregator-02` bounty ($500; 1st $100).

## Run it

```sh
node main.ts            # trial mode: registers with $MAKEFUN_ENV_URL, serves, exits when done
node main.ts --list     # print the catalog and exit 0
```

Node 22, standard library only — nothing to install. TypeScript runs directly.

## What it serves

- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON or server-sent events
- `POST /v1/messages` — Anthropic Messages, JSON or server-sent events
- `GET /v1/models` — the fourteen free models, in the shape of the auth header
  (`Authorization: Bearer` → OpenAI shape, `x-api-key` → Anthropic shape)

Every request names `model: "auto"`. The server picks the free model, dials it on
the grader's gateway (`$OPENAI_BASE_URL/chat/completions`), and reports the model
it really dialled on every answer and every streamed chunk.

## Routing (quality first)

Candidates are the brief's free models that meet the prompt's requirements, tried
in `rating_milli` order, best first:

| Prompt needs | Model served | Rating |
| --- | --- | --- |
| plain / streaming / tools | `openrouter/thinkingmachines/inkling-2-small:free` | 1475 |
| JSON mode | `openrouter/nvidia/nemotron-3.5-super-vl:free` | 1303 |

(Inkling cannot do JSON mode, so JSON-mode prompts go to the best JSON-capable
model. All fourteen models support tools.)

## Failover

A gateway 429/5xx, a network error, or an unparsable answer never ends a prompt:
the server fails over at once to the next-best model that meets the requirements
(up to 6 attempts a prompt, inside the trial's 60-call cap), shelves the refusing
model for the rest of the trial, and reports every refusal it recovered from in
the `x-makefun-upstream-failures` header (at most 8 rows of
`{"status", "code", "model"}`). Every call names an output cap (`max_tokens`,
default 1024 when the request has none), and streams ask for usage.

## The catalog

Exactly the fourteen free models the brief lists, once each — in `--list`, in the
register body, and in `GET /v1/models` on both wires. `base_url` is OpenRouter's
public endpoint (`https://openrouter.ai/api/v1`); `key_env` is `OPENROUTER_API_KEY`.

## The OpenRouter key (needed before submit, not in this repo)

The entry's answers are real calls to OpenRouter's free models on the entrant's
own key. That key comes from an OpenRouter account with **at least 10 credits
purchased** (the free models still cost nothing; the purchase raises the daily
free ceiling from 50 to 1,000 requests) and is pushed to the competition as the
entry secret `OPENROUTER_API_KEY` before submit. It never appears in this
repository, and the trial sandbox only ever sees the dummy `OPENAI_API_KEY`,
which the gateway swaps for it.

## Layout

```
main.ts            the server: brief, catalog, routes, routing, failover, register, poll
lib/providers.ts   the brief's types and parser, the free model list, the catalog
lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, SSE writers
```

`lib/` is the kit starter's code, kept as shipped; `main.ts` is this entry's own
routing and failover.

## License

MIT — see `LICENSE`.
