# Gemini Manifold 3.0.0

`gemini_manifold.py` exposes catalog-approved Gemini models in Open WebUI through the
`google-genai==2.11.0` Interactions API. Version 3.0 is a strict cutover: no maintained
GenerateContent request, response, storage, or compatibility path remains.

Authoritative references: [Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview),
[Google Gen AI SDK 2.11.0 release](https://github.com/googleapis/python-genai/releases/tag/v2.11.0),
[function calling](https://ai.google.dev/gemini-api/docs/function-calling), and
[model catalog source](../../../plugins/pipes/gemini_models.yaml).

## Required coordinated components

Install pipe 3.0.0 with companion 3.0.0 and catalog protocol 3 from the same
`gemini-suite/v3.0.0` manifest. The pipe rejects a missing/wrong companion version and a missing or
wrong catalog protocol. The Reason and Maps filters have breaking 2.0.0 feature keys; the old
Vertex AI toggle is replaced by the Enterprise toggle.

The companion's default catalog URL is pinned to:

```text
https://raw.githubusercontent.com/suurt8ll/open_webui_functions/gemini-suite/v3.0.0/plugins/pipes/gemini_models.yaml
```

That prospective URL works only after the tag exists. Do not enable the release before the tag
and catalog are reachable. An override must be immutable, reviewable, schema 3, and compatible
with both 3.0 components; mutable branch URLs are unsupported.

## Service support

- **Gemini Developer API:** deterministically validated `v1` Interactions route. Free and paid API
  keys use the same protocol but separate credentials/routing policy.
- **Gemini Enterprise:** the SDK's tested route is `v1beta1` with project/location identity.
  However, every bundled catalog entry currently has
  `services.enterprise.availability: unverified`; public-pipe generation is denied before network
  access. Configuration or a successful models-list call is not evidence of Interactions model
  support.
- **Custom base URL:** applies to Developer API and is included in endpoint identity. Interaction
  IDs and signed replay ledgers never cross service, credential, base-URL, API-version, or model
  scope.

Unknown or unverified models and capabilities fail closed. `MODEL_WHITELIST` and
`MODEL_BLACKLIST` can narrow the catalog but cannot grant unsupported capabilities.

## Inputs, outputs, and tools

The exact per-model, per-service catalog controls text, image, audio, video, document,
external-URL, Files API, storage, pricing, thinking, response-format, image-output, requested
maximum output, and tool availability. It records the provider input-context limit, while the
provider remains authoritative for actual input token accounting. Unverified/unsupported service
nodes cannot contain capabilities or pricing. Supported request construction
includes multimodal user content, Open WebUI files, YouTube URLs, PDFs, system instructions, JSON
schema response format, and reasoning summaries. The Interactions API does not support custom safety
configuration, so the suite exposes no valve, metadata field, request option, or payload key for it.
Automatic function calling (AFC), explicit cached-content input, Batch API configuration, and video
metadata are likewise outside this Interactions product contract and are rejected before provider
access. Custom Open WebUI functions use the explicit bounded loop described below, never SDK AFC.

Requestable server tools are Google Search, URL Context, Google Maps, and code execution. The
reducer also recognizes file-search results, but no bundled service policy enables file-search
requests. The companion translates Open WebUI search/code controls into neutral desired-feature
signals; the optional toggles emit canonical `reasoning`, `google_maps`, and `url_context` feature
flags. The pipe authorizes every requested feature against the selected service policy before an
API call.

Custom Open WebUI functions are deliberately narrower:

- With effective `store=true`, rounds continue through the same-scope interaction ID. With
  `store=false`, every round sends the complete validated signed Step ledger plus ordered function
  results and never sends `previous_interaction_id`. Unary requests use unary Interaction rounds;
  streaming requests use SSE rounds with indexed `arguments_delta` assembly.
- Only authorized request-local callables with object JSON schemas are exposed.
- Direct frontend tools are rejected.
- Calls from one provider round execute concurrently, while submitted results retain provider
  step order. Cancellation cleans up every in-flight callable before propagating.
- The loop allows at most 8 rounds, 16 calls per round, and 32 calls total.
- Each call has a 30-second timeout and each serialized result is limited to 1 MiB.
- Repeated call IDs must have identical names/arguments; results are reused rather than executed
  twice.

### Image generation controls

For image-output models, `IMAGE_OUTPUT_FORMAT` selects `Images & Text` (the default) or
`Images only`. Resolution and aspect ratio are validated against exact model evidence from
Google's [image generation documentation](https://ai.google.dev/gemini-api/docs/image-generation)
(checked 2026-07-14), never inferred from a model family:

| Model | Resolutions | Aspect ratios |
| --- | --- | --- |
| `gemini-3-pro-image` | `1K`, `2K`, `4K` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| `gemini-3.1-flash-image` | `512`, `1K`, `2K`, `4K` | `1:1`, `1:4`, `1:8`, `2:3`, `3:2`, `3:4`, `4:1`, `4:3`, `4:5`, `5:4`, `8:1`, `9:16`, `16:9`, `21:9` |

If a selected resolution is unsupported, the request omits it and lets the provider choose its
default. If an aspect ratio is unsupported, the pipe chooses the closest supported ratio
deterministically. Both adjustments emit a status event so the effective request is visible.

## Continuation, storage, and privacy

`STORE_INTERACTIONS` controls future provider-side Interaction storage. Effective storage is
monotonic: either the administrator or user can opt out, while a user cannot override an
administrator opt-out. The pipe explicitly sends `store` on every request; temporary/local chats
and Open WebUI task requests always send `store=false`. It never requests provider background
execution. Per Google's [Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)
(checked 2026-07-14), the provider default is `store=true`; stored Interactions are retained for
55 days on paid projects and one day on free projects. Separately, the paid-project
[AI Studio log setting](https://ai.google.dev/gemini-api/docs/logs-datasets) defaults to 55 days
and offers 7, 14, 28, or 55 days; saved datasets do not expire with that log window. The valves
permit or prevent future Interaction storage; they do not configure either retention setting or
erase an existing object.

For a persisted, unedited branch with matching endpoint identity, model, and stored completed
Interaction, the next request may use `previous_interaction_id`. Both the parent and continuation
request use `store=true`, as required by the service. When effective storage is false, the pipe
uses exact signed replay instead of a previous ID, including custom-function rounds in temporary,
task and privacy-opt-out requests. `store=false` cannot be combined with provider background
execution and cannot create an ID usable by a later `previous_interaction_id` request. If Google
returns not-found for an expired or deleted previous ID on the first continuation request, the
pipe retries once with the exact local replay input and current request configuration.

Every assistant message also receives one local `gemini_interaction` envelope containing exact
step payloads, status, usage, visible-content digest, grounding protocol, and endpoint-scope
fingerprint. It contains no credentials, but it can contain signed thought/tool replay data and
conversation content. Treat the Open WebUI database and backups as sensitive. Do not expose,
hand-edit, merge, or transplant envelopes between chats or credentials.

Stateless replay is selected for storage opt-out, temporary chats, edited history, branches,
foreign endpoint scope, missing/expired state, and other unsafe continuation cases. Replay is
exact only for supported 2.11 step/content variants. Unknown variants fail closed. Recovery is
not transactional across Open WebUI workers: duplicate suppression is process-local, and a crash
between provider completion and local persistence can require regeneration.

Changing model or service never reuses the prior Interaction ID. Local replay checks every prior
model-output modality against the selected target's catalogued input modalities and fails before
the provider request if any are incompatible. A provider Files URI is never replayed across a
different endpoint identity. Local Open WebUI media references are reloaded through the Open WebUI
Files API only after an owner-filtered lookup for the authenticated requesting user. Missing,
foreign, or transplanted file IDs fail before storage access or a Gemini request; administrator
role does not bypass this ownership rule. Do not transplant envelopes or file IDs between users,
chats, or backups.

Deleting an Open WebUI chat does not invoke provider deletion, and this pipe cannot promise
provider erasure because Open WebUI exposes no portable chat-deletion callback to installed
functions. A safe future integration could only best-effort delete a known ID created in the
matching active endpoint scope; it must never accept an arbitrary or transplanted envelope ID.
Provider deletion would not delete the local envelope, backups, or Files API objects. Those have
independent lifecycles.

`store=false` is not a general zero-data-retention promise. Google's
[zero-data-retention documentation](https://ai.google.dev/gemini-api/docs/zdr) states that Search-
and Maps-grounded data can be retained for 30 days. Files API objects have an independent lifecycle
until explicit deletion or provider expiry. Disable incompatible tools and `USE_FILES_API`, and
manage local Open WebUI records/backups, when those lifecycles are unsuitable.

## Reasoning, statuses, and failures

Reasoning uses Interactions `thinking_level` (`minimal`, `low`, `medium`, `high`) and optional
automatic summaries. Opaque signatures are retained only in the replay ledger and never emitted
through grounding/source events.

The reducer handles `in_progress`, `requires_action`, `completed`, `failed`, `cancelled`,
`incomplete`, and `budget_exceeded`. Function calls require `requires_action`; normal output must
finish `completed`. Error events, conflicting timelines, unknown open-union variants, nonterminal
stream endings, and failed terminal statuses surface as request failures rather than partial
success.

## Files and PDF mitigation

`USE_FILES_API` enables content-addressed Developer Files API uploads; Enterprise and temporary
chat paths use inline/local processing where supported. `PDF_LIMIT_MITIGATION` optimizes and
splits PDFs above 50 MiB or 1000 pages into ordered pieces with an original-page map. A single
page that still exceeds the limit fails rather than being silently rasterized. Temporary outputs
are cached for several hours in the process/system temp directory.

Uploaded files, provider interactions, and the local signed envelope have distinct lifecycles.
Disabling Interaction storage does not by itself disable Files API upload; disable `USE_FILES_API`
too when provider file persistence is unsuitable.

## Main valves

| Valve | Purpose |
| --- | --- |
| `GEMINI_FREE_API_KEY`, `GEMINI_PAID_API_KEY` | Separate Developer credentials; never put them in catalog overrides or logs. |
| `USE_ENTERPRISE`, `ENTERPRISE_PROJECT`, `ENTERPRISE_LOCATION` | Enterprise routing request; catalog policy can still deny it. |
| `GEMINI_API_BASE_URL` | Optional Developer endpoint override and part of continuation scope. |
| `ENABLE_FREE_TIER_FALLBACK` | Retry eligible free 429/503 failures on the paid credential. |
| `TASK_MODEL_ROUTING` | `only_free`, `free_fallback`, `only_paid`, or `match_main`. |
| `STORE_INTERACTIONS` | Provider storage/continuation permission for persisted chats. |
| `THINKING_LEVEL`, `THINKING_SUMMARIES` | Reasoning level and summary policy. |
| `USE_FILES_API`, `PDF_LIMIT_MITIGATION`, `PARSE_YOUTUBE_URLS` | Media ingestion policy. |
| `MAPS_GROUNDING_COORDINATES` | Optional `latitude,longitude` Maps bias. |
| `MODEL_WHITELIST`, `MODEL_BLACKLIST`, `CACHE_MODELS` | Catalog visibility controls. |

User valves can narrow storage and supply their own credentials. `USER_MUST_PROVIDE_AUTH_CONFIG`
forces non-whitelisted users onto their own Developer credentials and disables shared Enterprise
configuration.

### Removed 2.x options

The cutover removes GenerateContent-specific thinking budgets, dynamic search thresholds,
temperature-for-grounding overrides, `ENABLE_URL_CONTEXT_TOOL`, and legacy retrieval toggles.
`GEMINI_API_KEY` is replaced by explicit free/paid credential fields, reasoning uses levels and
summaries, and feature availability comes only from the catalog plus canonical toggle flags. No
old valve name is accepted as an alias.

## Smoke validation and troubleshooting

Run the offline gate first:

```shell
make check
```

Optional credential-gated, output-redacted smoke commands are in
[tests/live/README.md](../../../tests/live/README.md). Developer unary/SSE and stored continuation
are available. A separate direct-SDK Enterprise probe covers unary, SSE, and stored continuation
for an explicitly selected project/location/model/API-version tuple, but no successful Enterprise
run is currently recorded and the public pipe remains fail-closed while every catalog entry is
`unverified`. Normal CI runs only the non-network canonical-transport and pre-client-denial policy
contract.

### Process shutdown

The pipe owns cached asynchronous SDK clients and in-memory file/model caches. Open WebUI does not
currently expose a portable shutdown hook for installed functions. Hosts that embed the pipe must
stop accepting requests, wait for active requests to finish, and then call `await pipe.shutdown()`.
The call is idempotent, closes cached clients once, and clears pipe-owned caches. Calling it while a
request is active fails explicitly instead of closing resources underneath that request.

Common startup failures are intentional safety checks: wrong companion version, unreachable or
invalid catalog, catalog protocol mismatch, uncatalogued model, unsupported capability, or an
Enterprise-unverified model. Do not work around them by using a mutable catalog.

See the [coordinated release guide](../../development/gemini-suite-release.md) for upgrade and
rollback procedures.

## License

[MIT](../../../LICENSE)
