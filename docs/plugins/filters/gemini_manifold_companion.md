# Gemini Manifold Companion 3.0.0

The companion is the required request/response filter for Gemini Manifold 3.0.0. It translates
Open WebUI controls into the pipe's canonical Interactions feature contract, handles raw-document
handoff, and renders durable grounding citations and sources.

It is not compatible with pipe 2.x. Install companion 3.0.0 before pipe 3.0.0 from the same
`gemini-suite/v3.0.0` manifest.

## Request behavior

For models owned by `gemini_manifold_google_genai`, the inlet:

- Loads and validates the pinned schema-2 service-qualified model catalog for the pipe.
- Records companion version 3.0.0 and canonical chat controls in request metadata.
- Converts enabled Open WebUI web search to the neutral desired-feature signal
  `features.google_search_tool`; the pipe authorizes it against the selected service policy.
- Converts enabled code interpreter to `features.google_code_execution` under the same
  pipe-owned authorization rule.
- With `BYPASS_BACKEND_RAG`, removes Open WebUI RAG files and asks the pipe to process original
  documents. Temporary/local chats cannot use this bypass and fall back to Open WebUI RAG.

The companion does not set temperature, implement dynamic-retrieval thresholds, or use hard-coded
allowed-model lists. Those GenerateContent-era options were removed; the versioned catalog is the
only capability authority, and the pipe applies it after selecting the request's service.

## Durable grounding protocol

The pipe persists a project-owned `gemini_interaction.grounding` envelope with protocol version 1.
It contains neutral text-block coordinates, source records, citations, tool summaries, warnings,
and a SHA-256 digest of the exact visible assistant response. It contains no Google SDK object and
no opaque replay signature.

At outlet time the companion reads the envelope from the assistant message, with a chat-database
lookup fallback. It mutates the response only when the visible-content digest still matches. An
edited or already-cited response is left untouched, preventing duplicate markers and stale source
events. Provider-offset citations on non-ASCII text fail closed unless an explicit byte/codepoint
unit is known.

Search, URL Context, Maps, and file sources are emitted from the neutral envelope. Only the known
Vertex redirect host is resolved; other URLs remain as recorded. Callback failures are isolated,
and all request-local event queues are flushed and shut down before outlet completion.

## Valves

| Valve | Purpose |
| --- | --- |
| `BYPASS_BACKEND_RAG` | Let the pipe process original persistent-chat documents. |
| `MODEL_CONFIG_PATH` | Immutable HTTP(S) schema-2 catalog. Defaults to the suite tag, never `master`. |
| `URL_RESOLVE_TIMEOUT` | Per-attempt redirect timeout in seconds. |
| `URL_RESOLVE_MAX_RETRIES` | Additional redirect attempts. |
| `URL_RESOLVE_BASE_DELAY` | Exponential retry base delay. |
| `STATUS_EMISSION_BEHAVIOR` | `disable`, hidden compact/detailed, or visible/timed status mode. |
| `LOG_LEVEL` | Filter logging verbosity. |

The default catalog URL is prospective until `gemini-suite/v3.0.0` exists. Operators may stage an
immutable reviewed URL with identical schema/content, but must switch to the tagged catalog as part
of release publication. Mutable branch overrides are unsupported.

## Troubleshooting

- A catalog load/protocol error blocks the suite by design; verify the immutable URL and manifest
  hash.
- Missing citations commonly mean the response was edited, the digest differed, or an annotation
  offset could not be applied safely.
- RAG bypass requires a persisted chat and an attachment still available in Open WebUI storage.
- Search/code controls are ignored for non-Manifold models and denied for unsupported catalog
  entries.

See the [pipe guide](../pipes/gemini_manifold.md) and
[release guide](../../development/gemini-suite-release.md).

## License

[MIT](../../../LICENSE)
