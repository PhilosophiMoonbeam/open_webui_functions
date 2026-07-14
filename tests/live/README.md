# Optional Gemini Interactions live smoke tests

Live tests are disabled by default and never record response text, credentials,
project identifiers, or uploaded content.

Run the Developer API smoke tests with:

```shell
RUN_GEMINI_LIVE_TESTS=1 GEMINI_API_KEY=... uv run pytest tests/live -m 'live and developer_api'
```

`GEMINI_LIVE_MODEL` may override the default `gemini-2.5-flash`. The smoke
tests cover unary, SSE cleanup, and a stored previous-Interaction second turn.

The audited model catalog currently classifies every Enterprise Interactions
model as `unverified`, so this directory contains no Enterprise generation test.
The public pipe must deny Enterprise before client creation until a model is
promoted through catalog review and receives its own credential-backed test.
Sanitized Developer evidence was recorded on 2026-07-14 against the default
`gemini-2.5-flash` model: unary, SSE terminal/cleanup, and stored
`previous_interaction_id` continuation all passed (`3 passed`). Response text,
credentials, and Interaction IDs were not retained. The continuation test deletes
both stored provider Interactions in `finally`.

The manually dispatched `Optional Gemini Interactions Live Smoke` workflow uses
the protected `gemini-live-developer` environment. Its Developer job fails when
`GEMINI_API_KEY` is missing, so a green dispatch cannot consist only of skipped
tests. A separate no-network job proves the canonical Enterprise SDK transport
snapshot, the all-unverified catalog, and public pre-client denial. That policy
contract is not Enterprise endpoint or model-support evidence.
