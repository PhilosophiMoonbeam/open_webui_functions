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
model as `unverified`, so the public pipe must deny Enterprise before client
creation. A separate direct-SDK probe exists to gather the missing evidence; it
does not promote a model or change production policy by itself. Run it only with
an explicitly audited model and API-version candidate:

```shell
RUN_GEMINI_ENTERPRISE_LIVE_TESTS=1 \
GEMINI_ENTERPRISE_PROJECT=... \
GEMINI_ENTERPRISE_LOCATION=... \
GEMINI_ENTERPRISE_MODEL=... \
GEMINI_ENTERPRISE_API_VERSION=v1beta1 \
uv run pytest tests/live -m 'live and enterprise_api'
```

The Enterprise suite covers unary, SSE cleanup, and semantic stored continuation using a random
nonsensitive token present only in the first turn. Cleanup captures both stored IDs before semantic
assertions, attempts every reverse-order delete independently, closes the client, and redacts
provider errors. It fails rather than skips if any required setting is absent after opt-in. A
successful run proves only the selected project/location/model/API-version tuple; catalog promotion
additionally requires service-specific capability and pricing review.
Sanitized Developer evidence was recorded on 2026-07-14 against the default
`gemini-2.5-flash` model: unary, SSE terminal/cleanup, and stored
`previous_interaction_id` continuation all passed (`3 passed`). Response text,
credentials, and Interaction IDs were not retained. The continuation test deletes
both stored provider Interactions in `finally`.

The manually dispatched `Optional Gemini Interactions Live Smoke` workflow uses
the protected `gemini-live-developer` environment. Its Developer job fails when
`GEMINI_API_KEY` is missing, so a green dispatch cannot consist only of skipped
tests. A separate no-network job proves the canonical Enterprise SDK transport
snapshot, the all-unverified catalog, and public pre-client denial. The protected
`Optional Gemini Enterprise Interactions Live Smoke` workflow uses Workload
Identity Federation, runs only from the repository default branch, pins every action by full commit
SHA, validates exactly three non-skipped cases, and requires an explicit model and API version. The
`gemini-live-enterprise` environment must have required reviewers with no bypass and a default-branch
deployment rule. Its Google provider binding must constrain immutable repository ID, default ref,
environment, and audience; the service account receives only endpoint-required permissions. Record
separate green runs for `v1beta1` and `v1` and retain their run URLs. Neither the policy job nor an
unrecorded workflow dispatch is Enterprise support evidence.
