# Gemini suite release procedure

Gemini Manifold is a coordinated release because the pipe, companion, grounding envelope, model
catalog, and feature toggles share strict contracts. Suite 3.0.0 is defined by
[`release/gemini-suite-v3.0.0.yaml`](../../release/gemini-suite-v3.0.0.yaml) and the only valid tag
is `gemini-suite/v3.0.0`.

## Release invariants

- Pipe and companion frontmatter/constants are 3.0.0 and the pipe requires companion 3.0.0.
- `google-genai` is exactly 2.11.0 in plugin metadata, `pyproject.toml`, and `uv.lock`.
- The grounding-envelope protocol is 1 and the model-catalog protocol is 3.
- The companion catalog URL is the immutable suite-tag URL, never `master`.
- Reason 2.0.0 emits `reasoning`; Maps 2.0.0 emits `google_maps`.
- The Paid, URL Context, and Enterprise toggles are present in the same bundle.
- Archived `thinking_gemini.py` is excluded.
- Unsupported custom safety configuration is absent from valves, metadata, and Interaction payloads.
- Image resolution, aspect ratio, and text/image output choices are exact catalog evidence, with
  deterministic, user-visible fallback for unsupported selections.
- The retired flat-schema YAML wizard and root `requirements.txt` are absent; `uv` and catalog
  protocol 3 are the only supported dependency and catalog paths.
- Individual v3 tags for any suite member are rejected by both the helper and release workflow.

## Preflight and deterministic build

Start from a clean branch that is up to date with its configured upstream. Then run:

```shell
make check
uv run python .github/scripts/build_gemini_suite.py \
  --manifest release/gemini-suite-v3.0.0.yaml \
  --tag gemini-suite/v3.0.0 \
  --output dist/gemini-suite-v3.0.0.tar.gz
```

Build to a second path and compare the files byte-for-byte. The workflow does this automatically,
then attaches the archive, manifest, and `SHA256SUMS` to a draft release. The manifest hashes every
member and declares install order and protocol versions.

`dev/release.sh gemini-suite` performs clean-tree/upstream checks, the canonical gate, manifest
verification, two deterministic builds, checksum generation, and an explicit confirmation before
creating/pushing the coordinated tag. It never publishes the GitHub draft or Community entries.

## Catalog policy

`plugins/pipes/gemini_models.yaml` is fail-closed policy, not a discovery cache. Every model and
service/capability claim needs current authoritative evidence and a new immutable catalog/release
hash. An override must use HTTP(S), schema 3, and immutable reviewed content. Mutable branch URLs,
uncatalogued models, and `unverified` services are denied.

The dated, machine-readable provenance trust root for catalog protocol 3 is
[`gemini-model-provenance-v1.yaml`](gemini-model-provenance-v1.yaml). It keeps exact Interactions
availability, model properties, thinking controls, pricing, and product authorization as separate
claims. The release manifest includes its byte digest, and the catalog repeats that digest so a
catalog cannot be substituted independently of its reviewed evidence.

The companion rejects duplicate and merge keys before YAML parsing, validates freshness and every
evidence reference, and publishes one atomic app-state envelope containing the full validated
catalog plus its canonical JSON digest. The Pipe independently revalidates the complete envelope
and recomputes the digest before routing. Missing, expired, unknown, mismatched, or contradictory
claims fail closed. Pricing is explicit per input/output modality and cached state; unknown document,
tool-use, or cache prices remain visibly unpriced rather than being inferred as text or zero.

The 3.0 companion URL is prospective until the coordinated tag is pushed. Before exposing traffic,
verify that the raw tagged URL returns the manifest-hashed catalog. If staging requires an override,
use an immutable commit URL with identical content and switch to the tag URL during publication.

## CI and live evidence

Normal CI runs `make check` on Linux/Python 3.11 and pytest across Linux, macOS, Windows and Python
3.11–3.13 whenever code, tests, dependencies, catalog, docs, workflow, release, or development
configuration changes.

Offline tests are authoritative for deterministic contracts. Optional Developer live smoke is
manually dispatched into a protected GitHub environment, requires `GEMINI_API_KEY`, fails rather
than skips after explicit opt-in when that secret is absent, and does not intentionally emit response
text. The Developer workflow's Enterprise job is a no-network policy contract: it proves the
canonical SDK transport snapshot, the all-unverified catalog, and pre-client denial. A separate protected
Enterprise live workflow uses Workload Identity Federation, requires explicit project/location,
model, and API-version inputs, and runs unary, SSE, and stored-continuation probes with cleanup.
Even a green probe proves only that selected tuple; promote no catalog entry until its
service-specific capabilities and pricing are audited too.

The protected environment requires reviewers/no bypass and a default-branch deployment rule. The
Google WIF provider must bind immutable repository ID plus the default ref, environment, and
audience; grant its service account only endpoint-required permissions. The workflow fails a
non-default-ref dispatch before checkout, pins actions, disables persisted checkout credentials,
validates a bare model ID, masks identity fields before authentication, converts SDK errors to
constant redacted messages, checks exactly three non-skipped cases, and records a sanitized project
hash. Retain separate `v1beta1` and `v1` run URLs in Beads before promotion.

The latest sanitized local Developer run was recorded on 2026-07-14 against the default
`gemini-2.5-flash` model: unary, SSE terminal/cleanup, and stored continuation all passed. The
test does not intentionally emit response text or identifiers and deletes the provider-stored
continuation chain.

## Publication order

1. Create/push the suite tag and let the workflow produce a draft.
2. Verify `SHA256SUMS`, manifest members, tagged catalog URL, and draft contents.
3. Stage all Open WebUI Community companion/toggle entries without publishing the pipe.
4. Publish companion and toggles, then the pipe, within one maintenance window.
5. Require a green automated Developer unary/SSE and persisted two-turn continuation run. Then
   perform operator-observed media and enabled-tool smoke checks; those two paths are not part of
   the automated credentialed suite and must not be reported as such.
6. Publish the GitHub draft only after the separately hosted Community objects match the manifest.

## Rollback

Disable all Gemini functions and restore the prior complete suite backup. Never downgrade only the
pipe or companion. Restore any prior catalog override with that suite, re-enable the old components,
and start a new chat/branch if a v3 signed replay envelope cannot be consumed. Provider-stored
Interactions are not deleted by rollback or local chat deletion: installed functions receive no
portable Open WebUI chat-deletion callback. Deletion would require a known ID created in the same
endpoint scope and would not remove local envelopes/backups or Files API objects. As documented in
Google's [Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)
(checked 2026-07-14), paid stored Interaction objects currently retain for 55 days, free objects
retain for one day, and `store=false` cannot use background execution or later previous-ID
continuation. The separate paid-project [AI Studio log setting](https://ai.google.dev/gemini-api/docs/logs-datasets)
offers 7, 14, 28, or 55 days and does not expire saved datasets. Google's
[zero-data-retention documentation](https://ai.google.dev/gemini-api/docs/zdr) also records 30-day
Search/Maps exceptions and an independent Files lifecycle until deletion or expiry.
