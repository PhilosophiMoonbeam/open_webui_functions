# Gemini suite 3.0.0

This is a coordinated breaking release of the Gemini Interactions suite. The pipe,
companion, catalog, and toggles form one tested protocol set; do not combine a 3.0 pipe or
companion with a 2.x component.

## What changed

- Generation moves exclusively to `google-genai==2.11.0` Interactions.
- Developer API requests use `v1`.
- Stateful `previous_interaction_id` continuation and signed local stateless replay replace
  GenerateContent-era chat reconstruction.
- Grounding-envelope protocol 1 and evidence-bound model-catalog protocol 3 become hard boundaries.
- Catalog protocol 3 rejects duplicate/merge keys, stale or mismatched evidence, and provider/product
  contradictions before routing. Its full normalized policy is carried through app state with a
  canonical digest independently verified by the Pipe.
- Pricing records input/output modality and cached state explicitly, uses whole-prompt threshold
  selection, and reports unknown prices as incomplete instead of zero.
- The Reason toggle now emits `features.reasoning` and is version 2.0.0.
- The Maps toggle now emits `features.google_maps` and is version 2.0.0.
- The old Vertex AI toggle is removed and replaced by the Gemini Enterprise toggle.
- The unsupported custom content-filter control and its request configuration are removed.
- AFC, explicit cached-content input, Batch API configuration, and video metadata are rejected at
  the request boundary. Open WebUI functions run only through the suite's explicit bounded loop.
- Storage and privacy policy is explicit for persisted, temporary, task, edited, and branched
  chats.
- Every provider request explicitly sets `store`. Paid stored Interaction objects are currently
  retained for 55 days and free objects for one day. Separately, the paid-project AI Studio log
  setting offers 7, 14, 28, or 55 days and saved datasets do not expire with that window.
  `store=false` disables background execution and later previous-ID continuation; Search- and
  Maps-grounded data can retain for 30 days, Files persist until deletion or expiry, and local
  Open WebUI records/backups remain independent.
- Model or service changes never reuse a prior Interaction ID. Signed replay fails before network
  access when prior output modalities are not accepted by the selected target model.
- Open WebUI attachment and assistant-history file reads require an owner-filtered match for the
  authenticated requesting user. Missing or transplanted IDs fail before storage or Gemini access,
  including for administrators, without disclosing the identifier or backing path.
- Custom function rounds support both same-scope stored continuation and complete signed stateless
  replay for temporary, task, and privacy-opt-out requests.

## Upgrade sequence

1. Download the archive, manifest, and `SHA256SUMS` from the same draft/release.
2. Verify both downloaded files against `SHA256SUMS`, then verify each archive member against
   the manifest.
3. Back up the currently installed complete Gemini suite.
4. Install the companion first, followed by the Reason, Maps, URL Context, Paid API, and
   Enterprise toggles listed in `install_order`.
5. Install the 3.0.0 pipe last. Do not serve traffic while pipe and companion versions differ.
6. Run a non-sensitive temporary-chat request, a persisted two-turn request, and any enabled
   tool paths before reopening general traffic.

The Open WebUI Community pipe and filter entries are separately published objects. Their
3.0.0 updates must remain unpublished until every coordinated artifact is staged and verified;
then publish the companion/toggles before the pipe in one maintenance window.

## Rollback

Rollback means restoring the prior complete suite, not downgrading one file. Disable the Gemini
functions, restore the backed-up companion, toggles, catalog configuration, and pipe together,
then re-enable them. Messages containing a v3 `gemini_interaction` envelope are not a supported
input contract for a 2.x pipe; start a new branch/chat after rollback if replay fails.

## Current limitations

- Every Enterprise model in the bundled catalog is `unverified`; Enterprise requests are denied
  before network access. The SDK route under test is `v1beta1`, but that is not model-support
  evidence.
- The companion catalog default points to the prospective
  `gemini-suite/v3.0.0/plugins/pipes/gemini_models.yaml` tag. That URL returns successfully only
  after the coordinated tag is pushed. Stage the tag before enabling 3.0.0, or temporarily use
  an immutable reviewed catalog URL with the same schema and content.
- Deleting an Open WebUI chat does not delete provider-held Interactions because installed
  functions receive no portable chat-deletion callback. Provider deletion requires a known owned
  ID and would not remove Open WebUI envelopes/backups or Files API objects.
- Duplicate suppression is process-local; concurrent workers do not share a transaction.
- Custom Open WebUI function rounds support unary and SSE transport, execute same-round calls
  concurrently with deterministic result order, and remain bounded by the pipe's
  round/call/time/result-size limits. Stateless rounds resend the complete validated Step ledger
  and interaction-scoped configuration without a `previous_interaction_id`.

## Reproduce locally

```shell
uv run python .github/scripts/build_gemini_suite.py \
  --manifest release/gemini-suite-v3.0.0.yaml \
  --tag gemini-suite/v3.0.0
```
