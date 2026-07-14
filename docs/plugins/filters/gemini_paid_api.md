# Gemini Paid API toggle 1.0.0

This optional Open WebUI toggle asks Gemini Manifold to route the main request through the paid
Gemini Developer API credential. The pipe reads the toggle's active state; the filter has no inlet
payload and never contains an API key.

Configure `GEMINI_PAID_API_KEY` on the pipe or permitted user valves. This is still Developer API
`v1` Interactions; “paid” selects a credential/routing tier, not a different protocol. Model and
feature catalog checks remain mandatory.

`ENABLE_FREE_TIER_FALLBACK` can retry eligible free-tier 429/503 failures on the paid credential.
`TASK_MODEL_ROUTING` independently controls task requests. The toggle does not override an
administrator's authentication policy and is distinct from Gemini Enterprise.

See the [pipe guide](../pipes/gemini_manifold.md).

## License

[MIT](../../../LICENSE)
