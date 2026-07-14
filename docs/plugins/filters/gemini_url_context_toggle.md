# URL Context toggle 1.0.0

This optional per-chat toggle sets `metadata.features.url_context=true` for Gemini Manifold.
Pipe 3.0 then checks the model catalog and either adds the Interactions URL Context server tool or
rejects the unsupported request before network access.

The filter does not enable an administrator fallback valve and does not retrieve URLs itself.
Citation/source rendering requires companion 3.0.0 and a matching durable grounding envelope.
Provider URL statuses such as `paywall` or `unsafe` are surfaced as warnings rather than treated
as successful source reads.

Enable this filter only for Gemini Manifold models. The selected model must have
`services.<selected>.interactions.tools.url_context: true` in the immutable schema-2 catalog.

See the [pipe guide](../pipes/gemini_manifold.md) and
[companion guide](gemini_manifold_companion.md).

## License

[MIT](../../../LICENSE)
