# Reason toggle 2.0.0

This optional toggle sets `metadata.features.reasoning=true` for Gemini Manifold requests. Pipe
3.0 validates the selected model's catalog entry before requesting Interactions thinking and
automatic summaries.

Version 2.0.0 is breaking: older releases used the noncanonical `reason` feature key. Install the
2.0.0 toggle with Gemini suite 3.0.0. `THINKING_LEVEL`, `THINKING_SUMMARIES`, and any valid
per-request `reasoning_effort` determine the request; this filter does not implement a token
budget or expose opaque thought signatures.

Enable the filter globally or for the Gemini Manifold models where a per-chat lightbulb toggle is
desired. Without it, reasoning remains off unless another trusted integration supplies the same
canonical feature flag.

See the [pipe guide](../pipes/gemini_manifold.md).

## License

[MIT](../../../LICENSE)
