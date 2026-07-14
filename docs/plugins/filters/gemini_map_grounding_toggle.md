# Google Maps grounding toggle 2.0.0

This optional per-chat toggle sets `metadata.features.google_maps=true` for Gemini Manifold.
Pipe 3.0 checks the model catalog before adding the Interactions Google Maps server tool.

Version 2.0.0 is breaking: older releases used `google_maps_grounding`. Install this toggle with
Gemini suite 3.0.0. Optional `MAPS_GROUNDING_COORDINATES` on the pipe supplies a validated
`latitude,longitude` bias. The companion renders place sources and widget context records from
grounding-envelope protocol 1; the toggle itself does not call Maps or emit citations.

Unsupported or catalog-excluded Maps requests fail before an API call, including free-tier
routes where the catalog lists Maps as an excluded paid feature.

See the [pipe guide](../pipes/gemini_manifold.md) and
[companion guide](gemini_manifold_companion.md).

## License

[MIT](../../../LICENSE)
