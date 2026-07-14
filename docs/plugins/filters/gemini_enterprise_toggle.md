# Gemini Enterprise toggle 1.0.0

This optional Open WebUI toggle asks Gemini Manifold to prefer the administrator/user Enterprise
configuration. The pipe reads the toggle's active state; the filter sends no request payload of
its own.

The toggle is not proof that a selected model supports Enterprise Interactions. Suite 3.0's
catalog marks every Enterprise model `unverified`, so the pipe currently denies generation before
network access even when this toggle, project, location, and credentials are present. The audited
SDK transport route is `v1beta1`; support must be established per model and recorded in a new
immutable catalog release before that policy can change.

This replaces the removed `gemini_vertex_ai_toggle` identifier. There is no alias or compatibility
path; remove the old filter and install this one with the coordinated suite.

See the [pipe guide](../pipes/gemini_manifold.md).

## License

[MIT](../../../LICENSE)
