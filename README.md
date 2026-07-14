# Open WebUI Functions

This repository contains Open WebUI pipes and filters developed primarily for the author's own
workflows and shared for others who find them useful.

> [!CAUTION]
> Open WebUI functions execute code with access to your Open WebUI process, configured APIs, and
> data. Review every file and its dependencies before installation.

## Stability

- `master` is active development and may be temporarily inconsistent.
- Immutable tags and their release artifacts are the stable checkpoints.
- Breaking multi-plugin protocols are released as coordinated suites, not as unrelated files.

## Plugins

### Gemini Interactions suite

Gemini suite 3.0.0 integrates Open WebUI with the Google Gen AI SDK
`google-genai==2.11.0` Interactions API. It consists of:

- `gemini_manifold.py` 3.0.0: Developer API `v1` pipe with catalog-gated Enterprise routing.
- `gemini_manifold_companion.py` 3.0.0: request integration, durable grounding protocol 1,
  citations, and service-qualified catalog protocol 2.
- Reason 2.0.0, Maps 2.0.0, URL Context 1.0.0, Paid API 1.0.0, and Enterprise 1.0.0 toggles.
- The immutable model catalog shipped in the same release.

Install these artifacts from the same `gemini-suite/v3.0.0` release and verify the attached
manifest and checksums. The pipe and companion are not backward compatible with 2.x. See the
[pipe guide](docs/plugins/pipes/gemini_manifold.md),
[companion guide](docs/plugins/filters/gemini_manifold_companion.md), and
[release procedure](docs/development/gemini-suite-release.md).

The catalog keeps complete capabilities and pricing inside each service policy. Developer entries
are audited; every Enterprise entry is capability-empty and `unverified`, so the pipe denies it
before client creation. Deterministic transport tests are not Enterprise model-support evidence.

### Other plugins

- `plugins/pipes/venice_manifold.py`: Venice.ai image generation pipe.
- `plugins/filters/system_prompt_injector.py`: chat-level system prompt and temperature controls.
- Additional standalone filters are under `plugins/filters/`.

## Installation and updates

For standalone plugins, copy the file from an immutable tag into Open WebUI's **Admin Panel →
Functions** editor. Preserve the exact frontmatter `id`.

For Gemini 3.0, do not copy or update one file independently. Follow the manifest `install_order`:
install the companion and toggles first, then the pipe. The catalog URL is pinned to the same
prospective suite tag and becomes available only after that tag exists. Community entries are
separate published objects and must be staged together in one maintenance window.

The optional one-time synchronizer remains available for non-suite development workflows:

```shell
uv sync --locked
uv run python dev/function_updater.py --env dev/.env.install
```

Do not use it to bypass the Gemini suite manifest/version checks.

## Development

`pyproject.toml` and `uv.lock` define the environment. Python 3.11 matches the target Open WebUI
runtime. Use:

```shell
make quality  # formats/fixes, then runs the complete gate
make check    # read-only CI-equivalent gate
```

Optional credential-gated Gemini smoke commands are documented in
[tests/live/README.md](tests/live/README.md). They redact output and skip without explicit opt-in.

## Archived functions

`plugins/archived/` is excluded from maintained quality and release surfaces. In particular,
`thinking_gemini.py` remains historical GenerateContent code and is not compatible with or part
of the Gemini Interactions suite.

## Contributing and license

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). The repository is available
under the [MIT License](LICENSE).
