# Archived plugins

Everything in this directory is unsupported, excluded from maintained static analysis and release
bundles, and retained only as historical reference. Do not install archived code in a maintained
deployment and do not interpret its APIs as compatibility promises.

- `smallthinker_filter.py`: discontinued experimental reasoning formatter.
- `thinking_gemini.py`: historical `google-generativeai`/GenerateContent implementation for an
  obsolete experimental model. It is not migrated, tested, published, or included in Gemini
  Interactions suite 3.0.0. Use `plugins/pipes/gemini_manifold.py` and the coordinated suite
  instead.
