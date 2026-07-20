"""Generator-critic autoresearch substrate (docs/generator_critic.md, D023).

Execution method for I23 (pre-round prediction) and I13 (incremental
information). Submodules import only `cas.autoresearch.types` + stdlib/numpy/
sklearn to avoid import cycles; import them directly (e.g.
`from cas.autoresearch.features import build_features`) rather than via this
package, so a partially-built tree stays importable during development.
"""
