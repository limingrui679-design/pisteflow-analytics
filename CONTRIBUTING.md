# Contributing to PisteFlow

PisteFlow welcomes fixes, tests, documentation, new source adapters, and carefully bounded analyses.
The core contribution rule is simple: improve what the repository can prove without widening what
it claims.

## Start locally

```bash
git clone https://github.com/limingrui679-design/pisteflow-analytics.git
cd pisteflow-analytics
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,video,collectors,ffd]'
ruff check .
mypy
pytest -q
pisteflow showcase audit --evidence-root .
```

## Evidence rules

1. Define the unit before counting it. A row may be a person, membership, bout, touch, frame,
   clip, facility record, or repeated measurement.
2. Preserve provenance and rights. Publicly visible does not mean redistributable; unresolved media
   stays link-only or local-only.
3. Design the split around leakage. Keep people, source videos, competitions, dates, and duplicate
   groups isolated where the intended claim requires it.
4. Compare with a credible simple baseline. Chance is rarely the only operational alternative.
5. Keep negative and mixed results. Do not delete a valid analysis because the model loses.
6. Bound the conclusion. Internal reproducibility is not external validation, certification,
   adoption, deployment, causal effect, or user impact.

## New source adapters

- Add the source and rights decision to `data/registry/sources.yml`, or record a deliberate rejection
  in `data/registry/source-exclusions.yml`.
- Pin immutable identifiers or hashes where the publisher permits it.
- Store large, restricted, or row-level inputs under ignored local data paths.
- Commit aggregate manifests that expose row counts, exclusions, checksums, and a claim boundary.
- Add schema, parser, identity, and failure-path tests.

## New analyses and cases

- Put reusable logic in `src/pisteflow/analysis/` and expose an explicit CLI command.
- Write only aggregate public evidence to `reports/model-cards/`; keep sensitive predictions and
  model artifacts local.
- Add a case to `src/pisteflow/resources/showcase_cases.json` only after its evidence paths exist.
- Review the evidence change, then refresh all SHA-256 bindings and synchronized views:

```bash
python scripts/sync_showcase.py
python scripts/sync_showcase.py --check
pisteflow showcase audit --evidence-root .
```

## Pull requests

Keep a pull request focused. Explain the intended evidence claim, what can invalidate it, and the
commands used to verify it. Do not attach credentials, private athlete information, copyrighted
raw media, or data whose terms prohibit redistribution.
