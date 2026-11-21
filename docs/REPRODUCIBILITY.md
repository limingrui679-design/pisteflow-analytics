# Reproducibility guide

PisteFlow offers four reproducibility levels so a reviewer can choose how much data, time, and rights
risk to assume.

## Level 0 — inspect cases with no data

This path works from a wheel or editable install and does not read repository datasets.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
pisteflow showcase list
pisteflow showcase show ranking-trajectory
pisteflow showcase list --status no-lift
```

The catalog is package data under `src/pisteflow/resources/`. Every case includes its unit, method,
validation, baseline, result status, decision use, evidence paths, and claim boundary.

## Level 1 — verify committed evidence

```bash
python -m pip install -e '.[dev]'
pisteflow showcase audit --evidence-root .
pytest -q tests/test_showcase.py
```

The audit confirms all eight capability lenses are covered, evidence paths exist, analysis cards
contain claim boundaries, every one of the 23 evidence files matches its catalogued SHA-256, and the
static demo catalog exactly matches the package catalog.

## Level 2 — run bounded public acquisitions

Source commands are independent. Start with small, openly licensed examples rather than the full
catalog.

```bash
python -m pip install -e '.[dev,video,collectors]'
pisteflow sources ucf101-discover
pisteflow sources ucf101-download
pisteflow sources commons-discover
pisteflow sources commons-download --workers 2
```

Read `docs/DATA_POLICY.md` before acquiring larger sources. Link-only, unknown-license, restricted,
and privacy-sensitive sources remain local by design.

## Level 3 — reproduce an analysis

Each analysis names its prepared local input and writes aggregate evidence separately from local
models or row-level outputs.

```bash
pisteflow analysis competition-elo
pisteflow analysis foil-pose-outcomes
pisteflow analysis hot-hand-validation
pisteflow analysis france-spatial-access
```

For the spatial case, acquire or prepare these three local tables first:

```text
data/processed/france-fencing-licences-2024-department-age-sex.parquet
data/processed/france-fencing-clubs-2022-2024.parquet
data/processed/france-fencing-equipment.parquet
```

The analysis writes a local department table and the public aggregate card
`reports/model-cards/france-spatial-access.json`.

## Full repository verification

```bash
python -m pip install -e '.[dev,video,collectors,ffd]'
ruff check .
mypy
pytest -q
python scripts/sync_showcase.py --check
pisteflow showcase audit --evidence-root .
python -m build
python scripts/audit_distribution.py dist
```

CI runs the same quality gates on Python 3.11 and 3.12, regenerates the demo catalog for a byte-level
comparison, builds both distributions, installs the wheel without dependencies, and confirms that
all 15 package-bundled cases remain available.

CI also measures package-wide statement coverage and fails below 55%. The current broad adapter
surface measures 56%; the spatial analysis, case audit, and provenance controls are above 80%.
Coverage is a regression signal, not evidence of source truth, statistical validity, or external
impact, and adapters are not removed merely to raise the percentage.

The source distribution keeps code, tests, documentation, registries, aggregate manifests, model
cards, and synchronization tools. Raw or processed data, caches, models, artifacts, and row-level
JSONL indexes remain in their documented local or Git-repository workflows rather than inflating a
package-manager download. The distribution audit enforces a 5 MiB ceiling, rejects unsafe archive
paths and links, rejects excluded payload classes, and verifies the required runtime and review
surfaces are present.

## Determinism and hashes

- Randomized analyses pin their seed in code and public evidence.
- Presentation-side flipping uses stable hashes where source ordering would reveal the winner.
- Same-day Elo predictions use ratings entering the day and batch updates afterward.
- Resampling units follow the evidence unit: departments for spatial associations and pools for the
  hot-hand comparison.
- Source and output SHA-256 values appear in committed aggregate manifests or analysis cards where
  redistribution permits verification.
- The case catalog binds every referenced public evidence file by SHA-256. Maintainers run
  `python scripts/sync_showcase.py` only after reviewing an intentional evidence change; `--check`
  is non-mutating and enforced in CI.
- Persisted JSON provenance converts managed absolute paths to repository-relative POSIX paths, so
  a clean clone does not inherit an author's local home directory.
- JSON writes are key-sorted, UTF-8, atomic, and newline-terminated.

## What reproduction does not establish

Reproducing a metric confirms that the public code and named input produce the recorded internal
result. It does not establish source truth, causal effect, independent review, federation approval,
clinical validity, organizational adoption, production reliability, or user impact.

### Who may cross a ranking boundary by the next recorded snapshot?

For each forecast origin, estimate the chance of entering or leaving an illustrative top 16. The forecaster learns signed rank movements from development transitions whose outcomes precede every forecast origin. It groups by weapon and current-rank band (powers of two) and shrinks each cell distribution toward the pooled changes with 20 prior observations. The numerical rank determines the movement needed for a crossing. `predict_boundary_crossings(development, origins)` accepts an unlabelled set of origins with just weapon, rank and date.

Replay a review queue separately on each date with slots for 5% of observed eligible transitions, rounded up. Equal scores at a budget cutoff share the remaining slots uniformly. On 47,184 training and 15,945 held-out USA category-series transitions, 452 crossed the boundary. The probability score was **0.02466 Brier**, against 0.02754 for historical prevalence and 0.02835 for predicting no change in membership. Over 60 dates and 824 review slots, the risk queue captured **170.83 expected crossings**, compared with **167.99** for ranking by distance to the boundary and **23.63** for random review within date. Fractional hits integrate ties. The historical exercise evaluates series with an observed follow-up inside 180 days and reports conditional queue performance for that cohort.

Use version 1 of the [USA Fencing Team Archives](https://www.kaggle.com/datasets/ajaxofgreee/usa-fencing-team-archives), pinned by `usa_team_archives.py` to archive SHA-256 `a13d04ac1e7123de91705a25f5a5f71ffa214216fa60070316e24a62ad159872`. Run `pisteflow analysis ranking-trajectory` on the normalized USA history. The `test.boundary_review` field contains the queue and probability results. Source parquet SHA-256: `ddda27703275bc78bf0a21957fc467fbafa03cf694ca9849878240b0253020a9`.

Across 15 observed calendar months, the risk queue led proximity in eight, proximity led in five, and two tied. Dropping one whole month at a time leaves a gain of 0.04 to 4.19 expected crossings at the top-16 boundary. Replaying the same fixed design at the top-8 and top-32 boundaries gives 197.00 vs 190.58 and 159.77 vs 144.69 expected hits, respectively.
