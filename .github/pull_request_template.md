## What changed

Describe the user-facing or evidence-facing change.

## Evidence contract

- [ ] Units, source lineage, rights state, and privacy constraints remain explicit.
- [ ] People, events, source videos, dates, and duplicates are grouped before splitting where relevant.
- [ ] Comparative claims include a credible baseline; negative results remain visible.
- [ ] Internal validation is not described as external validation, adoption, or impact.
- [ ] New or changed case claims point to machine-readable evidence.

## Verification

- [ ] `ruff check .`
- [ ] `mypy`
- [ ] `pytest -q`
- [ ] `pisteflow showcase audit --evidence-root .`
