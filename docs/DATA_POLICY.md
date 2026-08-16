# Data policy

PisteFlow separates **discovery, access, local analysis, training, redistribution, and claim
strength**. A public page or downloadable file does not automatically satisfy all six.

## Inclusion states

| State | Raw acquisition | Local analysis | Commit raw rows / media | Public outputs |
|---|:---:|:---:|:---:|---|
| Verified open license | Yes | Yes | Only when attribution, privacy, and size allow | Code, permitted rows / derivatives, hashes, aggregates |
| Public facts without reuse terms | Bounded local snapshot | Yes | No | Source links, hashes, non-reconstructive aggregates |
| Link-only public media | No project media download | Metadata only | No | IDs, landing pages, reviewed labels, no media |
| Unknown dataset license | Optional local archive | Yes | No | Code, exact hashes, counts, aggregate metrics |
| Restricted source | No default adapter | No | No | Restriction decision only |
| Privacy-sensitive or no-derivatives | Local only | Bounded | No | Deidentified aggregates; no adapted row table |
| Synthetic | Yes | Yes | Yes | Always identified as synthetic and never added to real-data counts |

## Evaluation gates

1. **Source grouping:** frames or clips from one source video remain in one fold. Mirrored clips keep
   the original group ID.
2. **Pose outcomes:** 42 repeated source windows, scoring-light codes, target fields, and static joint
   position means are excluded. Five folds group all 9,328 clips by 1,332 source videos.
3. **Athlete measures:** the 45-person strength analysis uses participant-level outer folds with inner
   hyperparameter selection. Both lunge-velocity outcomes are excluded from predictors.
4. **Competition replay:** source rows place the winner on the left, so presentation orientation is
   deterministically randomized. All bouts on a date are predicted from ratings entering that date;
   results update ratings only after every same-day prediction.
5. **Ranking trajectories:** features use the current or prior snapshot only. Target dates before 2020
   train the model; 2020–2023 target dates form the test. Persistence is always reported.
6. **Weapon reporting:** epee, foil, and sabre metrics are shown separately before pooled values.
   Generic fencing or swordplay never receives an inferred weapon label.
7. **Negative results:** a model that fails to beat its simple baseline remains in the model card and
   README. It is not hidden behind a different metric.

## Deduplication and units

- Exact repeats, packaging duplicates, mirror augmentations, and overlapping historical snapshots are
  flagged before training counts are calculated.
- A clip, frame, bout, period, relay, athlete-event membership, ranking snapshot, participant, and
  measurement cell keep different units.
- Paper-reported samples are not “acquired rows” unless a public file is found, pinned, read, and
  validated.
- Remote archive size or a link count is not described as locally held media.

## Privacy

- Direct names, clubs, countries, regions, referee fields, free-text injury narratives, and exact birth
  dates are removed where they are unnecessary for analysis.
- Stable hashes are called **pseudonymous**, not anonymous.
- For U15 measurements, source row labels, sex code, urine-protein code, and puberty / maturation
  fields are excluded from the analytic table.
- Small clinical, injury, wheelchair, and intervention datasets are never presented as population
  norms or individual prescriptions.

## Publication boundary

The repository may publish acquisition and validation code, source URLs, rights decisions, immutable
hashes, open-license manifests, privacy-minimized schemas, non-reconstructive features when permitted,
aggregate charts, and aggregate evaluation reports. It does not publish copyrighted tournament video,
unknown-license model inputs or weights, private identifiers, scraped current FIE rows, or source data
whose terms prohibit adapted redistribution.

Internal tests, checksums, held-out folds, and chronological replay establish **internal
reproducibility only**. They do not establish external expert review, federation approval, live
deployment, user adoption, athlete benefit, injury prevention, or referee-grade decision accuracy.
