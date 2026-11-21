# PisteFlow case studies

Catalog 0.2.0 contains 15 bounded cases. Negative and mixed results remain first-class evidence.

| Case | Capability lenses | Evidence status | Headline |
|---|---|---|---|
| [Chronological three-weapon Elo](#three-weapon-elo) | decision-analytics, statistical-inference, risk-finance | lift | Test AUC 0.817 and log loss 0.526 |
| [Source-grouped foil pose outcomes](#foil-pose-outcomes) | reproducible-ml, statistical-inference, decision-analytics | lift | Balanced accuracy 0.465; macro-F1 0.464 |
| [Ranking trajectory negative result](#ranking-trajectory) | statistical-inference, decision-analytics, risk-finance | no-lift | Unblended challenger MAE 2.116 places |
| [Strength and lunge velocity associations](#strength-lunge-velocity) | population-health, statistical-inference, reproducible-ml | mixed | Advance RMSE 0.449 m/s; static RMSE 0.179 m/s |
| [Seed-aware junior pool outcomes](#junior-pool-outcomes) | decision-analytics, risk-finance, statistical-inference | no-lift | Model accuracy 0.8474 and AUC 0.9167 |
| [Sabre gaze and touch outcomes](#sabre-eye-outcomes) | behavioral-science, population-health, statistical-inference | no-lift | Model AUC 0.578 and log loss 0.691 |
| [Hot-hand incremental validation](#hot-hand-validation) | behavioral-science, statistical-inference, decision-analytics | no-lift | Recent-form and previous-outcome additions worsened point metrics |
| [Participant-held-out action recognition](#mfad-action-recognition) | reproducible-ml, population-health, statistical-inference | lift | Balanced accuracy 0.794 and macro-F1 0.798 |
| [Undocumented referee-label diagnostic](#referee-label-diagnostic) | data-systems-governance, reproducible-ml, statistical-inference | mixed | Balanced accuracy improves, while log loss worsens |
| [Pose-label semantics diagnostic](#pose-semantics-diagnostic) | data-systems-governance, reproducible-ml, statistical-inference | diagnostic | AUC 0.574, but target meaning remains undocumented |
| [Regional Elo replication](#new-zealand-elo-replication) | statistical-inference, decision-analytics, risk-finance | replication | Test accuracy 0.715 and AUC 0.799 |
| [French fencing spatial access context](#france-spatial-access) | spatial-policy, population-health, data-systems-governance, statistical-inference | descriptive | 101 departments; membership rate rho 0.580 with establishment density |
| [Source and rights governance system](#source-rights-governance) | data-systems-governance, risk-finance, decision-analytics | descriptive | Evidence state and reuse rights remain separate fields |
| [Health and accessibility evidence matrix](#health-data-evidence-matrix) | population-health, data-systems-governance, behavioral-science | descriptive | Heterogeneous health evidence stays analysis-ready without becoming one false cohort |
| [Cross-source image identity audit](#image-identity-audit) | data-systems-governance, reproducible-ml, statistical-inference | descriptive | 334 duplicate extras are isolated before any model split |

## Chronological three-weapon Elo

<a id="three-weapon-elo"></a>

A large chronological replay converts a biased winner-left source into orientation-balanced probability forecasts and delays every same-day update.

- **Question:** Can pre-day ratings rank held-out bout outcomes without leaking same-day results or source-side ordering?
- **Unit / sample:** decisive bout; 374,088 eligible bouts; 16,365 untouched 2020 test bouts
- **Method:** Pre-day batch Elo with deterministic presentation flipping and validation-only K selection
- **Validation:** K selected on 2019; untouched chronological test from 2020-01-01; same-day updates delayed
- **Result:** Test AUC 0.817 and log loss 0.526
- **Baseline:** Uninformative probability baseline: AUC 0.500 and log loss 0.693
- **Decision use:** Demonstrates calibrated ranking evidence and temporal controls for bounded forecasting decisions
- **Boundary:** This is an internal replay of a third-party historical snapshot, not a current official forecast, causal training model, athlete-independent external validation, or deployed decision system.
- **Evidence:** `reports/model-cards/competition-elo.json`

## Source-grouped foil pose outcomes

<a id="foil-pose-outcomes"></a>

A pose baseline excludes scoring lights and duplicate windows, then evaluates all clips with source-video-held-out cross-validation.

- **Question:** Do pose-only trajectory features retain outcome signal when every clip from the same source video stays in one fold?
- **Unit / sample:** 75-frame clip; 9,328 clips from 1,332 source-video groups
- **Method:** Pose-trajectory features with five-fold StratifiedGroupKFold by source video
- **Validation:** Zero source-video overlap across folds; scoring-state features and duplicate windows excluded
- **Result:** Balanced accuracy 0.465; macro-F1 0.464
- **Baseline:** Majority empirical-probability baseline: balanced accuracy 0.333; macro-F1 0.216
- **Decision use:** Shows how to turn video data into an auditable baseline while isolating source identity
- **Boundary:** The result is an internal foil-archive association baseline, not causal tactical evidence, a live referee, external event validation, athlete-held-out generalization, or production deployment.
- **Evidence:** `reports/model-cards/foil-pose-outcomes.json`

## Ranking trajectory negative result

<a id="ranking-trajectory"></a>

A chronological model tests an intuitive feature-rich approach against a strong persistence rule and preserves the inferior result.

- **Question:** Can engineered history features improve the next observed ranking snapshot over simply carrying the current rank forward?
- **Unit / sample:** next-snapshot transition; 47,184 development transitions; 742 crossing pairs purged; 15,945 held-out origins in 2,375 series
- **Method:** Two expanding development folds select a blend by series-balanced MAE. Forecasts use current and earlier features only.
- **Validation:** Fitted labels precede every fold origin. Final test origins are on or after 2020-01-01.
- **Result:** Unblended challenger MAE 2.116 places
- **Baseline:** Selected persistence MAE 2.062 places
- **Decision use:** Export the persistence rule chosen using development data. Held-out series-balanced improvement over persistence was exactly zero.
- **Boundary:** Category series are not verified unique people and may span both sides of the time split; this is not an official ranking feed, selection forecast, or causal athlete-development model.
- **Evidence:** `reports/model-cards/ranking-trajectory.json`

## Strength and lunge velocity associations

<a id="strength-lunge-velocity"></a>

Participant-level nested validation separates a small advance-lunge signal from a static-lunge result that fails to beat the mean.

- **Question:** Do cross-sectional strength and jump measurements improve held-out lunge-velocity estimates across weapons?
- **Unit / sample:** participant; 45 elite female fencers: 17 epee, 12 foil, and 16 sabre
- **Method:** Nested five-fold participant-level ridge regression with weapon-stratified outer folds
- **Validation:** Both outcomes excluded from predictors; inner-fold regularization selection; participant-held-out errors
- **Result:** Advance RMSE 0.449 m/s; static RMSE 0.179 m/s
- **Baseline:** Mean baselines: advance RMSE 0.477 m/s; static RMSE 0.177 m/s
- **Decision use:** Illustrates sample-size-aware health and performance analysis without turning associations into prescriptions
- **Boundary:** The one-time measurements support cross-sectional associations only; they are not a longitudinal intervention, causal training effect, population norm, external validation, or athlete prescription.
- **Evidence:** `reports/model-cards/strength-velocity.json`

## Seed-aware junior pool outcomes

<a id="junior-pool-outcomes"></a>

An identity-free chronological baseline gains strongly over chance but not over the domain heuristic that actually matters for comparison.

- **Question:** Does a multivariable probability model materially outperform the simple decision rule of choosing the higher seed?
- **Unit / sample:** junior pool bout; 158,580 eligible bouts; 37,813 untouched 2025-2026 test bouts
- **Method:** Regularized logistic regression using signed log seed ratio, weapon, and gender
- **Validation:** Train before 2024, select on 2024, test from 2025; competitions isolated across splits
- **Result:** Model accuracy 0.8474 and AUC 0.9167
- **Baseline:** Higher-seed rule accuracy 0.8471 and AUC 0.9167
- **Decision use:** Shows why deployment decisions should compare against the strongest simple operational heuristic
- **Boundary:** This is held-out historical association evidence from a third-party transcription, not causal coaching impact, senior or international generalization, official data-quality certification, or deployment evidence.
- **Evidence:** `reports/model-cards/osf-pool-outcomes.json`

## Sabre gaze and touch outcomes

<a id="sabre-eye-outcomes"></a>

Nested leave-one-participant-out analysis tests a behavioral mechanism while keeping the weaker probabilistic result visible.

- **Question:** Do eye-tracking and tactical-phase features generalize to a completely held-out participant beyond phase alone?
- **Unit / sample:** labelled touch; 379 eligible touches from 10 national-level participants
- **Method:** Logistic models over gaze and phase features with nested leave-one-participant-out selection
- **Validation:** Every outer participant is absent from fitting and hyperparameter selection for that fold
- **Result:** Model AUC 0.578 and log loss 0.691
- **Baseline:** Phase-only AUC 0.498 and log loss 0.685
- **Decision use:** Separates behavioral measurement from a claim that gaze variables improve decisions or training
- **Boundary:** The result is a small internal association study, not causal evidence that gaze training improves touches, a live referee, a model for other weapons, external validation, or a coaching prescription.
- **Evidence:** `reports/model-cards/sabre-eye-outcomes.json`

## Hot-hand incremental validation

<a id="hot-hand-validation"></a>

A common-sample, leave-one-pool-out design tests recent-form features and reports intervals around their incremental value.

- **Question:** Does recent within-pool success improve predictions beyond pre-event rating when uncertainty resamples whole pools?
- **Unit / sample:** complete-case pool bout; 161 bouts across 13 pools in one Junior Men's Foil event
- **Method:** Rating-only and recent-form logistic models with deterministic side flipping
- **Validation:** Leave-one-pool-out predictions on a common sample; 2,000 pool-level bootstrap repetitions
- **Result:** Recent-form and previous-outcome additions worsened point metrics
- **Baseline:** Rating-only AUC 0.868 and log loss 0.453; incremental intervals span zero
- **Decision use:** Tests an appealing behavioral story before allowing it to influence training or forecasting
- **Boundary:** One event with 161 complete cases cannot establish the presence or absence of a causal hot hand, athlete-independent generalization, coaching value, or real-world intervention impact.
- **Evidence:** `reports/model-cards/february-nac-2026-hot-hand.json`

## Participant-held-out action recognition

<a id="mfad-action-recognition"></a>

A depth-skeleton baseline removes identity and skill labels, groups complete participants, and exposes subgroup sample limits.

- **Question:** Can normalized skeleton dynamics distinguish three fencing actions for participants absent from model fitting?
- **Unit / sample:** depth-camera skeleton sequence; 928 sequences from 27 author-coded participants
- **Method:** Translation- and scale-normalized trajectory features with grouped multiclass classification
- **Validation:** Five-fold StratifiedGroupKFold by participant with zero participant overlap
- **Result:** Balanced accuracy 0.794 and macro-F1 0.798
- **Baseline:** Majority baseline balanced accuracy 0.327 and macro-F1 0.244
- **Decision use:** Demonstrates identity-safe human-motion modeling and transparent subgroup limitations
- **Boundary:** This is internal action classification, not weapon identification, proof that author-assigned skill labels are correct, external validation, a coaching prescription, or a deployed motion-analysis product.
- **Evidence:** `reports/model-cards/mfad-actions.json`

## Undocumented referee-label diagnostic

<a id="referee-label-diagnostic"></a>

The analysis treats missing provenance as a first-class limitation and refuses to relabel predictability as referee correctness.

- **Question:** How much internal label predictability remains when source grouping is imperfect and label semantics are unverified?
- **Unit / sample:** exact-unique clip; 395 clips grouped by 55 filename source numbers
- **Method:** Motion-feature classifier over upstream left, right, and simultaneous folders
- **Validation:** Five-fold grouped evaluation by the strongest available filename source signal
- **Result:** Balanced accuracy improves, while log loss worsens
- **Baseline:** Model balanced accuracy 0.536 vs 0.333; log loss 1.072 vs 1.024
- **Decision use:** Provides a governance stop signal when semantics and provenance do not support a product claim
- **Boundary:** Weapon, athletes, events, source URLs, original-video IDs, and rater data are absent, so this is label-predictability evidence only—not referee correctness, human agreement, or deployment evidence.
- **Evidence:** `reports/model-cards/asmi-referee-labels.json`

## Pose-label semantics diagnostic

<a id="pose-semantics-diagnostic"></a>

A large pose diagnostic shows measurable signal while preserving the more important conclusion that the upstream target is not semantically validated.

- **Question:** Can undocumented left-right labels be predicted after transformation-linked sequences are kept together and static position cues are removed?
- **Unit / sample:** exact-unique pose sequence; 28,727 sequences in 28,312 conservative geometry groups
- **Method:** Dynamic pose features with static center and body-size means excluded
- **Validation:** Five-fold StratifiedGroupKFold by transformation-invariant geometry hash
- **Result:** AUC 0.574, but target meaning remains undocumented
- **Baseline:** Empirical-probability baseline AUC 0.500
- **Decision use:** Shows that predictive signal cannot repair an undefined target or missing provenance
- **Boundary:** Original videos, events, athletes, weapons, frame rate, grouping, and left-right meaning are missing; this is not semantic validation, a referee model, unseen-event evidence, or deployment evidence.
- **Evidence:** `reports/model-cards/fencingly-side-labels.json`

## Regional Elo replication

<a id="new-zealand-elo-replication"></a>

A separate New Zealand pipeline reuses the temporal contract but reselects design choices before an untouched 2026 test.

- **Question:** Does the pre-day Elo design retain held-out signal in a different country, period, and public-results system?
- **Unit / sample:** decisive public-result bout; 18,514 eligible 2024-2026 bouts; 4,706 untouched 2026 test bouts
- **Method:** Pre-day Elo with validation-selected K and shared-versus-stage-specific rating key
- **Validation:** Design selected on 2025 and evaluated on an untouched 2026 chronological period
- **Result:** Test accuracy 0.715 and AUC 0.799
- **Baseline:** Uninformative probability baseline AUC 0.500
- **Decision use:** Tests transportability of a method without calling one regional replay external certification
- **Boundary:** This is an internal regional replication, not an official federation rating, current forecast, causal training model, independent external review, organizational adoption, or deployed decision system.
- **Evidence:** `reports/model-cards/fencingstatsnz-elo.json`

## French fencing spatial access context

<a id="france-spatial-access"></a>

A cross-source department analysis combines official open licence, club, population, and facility records while preserving mismatched dates and denominators.

- **Question:** How do membership rates, active-establishment density, and listed facility attributes vary across French departments?
- **Unit / sample:** French department and public facility record; 101 in-scope departments plus 1,124 listed fencing-related facilities
- **Method:** Department aggregation, rate denominators, rank correlations, and department bootstrap intervals
- **Validation:** Source reconciliation, geography normalization, denominator checks, coverage reporting, and deterministic resampling
- **Result:** 101 departments; membership rate rho 0.580 with establishment density
- **Baseline:** Not applicable; descriptive case
- **Decision use:** Supports place-based problem definition and identifies where better capacity or participation data are needed
- **Boundary:** Memberships are not unique people, facility records are not capacity or availability, 2024 measures and records updated through 2026 are temporally mismatched, and department associations do not establish access barriers or policy effects.
- **Evidence:** `reports/model-cards/france-spatial-access.json`, `data/manifests/france-fencing-licences-summary.json`, `data/manifests/france-fencing-infrastructure-summary.json`

## Source and rights governance system

<a id="source-rights-governance"></a>

A machine-readable registry separates discovered, link-only, locally retained, and redistributable evidence instead of treating public visibility as permission.

- **Question:** Can a multi-source sports repository make acquisition, licensing, privacy, and exclusion decisions reviewable before modeling?
- **Unit / sample:** source record and explicit exclusion decision; 103 unique source records with separate curated and exclusion registries
- **Method:** Schema-bound source inventory, acquisition state, rights status, checksums, privacy minimization, and exclusions
- **Validation:** Registry parsing, source-specific tests, immutable hashes where allowed, and link-only versus acquired separation
- **Result:** Evidence state and reuse rights remain separate fields
- **Baseline:** Not applicable; descriptive case
- **Decision use:** Creates auditable go, local-only, link-only, and exclude decisions before downstream analysis
- **Boundary:** A catalogued or downloadable source is not automatically accurate, reusable, representative, acquired, externally reviewed, adopted by an organization, or safe for production use.
- **Evidence:** `data/registry/sources.yml`, `data/registry/source-exclusions.yml`, `docs/DATA_POLICY.md`

## Health and accessibility evidence matrix

<a id="health-data-evidence-matrix"></a>

Independent adapters retain participant, measurement, survey, period, and membership units while attaching privacy and evidence boundaries to each source.

- **Question:** How can heterogeneous injury, physiology, cognition, disability-sport, and survey sources be normalized without merging incompatible units?
- **Unit / sample:** source-specific participant, measurement, survey, or membership record; Multiple open health, biomechanics, cognition, injury, and adapted-fencing sources
- **Method:** Source-specific normalization with pseudonymous keys, schema checks, provenance hashes, and unit preservation
- **Validation:** Adapter tests and aggregate manifests keep repeated measures, memberships, participants, and outcomes distinct
- **Result:** Heterogeneous health evidence stays analysis-ready without becoming one false cohort
- **Baseline:** Not applicable; descriptive case
- **Decision use:** Provides a safer information architecture for later population-level questions and study design
- **Boundary:** These independent public sources do not form one cohort, establish clinical validity, prove treatment effects, represent all fencers, provide healthcare guidance, or show real-world organizational adoption.
- **Evidence:** `data/manifests/injury-surveys-summary.json`, `data/manifests/physionet-actes-fencer-cardio-summary.json`, `data/manifests/plos-fencer-cognition-summary.json`, `data/manifests/riposte-adapted-fencing-summary.json`, `data/manifests/korea-gkl-wheelchair-fencing-summary.json`

## Cross-source image identity audit

<a id="image-identity-audit"></a>

The union audit separates exact duplicates, identity-linked extras, and perceptual review candidates instead of silently inflating sample size.

- **Question:** How many nominal image rows remain independent after exact-content and conservative source-identity grouping?
- **Unit / sample:** checksummed image row and conservative identity group; 20,240 rows resolving to 19,906 exact or source-identity groups
- **Method:** Cryptographic hashes, stable source identities, and review-only perceptual near-neighbor flags
- **Validation:** Exact duplicates are grouped deterministically; perceptual similarity never triggers an automatic merge
- **Result:** 334 duplicate extras are isolated before any model split
- **Baseline:** Not applicable; descriptive case
- **Decision use:** Protects evaluation from mirrored sources and duplicate leakage while retaining uncertain review cases
- **Boundary:** Identity groups reduce obvious duplication but do not prove semantic correctness, licensing compatibility, frame independence, complete source lineage, model generalization, or production readiness.
- **Evidence:** `data/manifests/image-union-audit-summary.json`
