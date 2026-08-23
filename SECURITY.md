# Security policy

## Supported versions

Security fixes are applied to the current `main` branch and the latest tagged release. Earlier
research snapshots are not maintained as separate security branches.

## Report privately

Use [GitHub private vulnerability reporting](https://github.com/limingrui679-design/pisteflow-analytics/security/advisories/new)
for vulnerabilities that could expose local files, credentials, private athlete data, or unsafe
download behavior. Do not place exploit details or sensitive sample data in a public issue.

Include the affected commit, operating system, minimal reproduction, impact, and any suggested
mitigation. Receipt and remediation timing depend on maintainer availability; no service-level
agreement is promised.

## Data safety boundary

PisteFlow is a local research toolkit, not a hosted service. Raw media, model artifacts, credentials,
and privacy-sensitive rows are excluded from Git by policy. A security report is not the right route
for ordinary data-quality disagreements; use a public issue when the discussion contains no private
or exploitable information.
