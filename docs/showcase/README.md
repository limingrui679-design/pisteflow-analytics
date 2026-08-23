# Evidence Studio

This directory is a dependency-free static browser for the package-bundled PisteFlow case catalog.
It works from any simple HTTP server and is deployed by the Pages workflow after GitHub Pages is
enabled for the repository.

```bash
python -m http.server 8000 --directory docs/showcase
```

Then open `http://localhost:8000/`. Opening `index.html` directly may prevent `fetch()` from reading
`cases.json`, so use an HTTP server for local review.

Do not edit `cases.json` independently. Update
`src/pisteflow/resources/showcase_cases.json`, review any intentional evidence changes, then refresh
the hashes and both synchronized views:

```bash
python scripts/sync_showcase.py
python scripts/sync_showcase.py --check
pisteflow showcase audit --evidence-root .
```

The app has no analytics, trackers, cookies, external fonts, or JavaScript dependencies.
