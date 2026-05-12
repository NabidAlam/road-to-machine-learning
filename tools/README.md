# Tools

Small maintenance utilities for keeping the repository consistent over time.

## `check_md_links.py`

Scans **all** `*.md` files under the repo for broken **relative** links and missing **heading anchors** (GitHub-style slug rules). Skips external URLs (`https://`, `mailto:`, etc.).

Run from repo root:

```bash
python3 tools/check_md_links.py
```

This is enforced in CI (`.github/workflows/python-check.yml`, job `markdown-links`).

## `check_links.py`

Checks **internal** markdown links:

- Relative file links (e.g. `../resources/foo.md`)
- Same-file anchors (e.g. `#table-of-contents`)
- Cross-file anchors (best-effort)

Run from repo root:

```bash
python tools/check_links.py README.md resources 00-prerequisites
```

