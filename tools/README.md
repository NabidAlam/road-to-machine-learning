# Tools

Small maintenance utilities for keeping the repository consistent over time.

## `check_links.py`

Checks **internal** markdown links:

- Relative file links (e.g. `../resources/foo.md`)
- Same-file anchors (e.g. `#table-of-contents`)
- Cross-file anchors (best-effort)

Run from repo root:

```bash
python tools/check_links.py README.md resources 00-prerequisites
```

