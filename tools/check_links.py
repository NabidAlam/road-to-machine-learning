"""
Simple Markdown link checker (internal links).

Goal:
- Catch broken *relative* links in markdown files before they land in main.

What it checks:
- Relative file links like ../foo/bar.md or ./baz.md
- Same-file anchors like #some-heading
- Cross-file anchors like other.md#some-heading (best-effort)

What it ignores:
- http/https links
- mailto: links
- images are treated like links (still checked for file existence)

Usage (PowerShell):
  python tools/check_links.py README.md resources 00-prerequisites

Exit codes:
  0 = no issues
  1 = broken links found
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import unquote


LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class LinkIssue:
    file: Path
    link: str
    reason: str


def is_external(link: str) -> bool:
    lower = link.lower()
    return lower.startswith(("http://", "https://", "mailto:"))


def strip_title(link: str) -> str:
    # Handles markdown: [text](path "title here")
    # Also handles: [text](path 'title')
    link = link.strip()
    if " " not in link:
        return link
    # If the URL is enclosed with <...>, keep it as-is.
    if link.startswith("<") and link.endswith(">"):
        return link[1:-1].strip()
    return link.split(" ", 1)[0].strip()


def github_slugify(text: str) -> str:
    """
    Best-effort GitHub-style anchor slug.
    Not perfect for all unicode/punctuation cases, but good enough
    for catching most broken anchors in this repo.
    """
    s = text.strip().lower()
    # Remove formatting/backticks
    s = re.sub(r"[`*_~]", "", s)
    # Replace '&' with 'and' feels "book-like" but GitHub uses '-' in many cases.
    # We'll keep '&' as space-equivalent by removing it, resulting hyphens.
    s = s.replace("&", " ")
    # Drop anything that's not alnum/space/hyphen
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    # spaces -> hyphens, collapse repeats
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def collect_heading_slugs(md_text: str) -> set[str]:
    slugs: list[str] = []
    counts: dict[str, int] = {}

    for line in md_text.splitlines():
        m = HEADING_RE.match(line)
        if not m:
            continue
        heading_text = m.group(2).strip()
        base = github_slugify(heading_text)
        if not base:
            continue

        if base not in counts:
            counts[base] = 0
            slugs.append(base)
        else:
            counts[base] += 1
            slugs.append(f"{base}-{counts[base]}")

    return set(slugs)


def iter_md_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".md":
            yield p
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                # skip hidden dirs
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    if f.lower().endswith(".md"):
                        yield Path(root) / f


def extract_links(md_text: str) -> Iterator[str]:
    for match in LINK_RE.finditer(md_text):
        raw = match.group(1).strip()
        if not raw:
            continue
        raw = strip_title(raw)
        raw = raw.strip()
        # remove <...> wrapper
        if raw.startswith("<") and raw.endswith(">"):
            raw = raw[1:-1].strip()
        yield raw


def check_file(md_file: Path) -> list[LinkIssue]:
    issues: list[LinkIssue] = []
    try:
        text = md_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = md_file.read_text(encoding="utf-8", errors="replace")

    base_dir = md_file.parent

    # Slugs for same-file anchor checks
    self_slugs = collect_heading_slugs(text)

    for link in extract_links(text):
        if is_external(link):
            continue

        # Split path and anchor
        if "#" in link:
            path_part, anchor = link.split("#", 1)
            anchor = unquote(anchor).strip()
        else:
            path_part, anchor = link, ""

        # Same-file anchor
        if not path_part and anchor:
            if anchor not in self_slugs:
                issues.append(
                    LinkIssue(md_file, link, f"Anchor '#{anchor}' not found in file headings")
                )
            continue

        # Local file link
        target = (base_dir / path_part).resolve() if path_part else md_file.resolve()
        if path_part and not target.exists():
            issues.append(LinkIssue(md_file, link, f"Target path not found: {path_part}"))
            continue

        # Cross-file anchor check (only for .md targets)
        if anchor and target.suffix.lower() == ".md":
            try:
                ttext = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                ttext = target.read_text(encoding="utf-8", errors="replace")
            slugs = collect_heading_slugs(ttext)
            if anchor not in slugs:
                issues.append(
                    LinkIssue(md_file, link, f"Anchor '#{anchor}' not found in {target.name}")
                )

    return issues


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to scan (defaults to repo root).",
    )
    args = ap.parse_args(argv)

    root = Path.cwd().resolve()
    scan_paths = [Path(p).resolve() if Path(p).is_absolute() else (root / p).resolve() for p in args.paths]

    md_files = sorted(set(iter_md_files(scan_paths)))
    all_issues: list[LinkIssue] = []

    for f in md_files:
        all_issues.extend(check_file(f))

    if not all_issues:
        print(f"OK: {len(md_files)} markdown files scanned, no broken internal links found.")
        return 0

    print(f"ERROR: {len(all_issues)} broken links found across {len(md_files)} markdown files.")
    for issue in all_issues[:200]:
        rel = issue.file.relative_to(root) if issue.file.is_relative_to(root) else issue.file
        print(f"- {rel}: `{issue.link}` -> {issue.reason}")
    if len(all_issues) > 200:
        print(f"... and {len(all_issues) - 200} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

