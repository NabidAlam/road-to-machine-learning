#!/usr/bin/env python3
"""
Script to check and fix table of contents links in markdown files.
"""
import os
import re
from pathlib import Path

def generate_anchor(text):
    """Generate GitHub-style anchor from heading text."""
    # Convert to lowercase
    anchor = text.lower()
    # Replace spaces with hyphens
    anchor = anchor.replace(' ', '-')
    # Remove special characters (keep hyphens and alphanumeric)
    anchor = re.sub(r'[^a-z0-9-]', '', anchor)
    # Remove multiple consecutive hyphens
    anchor = re.sub(r'-+', '-', anchor)
    # Remove leading/trailing hyphens
    anchor = anchor.strip('-')
    return anchor

def extract_headings(content):
    """Extract all headings from markdown content."""
    headings = []
    lines = content.split('\n')
    for i, line in enumerate(lines):
        # Match markdown headings (## or ###)
        match = re.match(r'^(#{2,3})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            anchor = generate_anchor(text)
            headings.append({
                'line': i + 1,
                'level': level,
                'text': text,
                'anchor': anchor
            })
    return headings

def extract_toc_links(content):
    """Extract TOC links from markdown content."""
    toc_links = []
    in_toc = False
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if '## Table of Contents' in line:
            in_toc = True
            continue
        if in_toc:
            if line.strip().startswith('---') or (line.strip().startswith('##') and not line.strip().startswith('## Table')):
                break
            # Match markdown links in TOC
            match = re.findall(r'\[([^\]]+)\]\(#([^\)]+)\)', line)
            for text, anchor in match:
                toc_links.append({
                    'line': i + 1,
                    'text': text,
                    'anchor': anchor
                })
    return toc_links

def check_file(filepath):
    """Check a single markdown file for TOC issues."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {'error': str(e)}
    
    # Check if file has TOC
    has_toc = '## Table of Contents' in content or '## Table of contents' in content
    
    if not has_toc:
        return {'has_toc': False, 'file': str(filepath)}
    
    headings = extract_headings(content)
    toc_links = extract_toc_links(content)
    
    # Check for broken links
    broken_links = []
    valid_anchors = {h['anchor'] for h in headings}
    
    for link in toc_links:
        if link['anchor'] not in valid_anchors:
            broken_links.append(link)
    
    return {
        'has_toc': True,
        'file': str(filepath),
        'headings_count': len(headings),
        'toc_links_count': len(toc_links),
        'broken_links': broken_links,
        'headings': headings[:10]  # First 10 for preview
    }

def main():
    """Main function to check all markdown files."""
    repo_root = Path('.')
    md_files = list(repo_root.rglob('*.md'))
    
    # Filter to main guide files (exclude READMEs, quick-refs, etc.)
    main_files = [
        f for f in md_files 
        if f.stat().st_size > 5000 and  # At least 5KB
        'README' not in f.name and
        'quick-reference' not in f.name and
        'project-tutorial' not in f.name and
        'advanced-topics' not in f.name
    ]
    
    print(f"Checking {len(main_files)} main guide files...\n")
    
    issues = []
    for filepath in sorted(main_files):
        result = check_file(filepath)
        if 'error' in result:
            print(f"ERROR in {result['file']}: {result['error']}")
            continue
        
        if not result['has_toc']:
            print(f"[!] {result['file']}: No TOC found")
            issues.append(result)
        elif result['broken_links']:
            print(f"[X] {result['file']}: {len(result['broken_links'])} broken TOC links")
            for link in result['broken_links']:
                print(f"   Line {link['line']}: {link['text']} -> #{link['anchor']}")
            issues.append(result)
        else:
            print(f"[OK] {result['file']}: TOC OK ({result['toc_links_count']} links)")
    
    print(f"\n\nSummary: {len([i for i in issues if i.get('broken_links')])} files with broken links")
    return issues

if __name__ == '__main__':
    main()
