"""Reject staged content that looks like a credential.

Runs as a pre-commit hook over the staged files. Patterns cover the token
formats this project could plausibly touch: GitHub personal access tokens, AWS
access key identifiers, and PEM private keys.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = [
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
]

SKIP = {"check_secrets.py", "secrets.yml", ".pre-commit-config.yaml"}


def main(paths: list[str]) -> int:
    findings = []
    for name in paths:
        path = Path(name)
        if path.name in SKIP or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{path}:{number}: {pattern.pattern}")
                    break

    for finding in findings:
        print(finding, file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
