#!/usr/bin/env python3
"""Печатает в stdout строки `export ...` для eval в pipeline.sh (SSH из config.json)."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse

_CONFIG = Path(__file__).resolve().parent / "config.json"


def main() -> None:
    data = json.loads(_CONFIG.read_text(encoding="utf-8"))
    url_s = data.get("SSH_URL")
    if not url_s:
        print("SSH_URL пустой в config.json (нужен vast_ai start).", file=sys.stderr)
        sys.exit(1)
    u = urlparse(str(url_s))
    if u.scheme != "ssh":
        print(f"Ожидался URL вида ssh://..., получено: {url_s!r}", file=sys.stderr)
        sys.exit(1)
    host = u.hostname or ""
    port = u.port or 22
    user = u.username or "root"
    if not host:
        print("В SSH_URL нет hostname.", file=sys.stderr)
        sys.exit(1)

    exp = data.get("EXPERIMENT_NAME")
    exp_s = "" if exp is None else str(exp)

    print(f"export PIPELINE_SSH_HOST={shlex.quote(host)}")
    print(f"export PIPELINE_SSH_PORT={shlex.quote(str(port))}")
    print(f"export PIPELINE_SSH_USER={shlex.quote(user)}")
    print(f"export PIPELINE_EXPERIMENT_NAME={shlex.quote(exp_s)}")


if __name__ == "__main__":
    main()
