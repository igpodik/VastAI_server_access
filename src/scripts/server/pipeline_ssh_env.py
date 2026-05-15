#!/usr/bin/env python3

import json
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse

_CONFIG = Path(__file__).resolve().parent / "config.json"
_SECRET_ENV = Path(__file__).resolve().parent / "secret.env"


def _strip_env_value(raw: str, prefix: str) -> str:
    line = raw.strip()
    if not line.startswith(prefix):
        return ""
    value = line.removeprefix(prefix).lstrip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip()


def load_ssh_key_path(secret_path: Path) -> str | None:
    """Путь к приватному ключу из SSH_KEY= в secret.env; None если строки нет."""
    if not secret_path.is_file():
        return None
    for raw_line in secret_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("SSH_KEY="):
            continue
        value = _strip_env_value(line, "SSH_KEY=")
        if not value:
            return None
        first = value.split(None, 1)[0]
        if first.startswith("ssh-"):
            print(
                "SSH_KEY в secret.env похож на публичный ключ; "
                "укажите путь к файлу приватного ключа (например ~/.ssh/id_ed25519).",
                file=sys.stderr,
            )
            sys.exit(1)
        p = Path(value).expanduser()
        if not p.is_file():
            print(f"Файл SSH_KEY не найден: {p}", file=sys.stderr)
            sys.exit(1)
        return str(p.resolve())
    return None


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
    identity = load_ssh_key_path(_SECRET_ENV)
    if identity:
        print(f"export PIPELINE_SSH_IDENTITY_FILE={shlex.quote(identity)}")


if __name__ == "__main__":
    main()
