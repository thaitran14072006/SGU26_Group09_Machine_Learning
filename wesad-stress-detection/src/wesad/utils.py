"""Đọc cấu hình, tính mã băm, ghi file và lưu thông tin phiên chạy."""

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_config(path, local=None):
    config = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    if local:
        overrides = yaml.safe_load(resolve_path(local).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or set(overrides) - {"data_root"}:
            raise ValueError("Local config may contain only data_root.")
        config.update(overrides)
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash():
    paths = sorted((PROJECT_ROOT / "src/wesad").rglob("*.py"))
    paths += sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    return fingerprint({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in paths})


def environment():
    versions = {}
    for name in ("numpy", "pandas", "PyYAML", "scipy", "scikit-learn", "imbalanced-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "packages": versions}


def provenance():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, timeout=10, check=True).stdout)
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return {"git_commit": head, "git_dirty": dirty, "code_sha256": code_hash()}


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def write_json(path, value):
    atomic_write(path, lambda temp: temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, frame):
    atomic_write(path, lambda temp: frame.to_csv(temp, index=False))


def log_event(folder, message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    with (Path(folder) / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(message, flush=True)


def import_callable(spec):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("Plugin must use 'module:function'.")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError(f"Not callable: {spec}")
    return function
