#!/bin/sh
# Shared helpers for repository git hooks. Sourced, not executed.
# Runs a repository Python tool with the first interpreter that works:
# uv (no project sync), then python3, then python.

repo_root() {
  git rev-parse --show-toplevel
}

run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run --no-project --quiet python "$@"
    return $?
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 "$@"
    return $?
  fi
  if command -v python >/dev/null 2>&1; then
    python "$@"
    return $?
  fi
  echo "hook: no Python interpreter found (install uv)" >&2
  return 2
}
