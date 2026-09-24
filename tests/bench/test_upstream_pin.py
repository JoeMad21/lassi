"""Tests for the upstream LASSI pin and the HeCBench pin check (task P1.1).

Bible: Source Papers (LASSI: the upstream repository line; the quirk table's
entropy row), Repository Layout (`third_party/`), Benchmark Suites, Toolchain
Pins, Agent Rules 4 and 10.

Upstream LASSI is pinned by a manifest and a fetch tool, not by a git
submodule: a staged gitlink fails the text-policy check of staged paths, which
may not be changed to pass a commit (AGENTS.md). What these tests check:

- `assets/upstream/lassi.yaml` names url https://github.com/SPEAR-UIC/LASSI,
  commit 74b46812523f2ff79b53b6880a4521690d7478b0 (UPSTREAM_PIN, the full id
  of upstream's 74b4681), and path `third_party/LASSI`.
- `tools/fetch_upstream.py [MANIFEST] [--dest DIR]` fetches exactly that
  commit into DIR (default: the repository root joined with the manifest's
  path; default manifest: `assets/upstream/lassi.yaml`) and checks that HEAD
  is the pin. It exits 0 when DIR holds the pin, fetched or already there, and
  prints the pin; 1 when it refuses or the fetch fails; 2 on a manifest error.
  It refuses a checkout with local edits or untracked files, a checkout of
  another repository, and a non-empty directory that is not a checkout, and
  leaves each of them as it found them. It moves a clean checkout of the same
  repository at another commit to the pin, unless that checkout holds ignored
  files (refused and kept). A fetch that passes the tool's timeout
  (FETCH_TIMEOUT_S) fails with status 1 like any failed fetch. A checkout
  whose `.git` is a file (the layout a submodule clone leaves, as on the
  owner's workstation) is
  accepted, and nothing it points to is deleted. Rerunning at the pin changes
  nothing and needs no network. `load_pin(path)` returns an object with
  `url`, `commit`, and `path`.
- Those behaviors are checked against throwaway repositories in `tmp_path`
  (a local upstream with the pin behind its tip, and an outer repository that
  ignores `third_party/LASSI/` as this one does), so no network is used.
- Nothing is tracked under `third_party/`, the index holds no gitlink, and
  `third_party/LASSI/` is ignored. A local checkout, when present, is at the
  pin and unedited (`third_party/` is read-only); tests that need it skip
  with a reason that names the fetch tool.
- `plans/spikes/p1-hecbench-pin.md` holds one table row per upstream `*_main`
  file with the columns `Upstream file` (the path in backticks), `Blob id`
  (40 hex digits), `Held at 7d2d3c5` (yes or no), and `Model-facing source`
  (`upstream LASSI 74b4681` or `HeCBench <commit>`); a section
  `## HeCBench commits holding all 20` listing commit ids, or None; a section
  `## Support files` with a line `HeCBench pin for support files: <commit>`
  and the `src/...` paths of the support files in backticks (entropy's
  `reference.h` among them); fenced blocks with the commands and outputs,
  including a local run of the fetch tool and one through `tools/rx.py run`;
  and no PLACEHOLDER left. The blob ids are checked against the pinned
  upstream tree: git computes them from the file bytes, so they are content
  hashes, not measurements.
- OQ-018: no 40-character run of upstream's `prompt_dictionary.py` or of its
  notebook appears in the spike or in this task's other files.
- Remote tests (marked `remote`, skipped unless $LASSI_SCRATCH is set, as the
  build host's gate sets it) run the fetch tool against GitHub on the build
  host, and check the spike's HeCBench claims against a blobless clone of
  HeCBench made in a temporary directory under the scratch root; they fail
  before any fetch or clone when that directory is outside $LASSI_SCRATCH
  (Agent Rule 7).

No upstream code runs here; git only reads trees and checks out files. No
value here is a measurement.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_PATH = "third_party/LASSI"
UPSTREAM_URL = "https://github.com/SPEAR-UIC/LASSI"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
HECBENCH_URL = "https://github.com/zjin-lcf/HeCBench"
HECBENCH_PIN = "7d2d3c567be522a2104065165de0a4a233a6ea1a"
TOOL = REPO / "tools" / "fetch_upstream.py"
UPSTREAM_MANIFEST = REPO / "assets" / "upstream" / "lassi.yaml"
SPIKE = REPO / "plans" / "spikes" / "p1-hecbench-pin.md"
APPS = (
    "atomicCost", "bsearch", "colorwheel", "dense-embedding", "entropy",
    "jacobi", "layout", "matrix-rotate", "pathfinder", "randomAccess",
)
EXTENSIONS = {"omp": "cpp", "cuda": "cu"}
MAIN_RE = re.compile(r"`(?:[\w./-]*/)?(?P<app>[A-Za-z-]+)-(?P<lang>omp|cuda)_main\.(?P<ext>cpp|cu)`")
SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
SOURCE_RE = re.compile(r"^(?:upstream LASSI 74b4681|HeCBench (?P<commit>[0-9a-f]{7,40}))$")
FETCH_HINT = "run `uv run python tools/fetch_upstream.py` to fetch the pinned upstream"
ON_HOST = bool(os.environ.get("LASSI_SCRATCH"))
HOST_REASON = "needs the build host (LASSI_SCRATCH unset); run it through `uv run tools/rx.py run`"
IDENTITY = {
    "GIT_AUTHOR_NAME": "Pin Test",
    "GIT_AUTHOR_EMAIL": "pin-test@example.invalid",
    "GIT_COMMITTER_NAME": "Pin Test",
    "GIT_COMMITTER_EMAIL": "pin-test@example.invalid",
}


def git(*args: str, cwd: Path = REPO, check: bool = True) -> subprocess.CompletedProcess:
    """Run git with `args` in `cwd` and return the finished process (text mode)."""
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check)


def fixture_git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run git in a throwaway repository with the test identity and unsigned commits."""
    env = {**os.environ, **IDENTITY}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", "-C", str(cwd), *args],
                          capture_output=True, text=True, check=check, env=env)


def head(repo: Path) -> str:
    """Return the full id of HEAD in `repo`."""
    return fixture_git(repo, "rev-parse", "HEAD").stdout.strip()


# ---------------------------------------------------------------- the tool, on throwaway repositories


@dataclass
class Upstream:
    """A local stand-in for upstream: its file URL, an older commit, the pin, and a newer tip."""

    root: Path
    url: str
    old: str
    pin: str
    tip: str


def commit_files(repo: Path, files: dict[str, str], message: str) -> str:
    """Write `files` (relative path to text) into `repo`, commit them, and return the new commit id."""
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="ascii")
    fixture_git(repo, "add", "-A")
    fixture_git(repo, "commit", "--quiet", "-m", message)
    return head(repo)


def make_upstream(root: Path, name: str) -> Upstream:
    """Create a repository with three commits; the middle one plays the pin, so the pin is not the tip."""
    repo = root / name
    repo.mkdir(parents=True)
    fixture_git(repo, "init", "--quiet", "-b", "main")
    fixture_git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    old = commit_files(repo, {"README.md": "first\n"}, "first")
    pin = commit_files(repo, {"README.md": "second\n", "src/main.cpp": "int main() { return 0; }\n"}, "second")
    tip = commit_files(repo, {"README.md": "third\n"}, "third")
    return Upstream(root=repo, url=repo.as_uri(), old=old, pin=pin, tip=tip)


@pytest.fixture
def upstream(tmp_path: Path) -> Upstream:
    """Return a local upstream repository in `tmp_path`."""
    return make_upstream(tmp_path, "upstream")


@pytest.fixture
def outer(tmp_path: Path) -> Path:
    """Return an outer repository that ignores `third_party/LASSI/`, standing in for this one."""
    repo = tmp_path / "outer"
    repo.mkdir()
    fixture_git(repo, "init", "--quiet", "-b", "main")
    commit_files(repo, {".gitignore": "third_party/LASSI/\n"}, "outer")
    return repo


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """Load tools/fetch_upstream.py as a module (registered first, as dataclasses and pickling expect)."""
    assert TOOL.is_file(), "tools/fetch_upstream.py is missing (task P1.1 writes it)"
    spec = importlib.util.spec_from_file_location("fetch_upstream_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_manifest(path: Path, url: str, commit: str, rel: str = UPSTREAM_PATH, drop: str = "") -> Path:
    """Write a pin manifest with `url`, `commit`, and `path` (leaving out the key `drop`) and return it."""
    values = {"url": url, "commit": commit, "path": rel}
    path.write_text("".join(f"{key}: {json.dumps(value)}\n" for key, value in values.items() if key != drop),
                    encoding="ascii")
    return path


def run_tool(tool: ModuleType, capsys: pytest.CaptureFixture, *argv: str) -> tuple[int, str]:
    """Run the tool's main in process and return its status and its stdout plus stderr."""
    status = tool.main(list(argv))
    captured = capsys.readouterr()
    return status, captured.out + captured.err


def dest_of(outer: Path) -> Path:
    """Return the checkout path inside the outer repository."""
    return outer / "third_party" / "LASSI"


def clone_at(upstream: Upstream, dest: Path, commit: str) -> None:
    """Clone `upstream` into `dest` and detach HEAD at `commit`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fixture_git(dest.parent, "clone", "--quiet", upstream.url, dest.name)
    fixture_git(dest, "checkout", "--quiet", "--detach", commit)


def submodule_style_checkout(outer: Path, upstream: Upstream, commit: str) -> Path:
    """Leave the layout an abandoned submodule leaves: a `.git` file into `.git/modules`, no index entry."""
    fixture_git(outer, "-c", "protocol.file.allow=always", "submodule", "add", "--quiet", "--force", upstream.url,
                UPSTREAM_PATH)
    dest = dest_of(outer)
    fixture_git(dest, "checkout", "--quiet", "--detach", commit)
    fixture_git(outer, "rm", "--cached", "--quiet", "--force", UPSTREAM_PATH, ".gitmodules")
    (outer / ".gitmodules").unlink()
    assert (dest / ".git").is_file(), "the fixture's checkout has a .git file"
    return dest


def snapshot(dest: Path) -> tuple[str, str, str, list[tuple[str, int, int]]]:
    """Return HEAD, the refs, the status, and (path, mtime, size) of every worktree file of `dest`."""
    files = sorted(
        (path.relative_to(dest).as_posix(), path.stat().st_mtime_ns, path.stat().st_size)
        for path in dest.rglob("*")
        if path.is_file() and path.relative_to(dest).parts[0] != ".git"
    )
    refs = fixture_git(dest, "for-each-ref").stdout
    return head(dest), refs, fixture_git(dest, "status", "--porcelain").stdout, files


def assert_at_pin(dest: Path, pin: str) -> None:
    """Assert that `dest` is its own repository, at `pin`, with a clean tree."""
    top = Path(fixture_git(dest, "rev-parse", "--show-toplevel").stdout.strip())
    assert top.resolve() == dest.resolve(), f"{dest} is not its own repository (git resolved {top})"
    assert head(dest) == pin
    assert fixture_git(dest, "status", "--porcelain").stdout == ""


def outer_state(outer: Path) -> tuple[str, str, str]:
    """Return the outer repository's HEAD, remotes, and status, which the tool must never change."""
    return head(outer), fixture_git(outer, "remote", "-v").stdout, fixture_git(outer, "status", "--porcelain").stdout


def test_fetch_script_checks_out_exactly_the_pin_into_a_missing_directory(
        tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    assert TOOL.is_file(), "tools/fetch_upstream.py is missing (task P1.1 writes it)"
    manifest = write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)
    dest = dest_of(outer)
    before = outer_state(outer)
    done = subprocess.run([sys.executable, str(TOOL), str(manifest), "--dest", str(dest)],
                          capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stdout + done.stderr
    assert upstream.pin in done.stdout, "the tool prints the commit it holds"
    assert_at_pin(dest, upstream.pin)
    assert (dest / "README.md").read_text(encoding="ascii") == "second\n", "the pin, not upstream's tip"
    assert outer_state(outer) == before


def test_rerun_at_the_pin_changes_nothing_and_needs_no_network(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    manifest = write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)
    dest = dest_of(outer)
    assert run_tool(tool, capsys, str(manifest), "--dest", str(dest))[0] == 0
    before = snapshot(dest)
    upstream.root.rename(upstream.root.with_name("upstream-gone"))
    status, output = run_tool(tool, capsys, str(manifest), "--dest", str(dest))
    assert status == 0, output
    assert upstream.pin in output
    assert snapshot(dest) == before


def test_refuses_a_checkout_with_local_edits_and_keeps_them(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    dest = dest_of(outer)
    clone_at(upstream, dest, upstream.pin)
    (dest / "README.md").write_text("edited here\n", encoding="ascii")
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 1, output
    assert (dest / "README.md").read_text(encoding="ascii") == "edited here\n"
    assert head(dest) == upstream.pin


def test_refuses_a_checkout_with_untracked_files_and_keeps_them(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    dest = dest_of(outer)
    clone_at(upstream, dest, upstream.pin)
    (dest / "notes.txt").write_text("mine\n", encoding="ascii")
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 1, output
    assert (dest / "notes.txt").read_text(encoding="ascii") == "mine\n"


def test_moves_a_clean_checkout_at_another_commit_to_the_pin(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    dest = dest_of(outer)
    clone_at(upstream, dest, upstream.old)
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 0, output
    assert_at_pin(dest, upstream.pin)


def test_refuses_a_dirty_checkout_at_another_commit(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    dest = dest_of(outer)
    clone_at(upstream, dest, upstream.old)
    (dest / "README.md").write_text("edited here\n", encoding="ascii")
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 1, output
    assert head(dest) == upstream.old
    assert (dest / "README.md").read_text(encoding="ascii") == "edited here\n"


def test_refuses_to_move_a_checkout_holding_ignored_files_and_keeps_them(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    dest = dest_of(outer)
    clone_at(upstream, dest, upstream.old)
    exclude = dest / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("build/\n", encoding="ascii")
    (dest / "build").mkdir()
    (dest / "build" / "out.o").write_text("mine\n", encoding="ascii")
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 1, output
    assert head(dest) == upstream.old, "a checkout holding ignored files is not moved"
    assert (dest / "build" / "out.o").read_text(encoding="ascii") == "mine\n"


def test_a_fetch_that_times_out_fails_naming_the_pin(
        tool: ModuleType, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        upstream: Upstream, outer: Path) -> None:
    monkeypatch.setattr(tool, "FETCH_TIMEOUT_S", 1e-6, raising=False)
    dest = dest_of(outer)
    before = outer_state(outer)
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 1, f"a stalled fetch ends in a refusal, not a hang or a traceback: {output}"
    assert upstream.pin in output
    assert "timed out" in output
    assert outer_state(outer) == before
    assert fixture_git(dest, "rev-parse", "--verify", "--quiet", "HEAD", check=False).returncode != 0, (
        "no commit is checked out when the fetch times out")


def test_refuses_a_checkout_of_another_repository(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    other = make_upstream(tmp_path, "other")
    dest = dest_of(outer)
    clone_at(other, dest, other.pin)
    before = snapshot(dest)
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 1, output
    assert snapshot(dest) == before
    assert fixture_git(dest, "remote", "get-url", "origin").stdout.strip() == other.url


def test_refuses_a_non_empty_directory_that_is_not_a_checkout(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    dest = dest_of(outer)
    dest.mkdir(parents=True)
    (dest / "notes.txt").write_text("mine\n", encoding="ascii")
    before = outer_state(outer)
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 1, output
    assert sorted(path.name for path in dest.iterdir()) == ["notes.txt"], "the directory is left as it was"
    assert outer_state(outer) == before, "git commands reached the outer repository"


def test_accepts_a_submodule_style_checkout_at_the_pin_unchanged(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    dest = submodule_style_checkout(outer, upstream, upstream.pin)
    gitfile = (dest / ".git").read_text(encoding="utf-8")
    before = snapshot(dest)
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 0, output
    assert (dest / ".git").is_file() and (dest / ".git").read_text(encoding="utf-8") == gitfile
    assert (outer / ".git" / "modules" / "third_party" / "LASSI").is_dir()
    assert snapshot(dest) == before


def test_moves_a_submodule_style_checkout_at_another_commit_to_the_pin(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    dest = submodule_style_checkout(outer, upstream, upstream.old)
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin)),
                              "--dest", str(dest))
    assert status == 0, output
    assert_at_pin(dest, upstream.pin)
    assert (outer / ".git" / "modules" / "third_party" / "LASSI").is_dir(), "nothing the old checkout used is deleted"


def test_fails_naming_a_pinned_commit_that_upstream_lacks(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    missing = "1234567890abcdef1234567890abcdef12345678"
    dest = dest_of(outer)
    before = outer_state(outer)
    status, output = run_tool(tool, capsys, str(write_manifest(tmp_path / "pin.yaml", upstream.url, missing)),
                              "--dest", str(dest))
    assert status == 1, output
    assert missing in output
    assert outer_state(outer) == before
    if (dest / ".git").exists():
        assert fixture_git(dest, "rev-parse", "--verify", "--quiet", "HEAD", check=False).returncode != 0, (
            "no commit is checked out when the pin cannot be fetched")


def test_refuses_a_manifest_without_a_commit(
        tool: ModuleType, capsys: pytest.CaptureFixture, tmp_path: Path, upstream: Upstream, outer: Path) -> None:
    manifest = write_manifest(tmp_path / "pin.yaml", upstream.url, upstream.pin, drop="commit")
    status, output = run_tool(tool, capsys, str(manifest), "--dest", str(dest_of(outer)))
    assert status == 2, output
    assert "commit" in output
    assert not dest_of(outer).exists()


# ---------------------------------------------------------------- the pin in this repository


def test_manifest_pins_upstream_lassi_at_74b4681() -> None:
    assert UPSTREAM_MANIFEST.is_file(), "assets/upstream/lassi.yaml is missing (task P1.1 writes it)"
    data = yaml.safe_load(UPSTREAM_MANIFEST.read_text(encoding="utf-8"))
    assert data["url"] in (UPSTREAM_URL, UPSTREAM_URL + ".git")
    assert data["commit"] == UPSTREAM_PIN
    assert data["path"] == UPSTREAM_PATH


def test_load_pin_reads_the_manifest(tool: ModuleType) -> None:
    pin = tool.load_pin(UPSTREAM_MANIFEST)
    assert (pin.url.removesuffix(".git"), pin.commit, pin.path) == (UPSTREAM_URL, UPSTREAM_PIN, UPSTREAM_PATH)


def test_nothing_is_tracked_under_third_party_and_no_gitlink_exists() -> None:
    assert git("ls-files", "-s", "--", "third_party").stdout == "", "third_party/ is fetched, never tracked"
    gitlinks = [line for line in git("ls-files", "-s").stdout.splitlines() if line.startswith("160000 ")]
    assert not gitlinks, f"the index holds a gitlink: {gitlinks}"
    assert git("ls-files", "--", ".gitmodules").stdout == "", "no submodule is declared"


def test_the_upstream_checkout_is_ignored() -> None:
    done = git("check-ignore", "--no-index", "--quiet", f"{UPSTREAM_PATH}/README.md", check=False)
    assert done.returncode == 0, f".gitignore does not ignore {UPSTREAM_PATH}/"


def need_checkout() -> Path:
    """Return the local upstream checkout, or skip naming the fetch tool when it or the pin is absent."""
    root = REPO / UPSTREAM_PATH
    if not (root / ".git").exists():
        pytest.skip(f"{UPSTREAM_PATH} is not checked out; {FETCH_HINT}")
    if git("cat-file", "-e", f"{UPSTREAM_PIN}^{{commit}}", cwd=root, check=False).returncode != 0:
        pytest.skip(f"{UPSTREAM_PATH} lacks the pinned commit; {FETCH_HINT}")
    return root


def test_local_checkout_is_at_the_pin_and_unedited() -> None:
    root = REPO / UPSTREAM_PATH
    if not (root / ".git").exists():
        pytest.skip(f"{UPSTREAM_PATH} is not checked out; {FETCH_HINT}")
    top = Path(git("rev-parse", "--show-toplevel", cwd=root).stdout.strip())
    assert top.resolve() == root.resolve(), f"{UPSTREAM_PATH} is not its own repository"
    assert git("rev-parse", "HEAD", cwd=root).stdout.strip() == UPSTREAM_PIN, FETCH_HINT
    assert git("status", "--porcelain", cwd=root).stdout == "", "third_party/ is read-only"


def test_default_run_keeps_the_local_checkout_at_the_pin(tool: ModuleType, capsys: pytest.CaptureFixture) -> None:
    root = need_checkout()
    if git("rev-parse", "HEAD", cwd=root).stdout.strip() != UPSTREAM_PIN or git(
            "status", "--porcelain", cwd=root).stdout:
        pytest.skip(f"{UPSTREAM_PATH} is not a clean checkout of the pin; {FETCH_HINT}")
    dotgit = root / ".git"
    kind = (dotgit.is_file(), dotgit.read_text(encoding="utf-8") if dotgit.is_file() else "")
    before = snapshot(root)
    status, output = run_tool(tool, capsys)
    assert status == 0, output
    assert UPSTREAM_PIN in output
    assert (dotgit.is_file(), dotgit.read_text(encoding="utf-8") if dotgit.is_file() else "") == kind
    assert snapshot(root) == before


def upstream_mains(root: Path) -> dict[tuple[str, str], str]:
    """Return {(app, lang): blob id} for the `*_main` files in the pinned upstream tree."""
    out = git("ls-tree", "-r", UPSTREAM_PIN, cwd=root).stdout
    found: dict[tuple[str, str], str] = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        match = MAIN_RE.search(f"`{path}`")
        if match:
            found[(match["app"], match["lang"])] = meta.split()[2]
    return found


def expected_mains() -> set[tuple[str, str]]:
    """Return the (app, language) pairs of the 20 upstream `*_main` files."""
    return {(app, lang) for app in APPS for lang in EXTENSIONS}


def test_pinned_upstream_holds_the_twenty_main_files() -> None:
    assert set(upstream_mains(need_checkout())) == expected_mains()


# ---------------------------------------------------------------- the spike


def read_spike() -> str:
    """Return the spike text, failing with a clear message when it is missing."""
    assert SPIKE.is_file(), f"{SPIKE.relative_to(REPO).as_posix()} is missing (task P1.1 writes it)"
    return SPIKE.read_text(encoding="utf-8")


def cells(line: str) -> list[str]:
    """Split a markdown table line into stripped cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def spike_rows(text: str) -> dict[tuple[str, str], dict[str, str]]:
    """Parse the spike's file table into {(app, lang): {column: cell}} keyed by the header names."""
    header: list[str] = []
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            header = [] if not line.strip() else header
            continue
        row = cells(line)
        if "Blob id" in row and not header:
            header = row
            continue
        match = next((MAIN_RE.search(cell) for cell in row if MAIN_RE.search(cell)), None)
        if header and match:
            key = (match["app"], match["lang"])
            assert key not in rows, f"the spike lists {key} twice"
            assert match["ext"] == EXTENSIONS[match["lang"]], f"{match[0]} has the wrong extension"
            assert len(row) == len(header), f"{key}: {len(row)} cells under {len(header)} columns"
            rows[key] = dict(zip(header, (cell.strip("`") for cell in row), strict=True))
    return rows


def section(text: str, title: str) -> str:
    """Return the body of the level-2 section named `title` (without its heading)."""
    match = re.search(rf"^## {re.escape(title)}\s*$(?P<body>.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"the spike has no '## {title}' section"
    return match["body"]


def column(row: dict[str, str], word: str) -> str:
    """Return the cell of the first column whose header contains `word`."""
    for name, value in row.items():
        if word in name:
            return value
    raise AssertionError(f"no column containing {word!r} in {sorted(row)}")


def fenced_blocks(text: str) -> list[str]:
    """Return the bodies of the text's fenced blocks."""
    return re.findall(r"^```[^\n]*\n(.*?)^```", text, re.M | re.S)


def test_spike_lists_every_main_file_with_blob_id_and_pin_column() -> None:
    rows = spike_rows(read_spike())
    assert set(rows) == expected_mains(), f"missing: {sorted(expected_mains() - set(rows))}"
    for key, row in rows.items():
        assert re.fullmatch(r"[0-9a-f]{40}", column(row, "Blob id")), f"{key}: no 40-digit blob id"
        assert column(row, "7d2d3c5") in ("yes", "no"), f"{key}: the 7d2d3c5 column says yes or no"
        assert SOURCE_RE.match(column(row, "Model-facing source")), f"{key}: the model-facing source is unnamed"


def test_spike_blob_ids_match_the_pinned_upstream_tree() -> None:
    root = need_checkout()
    rows = spike_rows(read_spike())
    listed = {key: column(row, "Blob id") for key, row in rows.items()}
    assert listed == upstream_mains(root)


def test_spike_lists_hecbench_commits_holding_all_twenty() -> None:
    body = section(read_spike(), "HeCBench commits holding all 20")
    assert SHA_RE.search(body) or re.search(r"\bNone\b", body), "list the commits, or say None"


def support_pin(body: str) -> str:
    """Return the commit on the section's `HeCBench pin for support files: <commit>` line."""
    match = re.search(r"^HeCBench pin for support files: `?(?P<commit>[0-9a-f]{40})`?", body, re.M)
    assert match, "name the HeCBench commit the support files come from"
    return match["commit"]


def test_spike_names_the_support_file_pin_and_reference_h() -> None:
    body = section(read_spike(), "Support files")
    support_pin(body)
    assert re.search(r"`src/entropy-(?:omp|cuda)/reference\.h`", body), "entropy's reference.h is named"


def test_spike_records_commands_and_outputs() -> None:
    blocks = fenced_blocks(read_spike())
    joined = "\n".join(blocks)
    assert len(blocks) >= 2, "commands and outputs go in fenced blocks"
    for needle in ("git ", "ls-tree", "74b4681", "7d2d3c5"):
        assert needle in joined, f"no fenced command or output mentions {needle!r}"


def test_spike_records_a_local_run_of_the_fetch_tool() -> None:
    runs = [block for block in fenced_blocks(read_spike())
            if re.search(r"^\$ uv run python tools/fetch_upstream\.py", block, re.M)]
    assert runs, "a fenced block holds `$ uv run python tools/fetch_upstream.py` and its output"
    assert any(UPSTREAM_PIN in block for block in runs), "the recorded output shows the full pin"


def test_spike_records_the_fetch_tool_run_on_the_build_host() -> None:
    runs = [block for block in fenced_blocks(read_spike())
            if re.search(r"^\$ uv run tools/rx\.py run -- .*fetch_upstream\.py", block, re.M)]
    assert runs, "a fenced block holds the `rx.py run` of the fetch tool and its output"
    assert any(UPSTREAM_PIN in block and re.search(r"^\[rx\] id=\S+ rc=0 ", block, re.M) for block in runs), (
        "the recorded host run shows the pin, its rx id, and rc=0")


def test_spike_has_no_placeholder_left() -> None:
    lines = [line for line in read_spike().splitlines() if "PLACEHOLDER" in line]
    assert not lines, f"record the real command output in place of: {lines}"


def upstream_texts(root: Path) -> list[str]:
    """Return the string constants of prompt_dictionary.py and the notebook's cell sources at the pin."""
    source = git("show", f"{UPSTREAM_PIN}:prompt_dictionary.py", cwd=root).stdout
    texts = [node.value for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    notebook = json.loads(git("show", f"{UPSTREAM_PIN}:LASSI_pipeline_v0.ipynb", cwd=root).stdout)
    texts.extend("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    return texts


def repeated_upstream_text(text: str, root: Path) -> list[str]:
    """Return the 40-character runs of upstream prompt or notebook text that `text` repeats."""
    windows = {text[i:i + 40] for i in range(len(text) - 39)}
    return [part[i:i + 40] for part in upstream_texts(root) for i in range(len(part) - 39)
            if part[i:i + 40] in windows]


def test_spike_quotes_no_upstream_prompt_or_notebook_text() -> None:
    hits = repeated_upstream_text(read_spike(), need_checkout())
    assert not hits, f"the spike repeats upstream text (OQ-018): {hits[0]!r}"


def test_task_files_quote_no_upstream_prompt_or_notebook_text() -> None:
    root = need_checkout()
    for path in (Path(__file__), TOOL, UPSTREAM_MANIFEST):
        if path.is_file():
            hits = repeated_upstream_text(path.read_text(encoding="utf-8"), root)
            assert not hits, f"{path.relative_to(REPO).as_posix()} repeats upstream text (OQ-018): {hits[0]!r}"


# ---------------------------------------------------------------- remote


def under_scratch(path: Path) -> Path:
    """Return `path` after checking that it lies under $LASSI_SCRATCH, where the gate puts TMPDIR (Agent Rule 7)."""
    scratch = Path(os.environ["LASSI_SCRATCH"]).resolve()
    assert path.resolve().is_relative_to(scratch), (
        f"{path} is outside $LASSI_SCRATCH ({scratch}); run remote tests through `uv run tools/rx.py` (Agent Rule 7)")
    return path


@pytest.mark.remote
@pytest.mark.skipif(not ON_HOST, reason=HOST_REASON)
def test_fetch_tool_fetches_the_pin_from_github_on_the_host(tmp_path: Path) -> None:
    dest = under_scratch(tmp_path) / "LASSI"
    done = subprocess.run([sys.executable, str(TOOL), str(UPSTREAM_MANIFEST), "--dest", str(dest)],
                          capture_output=True, text=True, cwd=REPO, timeout=900)
    assert done.returncode == 0, (done.stdout + done.stderr)[-2000:]
    assert git("rev-parse", "HEAD", cwd=dest).stdout.strip() == UPSTREAM_PIN
    assert git("status", "--porcelain", cwd=dest).stdout == ""
    assert set(upstream_mains(dest)) == expected_mains()


@pytest.mark.remote
@pytest.mark.skipif(not ON_HOST, reason=HOST_REASON)
def test_fetch_tool_default_run_checks_out_the_pin_on_the_host() -> None:
    for attempt in ("first", "rerun"):
        done = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True, cwd=REPO, timeout=900)
        assert done.returncode == 0, f"{attempt}: {(done.stdout + done.stderr)[-2000:]}"
    root = REPO / UPSTREAM_PATH
    top = Path(git("rev-parse", "--show-toplevel", cwd=root).stdout.strip())
    assert top.resolve() == root.resolve()
    assert git("rev-parse", "HEAD", cwd=root).stdout.strip() == UPSTREAM_PIN
    assert git("status", "--porcelain", cwd=root).stdout == ""


@pytest.fixture(scope="module")
def hecbench(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Clone HeCBench without blobs or a checkout (trees are enough for blob ids) and return its path."""
    if not ON_HOST:
        pytest.skip(HOST_REASON)
    under_scratch(Path(tmp_path_factory.getbasetemp()))
    dest = tmp_path_factory.mktemp("hecbench") / "hecbench"
    subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", HECBENCH_URL, str(dest)],
                   check=True, capture_output=True, text=True, timeout=1800)
    return dest


def objects_at(clone: Path, specs: list[str]) -> list[str]:
    """Return the object id each `<commit>:<path>` names, or '' when absent, from one cat-file call.

    Only trees are read, so the blobless clone never fetches a blob.
    """
    done = subprocess.run(["git", "-C", str(clone), "cat-file", "--batch-check=%(objectname)"],
                          input="".join(f"{spec}\n" for spec in specs), capture_output=True, text=True, check=True)
    found = [line if re.fullmatch(r"[0-9a-f]{40}", line) else "" for line in done.stdout.splitlines()]
    assert len(found) == len(specs), "cat-file answered every spec"
    return found


def main_path(app: str, lang: str) -> str:
    """Return the HeCBench path of an app's main file in one language."""
    return f"src/{app}-{lang}/main.{EXTENSIONS[lang]}"


def holds_all(clone: Path, commit: str, rows: dict[tuple[str, str], dict[str, str]]) -> list[tuple[str, str]]:
    """Return the (app, lang) pairs whose upstream blob `commit` does not hold at its HeCBench path."""
    keys = sorted(rows)
    found = objects_at(clone, [f"{commit}:{main_path(*key)}" for key in keys])
    return [key for key, blob in zip(keys, found, strict=True) if blob != column(rows[key], "Blob id")]


@pytest.mark.remote
def test_spike_hecbench_claims_hold_in_a_hecbench_clone(hecbench: Path) -> None:
    text = read_spike()
    rows = spike_rows(text)
    assert set(rows) == expected_mains()
    lacking_at_pin = set(holds_all(hecbench, HECBENCH_PIN, rows))
    for key, row in rows.items():
        assert column(row, "7d2d3c5") == ("no" if key in lacking_at_pin else "yes"), f"{key}: wrong 7d2d3c5 claim"
        source = SOURCE_RE.match(column(row, "Model-facing source"))
        if source and source["commit"]:
            assert not holds_all(hecbench, source["commit"], {key: row}), f"{key}: its source lacks the blob"
    listed = SHA_RE.findall(section(text, "HeCBench commits holding all 20"))
    for commit in listed:
        assert not holds_all(hecbench, commit, rows), f"{commit} does not hold all 20"
    support = section(text, "Support files")
    pin = support_pin(support)
    paths = re.findall(r"`(src/[^`]+)`", support)
    missing = [path for path, oid in zip(paths, objects_at(hecbench, [f"{pin}:{p}" for p in paths]), strict=True)
               if not oid]
    assert paths and not missing, f"support files absent at {pin}: {missing}"
