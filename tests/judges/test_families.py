"""Tests for the model family table, assets/models/families.yaml, and its reader, lassi.judges.families (task P15.1).

Bible: Agents And Judges (a judge uses a different model family from the
generator in the same trial, and both are recorded), Model Serving
(Serving Rules, the reasoning effort; OQ-080), Readability Standards (plain
ASCII). Plan: plans/p15-judges.md, P15.1, and its planning decision
"Model families (P15.1)".

The contract these tests fix:

- FAMILIES_FILE is the repository's assets/models/families.yaml.
  load_families(path) reads `path`, defaulting to FAMILIES_FILE as the
  module holds it at the call, and returns a FamilyTable: families (a
  tuple of the family names, as listed), reasoning_effort (a frozenset of
  the families whose models take a reasoning effort), and models (a
  read-only mapping of each model id to its family). family(model_id)
  gives a listed id's family and None for any other, exactly as written
  (no case folding, no prefix stripped).
- The file is plain ASCII YAML holding exactly the keys families (a
  non-empty list of distinct names: a lowercase letter, then lowercase
  letters, digits, and hyphens), reasoning_effort (a list of distinct
  listed families), and models (a non-empty mapping of non-empty model ids
  to listed families), with no repeated key, so a model id written twice
  is refused rather than read last-wins. Anything else is refused with
  FamilyError, a ValueError, naming the file and the key.
- The table lists every model id this repository's recipes name (each
  run recipe under projects/: its model.id under a served or in-process
  backend, each arm, and each judge's model; and the same in the bible's
  recipe blocks, with a train block's base_model) and every model id of
  plans/runs/rngd-sweep/models*.tsv, each mapped to the family its id
  names; the families of the owner's judge models (OQ-089) are llama and
  gpt-oss, and gpt-oss takes a reasoning effort.

Nothing here is a measurement.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
TABLE = REPO / "assets" / "models" / "families.yaml"
FIXTURE_TABLE = REPO / "tests" / "fixtures" / "judges" / "families.yaml"
SWEEP_TABLES = sorted((REPO / "plans" / "runs" / "rngd-sweep").glob("models*.tsv"))
# Backends whose model id names no model: a scripted reply list or a recording answers them.
SCRIPTED_BACKENDS = frozenset({"mock", "replay"})
# Each family, read from the ids (the plan's decision): one id per family the repository names.
EXPECTED = {
    "llama-3.3-70b": "llama",
    "furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic": "llama",
    "furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic": "llama",
    "gpt-oss-120b": "gpt-oss",
    "furiosa-ai/gpt-oss-20b": "gpt-oss",
    "furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8": "qwen",
    "registry:qwen3-coder-30b-gspo-lora-r8": "qwen",
    "furiosa-ai/EXAONE-4.0-32B-FP8": "exaone",
    "furiosa-ai/gemma-4-31B-it-FP8-block": "gemma",
    "furiosa-ai/Mistral-Nemo-Instruct-2407-FP8": "mistral",
    "wizardcoder-33b-fxb": "wizardcoder",
}
GOOD = "families: [alpha, beta]\nreasoning_effort: [beta]\nmodels:\n  m-1: alpha\n  m-2: beta\n"


@pytest.fixture(scope="module")
def families_module() -> ModuleType:
    """Import lassi.judges.families, failing with the task's name when it does not exist yet."""
    try:
        return importlib.import_module("lassi.judges.families")
    except ModuleNotFoundError as error:  # pragma: no cover - only before the task's module exists
        pytest.fail(f"lassi.judges.families does not exist (task P15.1): {error}")


def recipe_model_ids() -> dict[str, str]:
    """Return each model id the recipes under projects/ name, mapped to the first file that names it."""
    found: dict[str, str] = {}
    for path in sorted((REPO / "projects").rglob("*.yaml")):
        data = yaml.safe_load(path.read_bytes().decode("utf-8"))
        if not isinstance(data, dict) or "base_model" in data:
            continue
        for model_id in ids_of(data):
            found.setdefault(model_id, path.relative_to(REPO).as_posix())
    return found


def ids_of(data: dict) -> list[str]:
    """Return the model ids one run recipe mapping names: model.id (served or in-process), arms, and judge models."""
    ids: list[str] = []
    model = data.get("model")
    if isinstance(model, dict) and "id" in model and model.get("backend") not in SCRIPTED_BACKENDS:
        ids.append(model["id"])
    ids += list(data.get("arms", []) or [])
    for section in (data.get("judges") or {}).values():
        if isinstance(section, dict) and "model" in section:
            ids.append(section["model"])
    return ids


def bible_model_ids() -> list[str]:
    """Return the model ids the bible's Project Recipes blocks name: model ids, arms, judge models, base models."""
    lines = BIBLE.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Project Recipes")
    end = next(index for index in range(start + 1, len(lines)) if lines[index].startswith("## "))
    ids: list[str] = []
    cursor = start
    while "```yaml" in lines[cursor:end]:
        begin = lines.index("```yaml", cursor, end) + 1
        stop = lines.index("```", begin)
        data = yaml.safe_load("\n".join(lines[begin:stop]))
        ids += ids_of(data)
        if isinstance(data.get("base_model"), str):
            ids.append(data["base_model"])
        cursor = stop + 1
    return ids


def sweep_model_ids() -> list[str]:
    """Return the model column of every row of plans/runs/rngd-sweep/models*.tsv."""
    ids = []
    for path in SWEEP_TABLES:
        for line in path.read_text(encoding="ascii").splitlines():
            if line and not line.startswith("#"):
                ids.append(line.split("\t")[1])
    return ids


def load_error(families_module: ModuleType, path: Path) -> str:
    """Expect load_families(path) to raise FamilyError, a ValueError; return its message."""
    with pytest.raises(families_module.FamilyError) as caught:
        families_module.load_families(path)
    assert isinstance(caught.value, ValueError)
    return str(caught.value)


# ---------------------------------------------------------------------------
# The committed table


def test_the_table_file_is_assets_models_families(families_module: ModuleType) -> None:
    assert Path(families_module.FAMILIES_FILE) == TABLE


def test_the_table_loads_and_maps_ids_to_their_families(families_module: ModuleType) -> None:
    table = families_module.load_families()
    for model_id, family in EXPECTED.items():
        assert table.family(model_id) == family, model_id
    assert "gpt-oss" in table.reasoning_effort and "llama" not in table.reasoning_effort
    assert table.family("unlisted/model") is None
    assert table.family("LLAMA-3.3-70B") is None, "an id is matched exactly as written"
    assert set(table.models.values()) <= set(table.families)


def test_the_table_lists_every_model_id_the_recipes_and_sweep_tables_name(families_module: ModuleType) -> None:
    table = families_module.load_families()
    assert SWEEP_TABLES, "no plans/runs/rngd-sweep/models*.tsv table"
    named = {**recipe_model_ids(), **{model_id: "the bible" for model_id in bible_model_ids()}}
    named.update({model_id: "a sweep table" for model_id in sweep_model_ids()})
    assert "furiosa-ai/gpt-oss-120b" in named and "llama-3.3-70b" in named, "the scan reads the files"
    missing = sorted(f"{model_id} ({where})" for model_id, where in named.items() if table.family(model_id) is None)
    assert not missing, f"model ids with no family in {TABLE.relative_to(REPO).as_posix()}: {missing}"


def test_the_table_is_plain_ascii_with_lf_newlines() -> None:
    data = TABLE.read_bytes()
    data.decode("ascii")
    assert b"\r" not in data and data.endswith(b"\n")


def test_load_families_reads_the_module_file_at_each_call(
    families_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(families_module, "FAMILIES_FILE", FIXTURE_TABLE)
    assert families_module.load_families().family("SYNTHETIC-judge") == "synthetic-b"


# ---------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("- a list\n", "mapping"),
        (GOOD + "notes: x\n", "notes"),
        ("reasoning_effort: []\nmodels: {m-1: alpha}\n", "families"),
        ("families: [alpha]\nmodels: {m-1: alpha}\n", "reasoning_effort"),
        ("families: [alpha]\nreasoning_effort: []\n", "models"),
        ("families: []\nreasoning_effort: []\nmodels: {m-1: alpha}\n", "families"),
        ("families: [alpha, alpha]\nreasoning_effort: []\nmodels: {m-1: alpha}\n", "families"),
        ("families: [Alpha]\nreasoning_effort: []\nmodels: {m-1: Alpha}\n", "families"),
        ("families: alpha\nreasoning_effort: []\nmodels: {m-1: alpha}\n", "families"),
        ("families: [alpha]\nreasoning_effort: [gamma]\nmodels: {m-1: alpha}\n", "reasoning_effort"),
        ("families: [alpha]\nreasoning_effort: [alpha, alpha]\nmodels: {m-1: alpha}\n", "reasoning_effort"),
        ("families: [alpha]\nreasoning_effort: []\nmodels: {m-1: gamma}\n", "models"),
        ("families: [alpha]\nreasoning_effort: []\nmodels: {}\n", "models"),
        ("families: [alpha]\nreasoning_effort: []\nmodels: [m-1]\n", "models"),
        ("families: [alpha]\nreasoning_effort: []\nmodels: {'': alpha}\n", "models"),
        ("families: [alpha]\nreasoning_effort: []\nmodels: {1: alpha}\n", "models"),
        ("families: [alpha, beta]\nreasoning_effort: []\nmodels:\n  m-1: alpha\n  m-1: beta\n", "m-1"),
        ("families: [alpha\n", "families.yaml"),
    ],
)
def test_a_bad_table_is_refused_naming_the_key(
    families_module: ModuleType, tmp_path: Path, text: str, key: str
) -> None:
    path = tmp_path / "families.yaml"
    path.write_bytes(text.encode("ascii"))
    message = load_error(families_module, path)
    assert key in message and "families.yaml" in message, message


def test_a_table_that_is_not_ascii_or_missing_is_refused(families_module: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "families.yaml"
    path.write_bytes(("# caf\N{LATIN SMALL LETTER E WITH ACUTE}\n" + GOOD).encode("utf-8"))
    assert "families.yaml" in load_error(families_module, path)
    assert "absent.yaml" in load_error(families_module, tmp_path / "absent.yaml")


def test_a_good_table_loads(families_module: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "families.yaml"
    path.write_bytes(GOOD.encode("ascii"))
    table = families_module.load_families(path)
    assert table.families == ("alpha", "beta")
    assert table.reasoning_effort == frozenset({"beta"})
    assert dict(table.models) == {"m-1": "alpha", "m-2": "beta"}
    with pytest.raises(TypeError):
        table.models["m-3"] = "alpha"  # type: ignore[index]
