"""Tests for the prompt templates and lassi.prompts (P0.11).

Prompts are files under assets/prompts/<set>/, one plain-text file per
prompt with string.Template `$name` placeholders, never string literals in
code (bible Design Principle 3). The P0 smoke set, p0-smoke, holds
generate.txt and correct.txt, and the recipe key `prompts` picks the set.
lassi.prompts.prompt_dir finds a set's directory and lassi.prompts.render
fills one template with Template.substitute; a missing field or a missing
file raises ValueError naming the file. The field values below are synthetic
text, and no value in this module is a measurement.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

import pytest

from lassi.core.files import render_file_blocks
from lassi.prompts import prompt_dir, render

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "assets" / "prompts"
SMOKE_SET = PROMPTS / "p0-smoke"

# The placeholders each p0-smoke template uses (P0.11 contract, Prompts).
FIELDS = {
    "generate": ("source_language", "target_language", "source_files", "target_files"),
    "correct": ("target_language", "files", "diagnostics", "target_files"),
}

# One synthetic value per placeholder, each distinct, so a filled template shows where every value went.
SAMPLE = {
    "source_language": "omp",
    "target_language": "cuda",
    "source_files": render_file_blocks({"main.cpp": "int main() { return 1; }\n"}),
    "target_files": "main.cu, kernels.cuh",
    "files": render_file_blocks({"main.cu": "#error fixture\nint main() { return 2; }\n"}),
    "diagnostics": "error main.cu:1:1 [fake-error] the source has an #error line\nwarning fixture warning",
}


def fields_for(name: str) -> dict[str, str]:
    """Return the sample value of every placeholder of template `name`."""
    return {field: SAMPLE[field] for field in FIELDS[name]}


def template_text(name: str) -> str:
    """Return the text of assets/prompts/p0-smoke/<name>.txt after checking it is plain ASCII with LF newlines."""
    data = (SMOKE_SET / f"{name}.txt").read_bytes()
    assert data.isascii(), f"{name}.txt is not ASCII"
    assert b"\r" not in data, f"{name}.txt has a CR"
    return data.decode("ascii")


def placeholders(text: str) -> set[str]:
    """Return the names of the `$name` and `${name}` placeholders in a template; fail on a bad `$`."""
    names: set[str] = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None, f"a '$' that is no placeholder at offset {match.start()}"
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


def test_prompt_dir_defaults_to_assets_prompts() -> None:
    found = prompt_dir("p0-smoke")
    assert found.resolve() == SMOKE_SET.resolve()
    assert found.is_dir()


def test_prompt_dir_takes_another_root(tmp_path: Path) -> None:
    (tmp_path / "fixture-set").mkdir()
    assert prompt_dir("fixture-set", roots=[tmp_path]).resolve() == (tmp_path / "fixture-set").resolve()


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_smoke_template_uses_exactly_its_fields(name: str) -> None:
    assert placeholders(template_text(name)) == set(FIELDS[name])


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_smoke_template_asks_for_file_blocks(name: str) -> None:
    text = template_text(name)
    assert "// FILE:" in text, f"{name}.txt must tell the model to start each fenced block with // FILE: <path>"
    assert text.endswith("\n")


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_render_fills_every_placeholder(name: str) -> None:
    fields = fields_for(name)
    rendered = render("p0-smoke", name, fields)
    assert rendered == Template(template_text(name)).substitute(fields)
    for field, value in fields.items():
        assert value in rendered
        assert f"${field}" not in rendered and f"${{{field}}}" not in rendered


def test_render_keeps_dollar_signs_inside_field_values() -> None:
    fields = fields_for("generate")
    fields["source_files"] = render_file_blocks({"run.sh": "echo $target_language ${files} $$\n"})
    rendered = render("p0-smoke", "generate", fields)
    assert fields["source_files"] in rendered


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_render_without_a_field_raises_value_error_naming_the_file(name: str) -> None:
    for missing in FIELDS[name]:
        fields = {field: value for field, value in fields_for(name).items() if field != missing}
        with pytest.raises(ValueError, match=rf"{name}\.txt"):
            render("p0-smoke", name, fields)


def test_render_of_a_missing_prompt_raises_value_error_naming_the_file() -> None:
    with pytest.raises(ValueError, match=r"no-such-prompt\.txt"):
        render("p0-smoke", "no-such-prompt", SAMPLE)


def test_render_of_a_missing_set_raises_value_error_naming_it() -> None:
    with pytest.raises(ValueError, match="no-such-set"):
        render("no-such-set", "generate", SAMPLE)
