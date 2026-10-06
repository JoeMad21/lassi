"""Tests for the hf_local backend on Transformers and PyTorch, on the CPU, with a tiny model (task P17.4).

Bible: Model Serving (Serving Rules), Component Interfaces (LLMBackend),
Agent Rules 7, 10, and 12; plans/p17-portable.md, task P17.4; the P17.4
design (sections 4 to 9).

These tests need a framework extra and skip without one, with a reason that
names it (tests/tiny_hf.py NEEDS_EXTRA): run them as `uv run --extra cpu
pytest tests/llm/test_hf_local.py`. tests/llm/test_hf_local_settings.py holds
the checks that run without the extra.

The contract these tests fix:

- hf_local loads model.id at the commit model.revision names from
  $HF_HOME/hub only, with local_files_only=True and trust_remote_code=False
  on every from_pretrained call and no token; it sets the hub library's
  offline flag, so no load or request looks up a host, even when the library
  starts online with no agent registry cached; it reads HF_HOME when it loads,
  refuses an unset or empty HF_HOME naming it, and turns a revision the cache
  lacks into a ServingError naming the id and the revision. An ImportError
  during the load (a tokenizer that needs a package the extra lacks) is a
  ServingError too. Building the backend loads nothing; check() loads once.
- The model sits on the device the recipe names (cpu here).
- The prompt is the tokenizer's chat template with the generation prompt; a
  tokenizer without a template is refused. Token counts are the tokenizer's:
  prompt_tokens is the templated prompt's length and completion_tokens the
  ids generate() appended, an end id included when one stopped it. The stop
  ids are the checkpoint generation config's eos ids, then the tokenizer's.
- No sampling value of the checkpoint's generation_config.json reaches generate():
  greedy and sampled output equal an independent decoding loop although the
  snapshot holds misleading sampling defaults (tiny_hf.TRAP_DEFAULTS).
- Call k of a backend runs under torch.manual_seed(model.seed + k), counted
  from 0 when a call starts, a refused call included, so one seed repeats a
  run.
- A request whose prompt tokens plus max_tokens pass max_position_embeddings
  raises ContextExceeded (lassi.core.interfaces) with the three numbers before
  generate() runs; nothing is truncated. A request that fills the context
  exactly runs.
- framework() reads torch's build metadata only, never
  torch.cuda.is_available() or device_count() (OQ-002).
- serving() returns the record provenance.json holds as `serving`: model,
  revision, max_position_embeddings, dtype, device, threads, seed,
  stop_token_ids, and the installed versions of torch, transformers,
  tokenizers, huggingface-hub, and safetensors; torch and transformers match
  the cpu extra's pins in pyproject.toml.

The model's weights are random and every text SYNTHETIC; the autouse
no_network fixture (tests/llm/conftest.py) refuses any lookup of a host other
than the loopback names. No value in this module is a measurement.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
import socket
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from tiny_hf import (
    END_TOKEN,
    NEEDS_EXTRA,
    OTHER_REVISION,
    REPO_ID,
    REVISION,
    build_snapshot,
    chat_ids,
    load_reference,
    reference_decode,
    snapshot_dir,
    write_generation_config,
)

torch = pytest.importorskip("torch", reason=NEEDS_EXTRA)
transformers = pytest.importorskip("transformers", reason=NEEDS_EXTRA)

from lassi.core.devices import FrameworkBuild  # noqa: E402
from lassi.core.interfaces import Message, Sampling  # noqa: E402
from lassi.core.record import json_text  # noqa: E402
from lassi.llm import ServingError  # noqa: E402

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
SEED = 1234
CPU = {"kind": "cpu"}
MESSAGES = (Message("system", "translate the program below"), Message("user", "int main() { return 0; }"))
GREEDY = Sampling(temperature=0.0, top_p=1.0, max_tokens=12)
VERSION_NAMES = ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")
SERVING_KEYS = {
    "model", "revision", "max_position_embeddings", "dtype", "device", "threads", "seed", "stop_token_ids", "versions"
}
# A SYNTHETIC token value; hf_local never reads HF_TOKEN, so it must reach no load call.
TOKEN_VALUE = "hf_SYNTHETIC0000000000000000000000"


def hf_local() -> ModuleType:
    """Import lassi.llm.hf_local, failing the test clearly while it does not exist."""
    try:
        return importlib.import_module("lassi.llm.hf_local")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.llm.hf_local does not exist yet (task P17.4): {error}")


def context_exceeded() -> type:
    """Return lassi.core.interfaces.ContextExceeded, failing the test clearly while it does not exist."""
    module = importlib.import_module("lassi.core.interfaces")
    if not hasattr(module, "ContextExceeded"):
        pytest.fail("lassi.core.interfaces has no ContextExceeded yet (task P17.4)")
    return module.ContextExceeded


@pytest.fixture
def hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return an empty fake HF_HOME and point the variable at it."""
    home = tmp_path / "hf-home"
    home.mkdir()
    monkeypatch.setenv("HF_HOME", str(home))
    return home


def backend(revision: str = REVISION, seed: int = SEED, model_id: str = REPO_ID) -> Any:
    """Return hf_local built as the runner builds it: factory(model.id, device=..., revision=..., seed=...)."""
    return hf_local().HFLocalBackend(model_id, device=dict(CPU), revision=revision, seed=seed)


def messages_of(messages: Sequence[Message]) -> list[tuple[str, str]]:
    """Return (role, content) pairs for tiny_hf.chat_ids."""
    return [(message.role, message.content) for message in messages]


def words(text: str) -> set[str]:
    """Return the runs of digits in `text`, so a test can look for a number as a whole."""
    return set(re.findall(r"\d+", text))


# ---------------------------------------------------------------------------
# Spies on the framework: generate() and from_pretrained()


@dataclass
class GenerateCall:
    """One generate() call: the prompt ids it got, the ids it appended, and the model it ran on."""

    prompt: list[int]
    new: list[int]
    model: Any


def spy_generate(monkeypatch: pytest.MonkeyPatch) -> list[GenerateCall]:
    """Wrap the tiny model class's generate() and return the list each call is appended to."""
    cls = transformers.Qwen2ForCausalLM
    original = cls.generate

    def generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the real generate() and record its prompt, the ids it appended, and the model."""
        output = original(self, *args, **kwargs)
        ids = kwargs.get("input_ids", kwargs.get("inputs", args[0] if args else None))
        prompt = [int(item) for item in ids[0].tolist()]
        sequences = getattr(output, "sequences", output)
        calls.append(GenerateCall(prompt, [int(item) for item in sequences[0, len(prompt) :].tolist()], self))
        return output

    calls: list[GenerateCall] = []
    monkeypatch.setattr(cls, "generate", generate)
    return calls


@dataclass
class LoadCall:
    """One outermost from_pretrained call: the Auto class, its arguments, and what it returned."""

    owner: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any


def spy_loads(monkeypatch: pytest.MonkeyPatch) -> list[LoadCall]:
    """Wrap from_pretrained of AutoConfig, AutoTokenizer, and AutoModelForCausalLM; record the outermost calls."""
    calls: list[LoadCall] = []
    depth = [0]

    def wrap(owner: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """Return `original` wrapped so that an outermost call is recorded under `owner`."""

        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            """Call the real from_pretrained and record the call when it is the outermost one."""
            depth[0] += 1
            try:
                result = original(*args, **kwargs)
            finally:
                depth[0] -= 1
            if depth[0] == 0:
                calls.append(LoadCall(owner, args, dict(kwargs), result))
            return result

        return from_pretrained

    for owner in ("AutoConfig", "AutoTokenizer", "AutoModelForCausalLM"):
        cls = getattr(transformers, owner)
        monkeypatch.setattr(cls, "from_pretrained", wrap(owner, cls.from_pretrained))
    return calls


def first_new_stop(ids: Sequence[int], taken: Sequence[int]) -> int:
    """Return the index of the first id after the first that is not earlier in `ids` and not in `taken`."""
    for index in range(1, len(ids)):
        if ids[index] not in ids[:index] and ids[index] not in taken:
            return index
    pytest.fail(f"the greedy reference {ids} has no id fit to serve as a second stop id")


# ---------------------------------------------------------------------------
# Loading by id and revision from HF_HOME, offline


def test_loads_the_pinned_revision_from_hf_home(hf_home: Path) -> None:
    build_snapshot(hf_home, revision=REVISION, context=64)
    build_snapshot(hf_home, revision=OTHER_REVISION, context=96, seed=1)  # refs/main now names OTHER_REVISION
    first, second = backend(REVISION), backend(OTHER_REVISION)
    assert first.check() == {"id": REPO_ID, "revision": REVISION, "max_position_embeddings": 64}
    assert second.check() == {"id": REPO_ID, "revision": OTHER_REVISION, "max_position_embeddings": 96}
    assert (first.serving()["revision"], first.serving()["max_position_embeddings"]) == (REVISION, 64)
    assert (second.serving()["revision"], second.serving()["max_position_embeddings"]) == (OTHER_REVISION, 96)


def test_check_returns_a_fresh_copy(hf_home: Path) -> None:
    build_snapshot(hf_home)
    loaded = backend()
    entry = loaded.check()
    entry["max_position_embeddings"] = -1
    assert loaded.check()["max_position_embeddings"] == 64


def test_an_absent_revision_is_a_serving_error_naming_id_and_revision(hf_home: Path) -> None:
    build_snapshot(hf_home, revision=REVISION)
    with pytest.raises(ServingError) as info:
        backend(OTHER_REVISION).check()
    message = str(info.value)
    assert REPO_ID in message and OTHER_REVISION in message and "HF_HOME" in message, message


def test_an_absent_model_is_a_serving_error(hf_home: Path) -> None:
    with pytest.raises(ServingError) as info:
        backend().complete(MESSAGES, GREEDY)
    assert REPO_ID in str(info.value) and REVISION in str(info.value), str(info.value)


def test_an_import_error_during_the_load_is_a_serving_error(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_snapshot(hf_home)

    def needs_a_package(*args: Any, **kwargs: Any) -> Any:
        """Fail as a tokenizer that needs a package the extra lacks fails."""
        raise ImportError("SYNTHETIC: this tokenizer needs a package the extra does not install")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", needs_a_package)
    with pytest.raises(ServingError) as info:
        backend().check()
    message = str(info.value)
    assert REPO_ID in message and REVISION in message and "SYNTHETIC" in message, message


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_unset_hf_home_is_refused_naming_hf_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    build_snapshot(tmp_path / "hf-home")
    if value is None:
        monkeypatch.delenv("HF_HOME", raising=False)
    else:
        monkeypatch.setenv("HF_HOME", value)
    loads = spy_loads(monkeypatch)
    built = backend()
    with pytest.raises(ServingError) as info:
        built.check()
    assert "HF_HOME" in str(info.value), str(info.value)
    assert loads == [], "hf_local never falls back to a cache under the home directory"


def test_load_passes_local_files_only_and_the_hub_cache(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_snapshot(hf_home)
    monkeypatch.setenv("HF_TOKEN", TOKEN_VALUE)
    loads = spy_loads(monkeypatch)
    backend().check()
    assert {call.owner for call in loads} == {"AutoConfig", "AutoTokenizer", "AutoModelForCausalLM"}, loads
    for call in loads:
        named = call.args[0] if call.args else call.kwargs.get("pretrained_model_name_or_path")
        assert named == REPO_ID, (call.owner, named)
        assert call.kwargs.get("revision") == REVISION, call.owner
        assert Path(call.kwargs["cache_dir"]).resolve() == (hf_home / "hub").resolve(), call.owner
        assert call.kwargs.get("local_files_only") is True, call.owner
        assert call.kwargs.get("trust_remote_code") is False, call.owner
        assert not call.kwargs.get("token"), call.owner
        assert TOKEN_VALUE not in repr(call.kwargs), call.owner


@pytest.fixture
def hub_lookups(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Return the host lookups made during the test, with the hub library online and no agent registry cached.

    As in a fresh process whose environment sets neither HF_HUB_OFFLINE nor
    HF_HUB_DISABLE_TELEMETRY: the library's offline and telemetry flags are
    off, no registry is held in memory, and its registry file would sit in the
    empty hf_home. Each lookup is recorded, then refused by no_network
    (tests/llm/conftest.py, autouse, so set up first).
    """
    constants = importlib.import_module("huggingface_hub.constants")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", False)
    monkeypatch.setattr(constants, "AGENT_HARNESSES_PATH", str(hf_home / ".agent_harnesses.json"))
    monkeypatch.setattr(importlib.import_module("huggingface_hub.utils._detect_agent"), "_registry", None)
    guarded = socket.getaddrinfo
    lookups: list[str] = []

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        """Record the lookup of `host`, then pass it to the no_network guard."""
        lookups.append(str(host))
        return guarded(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return lookups


def test_no_load_or_request_reaches_the_hub(hf_home: Path, hub_lookups: list[str]) -> None:
    # huggingface_hub 1.33.0 fetches an agent registry from the Hub on its first request, local_files_only or not,
    # unless its offline flag is set; hf_local sets it before loading.
    build_snapshot(hf_home)
    built = backend()
    built.check()
    built.serving()
    built.complete(MESSAGES, GREEDY)
    with pytest.raises(ServingError):
        backend(OTHER_REVISION).check()
    assert hub_lookups == [], f"hf_local looked up {hub_lookups}"
    assert not (hf_home / ".agent_harnesses.json").exists(), "the hub library wrote a fetched registry"
    assert importlib.import_module("huggingface_hub.constants").HF_HUB_OFFLINE is True


def test_construction_loads_nothing(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loads = spy_loads(monkeypatch)
    built = backend()  # no snapshot exists yet, so a load at construction would fail
    assert loads == []
    assert built.model_id == REPO_ID
    build_snapshot(hf_home)
    built.check()
    built.check()
    built.complete(MESSAGES, GREEDY)
    owners = [call.owner for call in loads]
    assert owners.count("AutoModelForCausalLM") == 1, f"check() loads once: {owners}"


def test_the_model_sits_on_the_recipe_device(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_snapshot(hf_home)
    loads = spy_loads(monkeypatch)
    generated = spy_generate(monkeypatch)
    built = backend()
    built.complete(MESSAGES, GREEDY)
    (model,) = [call.result for call in loads if call.owner == "AutoModelForCausalLM"]
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    assert [call.model for call in generated] == [model]
    assert built.serving()["device"] == "cpu"


# ---------------------------------------------------------------------------
# The chat template and the token counts


def test_the_chat_template_is_applied(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = build_snapshot(hf_home)
    tokenizer, _ = load_reference(snapshot)
    expected = chat_ids(tokenizer, messages_of(MESSAGES))
    assert expected[-3:] == tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)[-3:]
    generated = spy_generate(monkeypatch)
    completion = backend().complete(MESSAGES, GREEDY)
    (call,) = generated
    assert call.prompt == expected, "the prompt is the tokenizer's chat template with the generation prompt"
    assert completion.prompt_tokens == len(expected)


def test_a_tokenizer_without_a_chat_template_is_refused(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_snapshot(hf_home, chat_template=None)
    generated = spy_generate(monkeypatch)
    with pytest.raises(ServingError) as info:
        backend().check()
    message = str(info.value).lower()
    assert "chat_template" in message or "chat template" in message, message
    with pytest.raises(ServingError):
        backend().complete(MESSAGES, GREEDY)
    assert generated == [], "hf_local never builds a prompt itself"


def test_token_counts_are_the_tokenizers(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = build_snapshot(hf_home)
    tokenizer, model = load_reference(snapshot)
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    prompt = chat_ids(tokenizer, messages_of(MESSAGES))
    greedy = reference_decode(model, prompt, temperature=0.0, seed=0, max_new=12, stops=[end])
    # A second end id in the checkpoint's generation config, taken from the greedy output, stops generation there.
    index = first_new_stop(greedy, [end])
    write_generation_config(snapshot, {}, [end, greedy[index]])
    generated = spy_generate(monkeypatch)
    built = backend()
    completion = built.complete(MESSAGES, GREEDY)
    expected = greedy[: index + 1]
    assert generated[0].new == expected
    assert completion.prompt_tokens == len(prompt)
    assert completion.completion_tokens == len(expected), "the end id that stopped generation is counted"
    assert completion.text == tokenizer.decode(expected, skip_special_tokens=True)
    assert built.serving()["stop_token_ids"] == [end, greedy[index]], "the generation config's ids, in order"


# ---------------------------------------------------------------------------
# Sampling with a recorded seed; nothing from the checkpoint's generation config


def test_greedy_ignores_the_checkpoint_generation_defaults(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = build_snapshot(hf_home)
    tokenizer, model = load_reference(snapshot)
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    prompt = chat_ids(tokenizer, messages_of(MESSAGES))
    expected = reference_decode(model, prompt, temperature=0.0, seed=0, max_new=12, stops=[end])
    ids = torch.tensor([prompt])
    trapped = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=12, do_sample=False)
    assert trapped[0, len(prompt) :].tolist() != expected, "the snapshot's defaults must change greedy output"
    generated = spy_generate(monkeypatch)
    completion = backend().complete(MESSAGES, GREEDY)
    assert generated[0].new == expected
    assert completion.completion_tokens == len(expected)


@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_sampling_ignores_the_checkpoint_generation_defaults(
    hf_home: Path, monkeypatch: pytest.MonkeyPatch, temperature: float
) -> None:
    snapshot = build_snapshot(hf_home, context=256)
    tokenizer, model = load_reference(snapshot)
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    prompt = chat_ids(tokenizer, messages_of(MESSAGES))
    sampling = Sampling(temperature=temperature, top_p=1.0, max_tokens=12)
    generated = spy_generate(monkeypatch)
    built = backend()
    for k in range(2):
        built.complete(MESSAGES, sampling)
        expected = reference_decode(model, prompt, temperature=temperature, seed=SEED + k, max_new=12, stops=[end])
        assert generated[k].new == expected, f"call {k} samples under torch.manual_seed(seed + {k})"


def test_each_call_seeds_torch_with_seed_plus_its_index(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Call 1 is refused (its max_tokens passes the context), and still takes its place: call 2 runs under seed + 2.
    snapshot = build_snapshot(hf_home, context=64)
    tokenizer, model = load_reference(snapshot)
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    prompt = chat_ids(tokenizer, messages_of(MESSAGES))
    sampling = Sampling(temperature=1.0, top_p=1.0, max_tokens=8)
    generated = spy_generate(monkeypatch)
    built = backend()
    built.complete(MESSAGES, sampling)
    with pytest.raises(context_exceeded()):
        built.complete(MESSAGES, Sampling(temperature=1.0, top_p=1.0, max_tokens=64))
    built.complete(MESSAGES, sampling)
    seeds = [SEED, SEED + 2]
    for call, seed in zip(generated, seeds, strict=True):
        assert call.new == reference_decode(model, prompt, temperature=1.0, seed=seed, max_new=8, stops=[end]), seed


def test_the_same_seed_repeats_the_run(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_snapshot(hf_home, context=256)
    sampling = Sampling(temperature=1.0, top_p=1.0, max_tokens=12)
    other = (Message("user", "#pragma omp parallel for"),)
    generated = spy_generate(monkeypatch)
    runs = []
    for seed in (SEED, SEED, SEED + 100):
        built = backend(seed=seed)
        runs.append([built.complete(MESSAGES, sampling), built.complete(other, sampling)])
    assert runs[0] == runs[1], "the same seed repeats every call's text and counts"
    assert generated[0].new == generated[2].new and generated[1].new == generated[3].new
    assert generated[4].new != generated[0].new, "another seed samples another output"


# ---------------------------------------------------------------------------
# The context: refused, never truncated


def test_a_prompt_past_the_context_is_refused_unsent(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = context_exceeded()
    snapshot = build_snapshot(hf_home, context=64)
    tokenizer, _ = load_reference(snapshot)
    prompt = len(chat_ids(tokenizer, messages_of(MESSAGES)))
    assert prompt < 64
    generated = spy_generate(monkeypatch)
    built = backend()
    over = 64 - prompt + 1
    with pytest.raises(refusal) as info:
        built.complete(MESSAGES, Sampling(temperature=0.0, top_p=1.0, max_tokens=over))
    error = info.value
    assert (error.prompt_tokens, error.max_tokens, error.context) == (prompt, over, 64)
    assert {str(prompt), str(over), "64"} <= words(str(error)), str(error)
    assert not isinstance(error, (ServingError, ValueError)), "no existing handler catches it by accident"
    long_messages = (Message("user", "int main() { return 0; } " * 20),)
    long_prompt = len(chat_ids(tokenizer, messages_of(long_messages)))
    assert long_prompt > 64
    with pytest.raises(refusal) as info:
        built.complete(long_messages, Sampling(temperature=0.0, top_p=1.0, max_tokens=1))
    assert (info.value.prompt_tokens, info.value.max_tokens, info.value.context) == (long_prompt, 1, 64)
    assert generated == [], "a refused request never reaches generate()"


def test_a_prompt_that_fills_the_context_exactly_runs(hf_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = build_snapshot(hf_home, context=64)
    tokenizer, _ = load_reference(snapshot)
    prompt = len(chat_ids(tokenizer, messages_of(MESSAGES)))
    generated = spy_generate(monkeypatch)
    completion = backend().complete(MESSAGES, Sampling(temperature=0.0, top_p=1.0, max_tokens=64 - prompt))
    assert len(generated) == 1
    assert completion.prompt_tokens == prompt
    assert 1 <= completion.completion_tokens <= 64 - prompt


# ---------------------------------------------------------------------------
# framework() and the serving record


def test_framework_reports_the_installed_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        """Fail the test: framework() must not call this."""
        raise AssertionError("framework() initialized the device runtime (OQ-002)")

    monkeypatch.setattr(torch.cuda, "is_available", refuse)
    monkeypatch.setattr(torch.cuda, "device_count", refuse)
    build = hf_local().HFLocalBackend.framework()
    assert build == FrameworkBuild(
        name="torch", version=torch.__version__, cuda=torch.version.cuda, hip=torch.version.hip
    )


def cpu_extra_pins() -> dict[str, str]:
    """Return package -> pinned version of the cpu extra in pyproject.toml; fail clearly when it has none."""
    data = tomllib.loads(PYPROJECT.read_bytes().decode("ascii"))
    extras = data["project"].get("optional-dependencies", {})
    if "cpu" not in extras:
        pytest.fail("pyproject.toml has no cpu extra yet (task P17.4)")
    pins: dict[str, str] = {}
    for requirement in extras["cpu"]:
        name, _, version = requirement.split(";")[0].strip().partition("==")
        pins[name.strip()] = version.strip()
    return pins


def test_serving_record_holds_versions_and_pins(hf_home: Path) -> None:
    snapshot = build_snapshot(hf_home)
    tokenizer, _ = load_reference(snapshot)
    built = backend()
    record = built.serving()
    assert set(record) == SERVING_KEYS, sorted(record)
    assert (record["model"], record["revision"], record["seed"]) == (REPO_ID, REVISION, SEED)
    assert record["max_position_embeddings"] == 64
    assert (record["dtype"], record["device"]) == ("float32", "cpu")
    assert record["threads"] == torch.get_num_threads()
    assert record["stop_token_ids"] == [tokenizer.convert_tokens_to_ids(END_TOKEN)]
    assert record["versions"] == {name: importlib.metadata.version(name) for name in VERSION_NAMES}
    pins = cpu_extra_pins()
    for name in ("torch", "transformers"):
        assert record["versions"][name].split("+")[0] == pins[name], (name, record["versions"][name], pins[name])
    assert json.loads(json_text(record)) == record, "provenance.json can hold it as written"
    record["seed"] = -1
    assert built.serving()["seed"] == SEED, "serving() returns a fresh record"


def test_the_snapshot_is_read_from_its_revision_directory(hf_home: Path) -> None:
    # Guard on the fixture itself: the snapshot sits where a named fetch step writes it.
    target = build_snapshot(hf_home)
    assert target == snapshot_dir(hf_home, REPO_ID, REVISION)
    assert (target / "config.json").is_file() and (target / "tokenizer.json").is_file()
