"""The in-process backend on Transformers and PyTorch, registered as LLMBackend "hf_local" (bible Model Serving).

Registration rule (PHASE-NOTES P0): this module imports only the standard
library and lassi at module level, so importing lassi.llm registers hf_local
without the framework extra. torch is imported only when framework() is
called or the model loads, and transformers only when the model loads;
building the backend without them raises FrameworkMissing, whose message
names the extra.

Settings (task P17.4). The runner builds it as factory(model.id,
device=..., revision=..., seed=...): model.id is a Hub repository id
(<owner>/<name>, never a path, since from_pretrained would load a directory
the id names), model.revision a full commit id, model.seed an integer from 0
to 2**32 - 1, and model.device a device section with at most one index. A
missing or malformed one is a ValueError naming the key. Construction loads
nothing and reads no file; check() loads the model once.

Loading. The model, its config, and its tokenizer load from $HF_HOME/hub
only, at the pinned revision, with local_files_only=True,
trust_remote_code=False, no token, and the hub library's offline flag set
for the process (hub_offline), in the checkpoint's dtype, on the named
device (cpu, or cuda:<i> for cuda and rocm, since ROCm PyTorch reaches its
GPUs through torch's cuda device type). An unset HF_HOME, a
revision the cache lacks, a config without max_position_embeddings, a
tokenizer without a chat template, or a model without an end token is a
ServingError.

Requests. The prompt is the tokenizer's chat template with the generation
prompt; nothing is truncated. The model's own generation config is replaced
by a neutral one at load, and every sampling field is passed on each call
(generation_settings), so no sampling value of the checkpoint's
generation_config.json reaches generation; only its eos ids do, as stop ids.
Token counts are the tokenizer's.

Seed rule: the backend counts its complete() calls from 0 when each starts,
a refused one included, and call k samples under torch.manual_seed(seed +
k); a refused call seeds nothing. So a rerun of the same recipe repeats
every call's seed.

Context rule: a request whose prompt tokens plus llm.sampling.max_tokens pass
the config's max_position_embeddings raises ContextExceeded
(lassi.core.interfaces) before anything is generated.

framework() reads torch's build metadata only, never a call that
initializes a device runtime (OQ-002). serving() returns the record a run
writes to provenance.json as `serving`.

Trained checkpoints (2026-10-23; OQ-076 (3a)). Two optional settings,
model.checkpoint (an absolute path to a checkpoint directory lassi train
wrote) and model.checkpoint_sha256 (its content digest, lassi.core.digest
tree_sha256, which lassi train records in provenance.json as
checkpoint_sha256), are set together or not at all; without them hf_local
loads exactly as above. model.id and model.revision still pin the Hub
snapshot the checkpoint was trained from: its base, or a scratch base's
tokenizer. check() refuses, before any framework import, a checkpoint whose
digest differs, whose checkpoint.json (CHECKPOINT_RECORDS) names a fixture
base or a base other than model.id at model.revision, whose weights are
not full, lora, or prompt_tuning (an adapter needs a Hub base), or, for
prompt_tuning, whose peft record is not a mapping naming num_virtual_tokens
of at least 1. Full weights load from the checkpoint directory alone; lora
and prompt_tuning load the base from HF_HOME's hub cache at the revision and
apply the adapter with PEFT's PeftModel.from_pretrained (is_trainable
False). The tokenizer is the checkpoint's own. A prompt-tuned model's soft prompt takes
num_virtual_tokens positions, which the context rule counts
(prompt_positions). serving() adds checkpoint and checkpoint_sha256 when a
checkpoint is served.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType
from typing import Any

from lassi.core.capabilities import MODEL_CHECK, TAKES_DEVICE
from lassi.core.devices import DeviceSpec, FrameworkBuild, parse_device
from lassi.core.digest import tree_sha256
from lassi.core.interfaces import Completion, ContextExceeded, Message, Sampling
from lassi.core.registry import register
from lassi.llm._http import ServingError

# The framework modules hf_local imports when it loads, in the order a missing one is named.
FRAMEWORK_MODULES = ("torch", "transformers")
# The installed distributions whose versions the serving record holds.
VERSION_NAMES = ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")

# A Hub repository id, <owner>/<name>; a path ("", "/x", "C:/x", "./x", "../x") never fits.
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEED_LIMIT = 2**32
# The records file lassi train writes into every checkpoint (lassi.train.checkpoint RECORDS_JSON; named here, since
# lassi.train imports this module), and the weight modes a checkpoint is served in.
CHECKPOINT_RECORDS = "checkpoint.json"
ADAPTER_WEIGHTS = ("lora", "prompt_tuning")


class FrameworkMissing(ImportError):
    """torch or transformers is not installed: hf_local needs a framework extra; the message names it."""


def framework_missing_message(module: str) -> str:
    """Return the message of FrameworkMissing for the module `module` that could not be imported."""
    return (
        "hf_local needs PyTorch and Transformers, which install only with a framework extra: run "
        "`uv sync --extra cpu` (or --extra cuda, --extra rocm) and run tests with `uv run --extra cpu ...`; "
        f"{module} could not be imported"
    )


def _import(name: str) -> ModuleType:
    """Return the framework module `name`; raise FrameworkMissing naming it when it cannot be imported."""
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise FrameworkMissing(framework_missing_message(name)) from error


def _import_framework() -> tuple[ModuleType, ModuleType]:
    """Return the torch and transformers modules; raise FrameworkMissing naming the first that cannot be imported."""
    return _import(FRAMEWORK_MODULES[0]), _import(FRAMEWORK_MODULES[1])


def hub_offline() -> None:
    """Set the hub library offline for the process, so no load reaches the Hub.

    The one function that sets the flag: hf_local and the trl trainer
    (lassi.train.trl_trainer) call it before any load (task P17.9).
    local_files_only alone does not keep a load off the Hub: in huggingface_hub 1.33.0, with telemetry
    on (utils/_headers.py:183), a request's headers call detect_agent()
    (utils/_headers.py:187), which fetches
    {ENDPOINT}/api/agent-harnesses and writes it under HF_HOME unless
    constants.HF_HUB_OFFLINE is true (utils/_detect_agent.py:191), a flag read
    when called, as is the offline check that refuses every hub request
    (utils/_http.py:283).
    """
    importlib.import_module("huggingface_hub.constants").HF_HUB_OFFLINE = True


def torch_device(spec: DeviceSpec) -> str:
    """Return the torch device of a device section: "cpu", or "cuda:<i>" for cuda or rocm index i.

    ROCm PyTorch reaches its GPUs through torch's cuda device type. More than
    one index is a ValueError naming model.device: hf_local runs on one device.
    """
    if spec.kind == "cpu":
        return "cpu"
    if len(spec.indices) != 1:
        raise ValueError(f"hf_local runs on one device; model.device.indices names {len(spec.indices)}")
    return f"cuda:{spec.indices[0]}"


def generation_settings(sampling: Sampling, stops: Sequence[int], pad: int) -> dict[str, Any]:
    """Return every generation field for `sampling`, passed as GenerationConfig(**settings).

    A temperature above 0 samples with temperature, top_p, and top_k 0 (off);
    0 decodes greedily without those three. Both fix min_p None, typical_p 1.0,
    repetition_penalty 1.0, no_repeat_ngram_size 0, num_beams 1,
    num_return_sequences 1, max_new_tokens (llm.sampling.max_tokens), the stop
    ids as a list, and the pad id. A temperature below 0 or a top_p outside
    (0, 1] is a ServingError naming llm.sampling.<key>.
    """
    if sampling.temperature < 0:
        raise ServingError(f"llm.sampling.temperature must be at least 0 for hf_local, not {sampling.temperature}")
    if not 0 < sampling.top_p <= 1:
        raise ServingError(f"llm.sampling.top_p must be above 0 and at most 1 for hf_local, not {sampling.top_p}")
    settings: dict[str, Any] = {"do_sample": sampling.temperature > 0}
    if sampling.temperature > 0:
        settings.update(temperature=float(sampling.temperature), top_p=float(sampling.top_p), top_k=0)
    settings.update(
        min_p=None,
        typical_p=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=sampling.max_tokens,
        eos_token_id=list(stops),
        pad_token_id=pad,
    )
    return settings


def _hub_cache() -> Path:
    """Return $HF_HOME/hub, read now; ServingError naming HF_HOME when it is unset or empty (Agent Rule 7)."""
    home = os.environ.get("HF_HOME", "")
    if not home:
        raise ServingError(
            "HF_HOME is not set or is empty: hf_local reads models only from HF_HOME's hub cache and never from a "
            "default under the home directory"
        )
    return Path(home) / "hub"


@dataclass(frozen=True)
class _Loaded:
    """A loaded model: tokenizer, model, context, stop ids, pad id, torch device, dtype, and soft-prompt length."""

    tokenizer: Any
    model: Any
    context: int
    stops: tuple[int, ...]
    pad: int
    device: str
    dtype: str
    virtual_tokens: int = 0


def prompt_positions(prompt_tokens: int, loaded: _Loaded) -> int:
    """Return the context positions a prompt takes: its tokens plus a prompt-tuned model's soft prompt."""
    return prompt_tokens + loaded.virtual_tokens


def _as_ids(value: Any) -> list[int]:
    """Return an eos_token_id value (None, an int, or a list of ints) as a list."""
    if value is None:
        return []
    return [int(item) for item in value] if isinstance(value, (list, tuple)) else [int(value)]


def _load(model_id: str, revision: str, device: str) -> _Loaded:
    """Load model_id at revision from HF_HOME's hub cache with the hub offline, on `device`; ServingError if not.

    An OSError, ValueError, or ImportError from the framework import or any of
    the three loads becomes a ServingError naming the id and the revision.
    The hub library is set offline first (hub_offline).
    """
    cache = str(_hub_cache())
    where = f"{model_id} at revision {revision}"
    try:
        _, transformers = _import_framework()
        hub_offline()
        options = {"revision": revision, "cache_dir": cache, "local_files_only": True, "trust_remote_code": False}
        config = transformers.AutoConfig.from_pretrained(model_id, **options)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **options)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype="auto", **options)
    except (OSError, ValueError, ImportError) as error:
        raise ServingError(
            f"hf_local could not load {where} from HF_HOME's hub cache with the hub offline: {error}"
        ) from error
    return _finish(transformers, where, config, tokenizer, model, device)


def _finish(
    transformers: ModuleType, where: str, config: Any, tokenizer: Any, model: Any, device: str, virtual_tokens: int = 0
) -> _Loaded:
    """Check the context, chat template, and end ids, set the neutral generation config, and place the model."""
    context = getattr(config, "max_position_embeddings", None)
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        raise ServingError(f"hf_local refuses {where}: its config has no positive max_position_embeddings")
    if not getattr(tokenizer, "chat_template", None):
        raise ServingError(
            f"hf_local refuses {where}: its tokenizer has no chat template (chat_template), and hf_local never "
            "builds a prompt itself"
        )
    stops = tuple(dict.fromkeys(_as_ids(model.generation_config.eos_token_id) + _as_ids(tokenizer.eos_token_id)))
    if not stops:
        raise ServingError(f"hf_local refuses {where}: neither its generation config nor its tokenizer names an end id")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stops[0]
    model.generation_config = transformers.GenerationConfig(eos_token_id=list(stops), pad_token_id=pad)
    model.to(device)
    model.eval()
    dtype = str(model.dtype).removeprefix("torch.")
    return _Loaded(tokenizer, model, context, stops, int(pad), device, dtype, virtual_tokens)


@dataclass(frozen=True)
class _Trained:
    """What a checkpoint's records say hf_local needs: its weight mode and its soft prompt's length (0 for none)."""

    weights: str
    virtual_tokens: int


def _base_pin(base: Any) -> tuple[str, str] | None:
    """Return (repo, revision) a checkpoint's base_model record pins: a Hub base's, or a scratch base's tokenizer's."""
    if not isinstance(base, dict):
        return None
    pinned = base if base.get("kind") == "hub" else base.get("tokenizer") if base.get("kind") == "scratch" else None
    if not isinstance(pinned, dict):
        return None
    return pinned.get("repo"), pinned.get("revision")


def _trained(checkpoint: Path, expected: str, model_id: str, revision: str) -> _Trained:
    """Check the checkpoint's digest and records against the settings; ServingError naming what differs."""
    where = f"the checkpoint {checkpoint}"
    try:
        found = tree_sha256(checkpoint)
        records = json.loads((checkpoint / CHECKPOINT_RECORDS).read_bytes().decode("ascii"))["trainer_records"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ServingError(f"hf_local cannot read {where} and its {CHECKPOINT_RECORDS}: {error}") from error
    if found != expected:
        raise ServingError(f"hf_local refuses {where}: its content digest is {found}, not model.checkpoint_sha256 "
                           f"{expected}")
    base, weights = records.get("base_model"), records.get("weights")
    pin = _base_pin(base)
    if pin is None:
        name = base.get("name") if isinstance(base, dict) else base
        raise ServingError(f"hf_local refuses {where}: its base {name!r} is a fixture or unpinned base, and only a "
                           "Hub or scratch base is served")
    if pin != (model_id, revision):
        raise ServingError(f"hf_local refuses {where}: it was trained from {pin[0]} at {pin[1]}, not model.id "
                           f"{model_id} at model.revision {revision}")
    if weights not in ("full", *ADAPTER_WEIGHTS) or (weights in ADAPTER_WEIGHTS and base.get("kind") != "hub"):
        raise ServingError(f"hf_local refuses {where}: it serves full weights, and lora or prompt_tuning on a Hub "
                           f"base, not {weights!r} on a {base.get('kind')} base")
    if weights != "prompt_tuning":
        return _Trained(weights=weights, virtual_tokens=0)
    peft = records.get("peft")
    virtual = peft.get("num_virtual_tokens") if isinstance(peft, dict) else None
    if not isinstance(virtual, int) or isinstance(virtual, bool) or virtual < 1:
        raise ServingError(f"hf_local refuses {where}: a prompt_tuning checkpoint's peft record must be a mapping "
                           f"whose num_virtual_tokens is an integer of at least 1, not {peft!r}")
    return _Trained(weights=weights, virtual_tokens=virtual)


def _load_checkpoint(model_id: str, revision: str, checkpoint: Path, expected: str, device: str) -> _Loaded:
    """Load a trained checkpoint pinned by its digest, with the hub offline, on `device`; ServingError if not."""
    trained = _trained(checkpoint, expected, model_id, revision)
    where = f"the checkpoint {checkpoint} of {model_id} at revision {revision}"
    local = {"local_files_only": True, "trust_remote_code": False}
    try:
        _, transformers = _import_framework()
        hub_offline()
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(checkpoint), **local)
        if trained.weights == "full":
            config = transformers.AutoConfig.from_pretrained(str(checkpoint), **local)
            model = transformers.AutoModelForCausalLM.from_pretrained(str(checkpoint), config=config, dtype="auto",
                                                                      **local)
        else:
            options = {"revision": revision, "cache_dir": str(_hub_cache()), **local}
            config = transformers.AutoConfig.from_pretrained(model_id, **options)
            base = transformers.AutoModelForCausalLM.from_pretrained(model_id, config=config, dtype="auto", **options)
            model = _import("peft").PeftModel.from_pretrained(base, str(checkpoint), is_trainable=False)
    except (OSError, ValueError, ImportError) as error:
        raise ServingError(f"hf_local could not load {where} with the hub offline: {error}") from error
    return _finish(transformers, where, config, tokenizer, model, device, trained.virtual_tokens)


def _encode(tokenizer: Any, messages: Sequence[Message]) -> Any:
    """Return the tokenizer's chat template applied to `messages` with the generation prompt, as tensors; no cut."""
    return tokenizer.apply_chat_template(
        [{"role": message.role, "content": message.content} for message in messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _version(name: str) -> str | None:
    """Return the installed version of distribution `name`, or None when it is not installed."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _checked_settings(model_id: Any, revision: Any, seed: Any, device: Any) -> DeviceSpec:
    """Check model.id, model.revision, and model.seed, and return the parsed model.device; ValueError names the key."""
    if not isinstance(model_id, str) or _REPO_ID.fullmatch(model_id) is None:
        raise ValueError(
            f"model.id must be a Hub repository id <owner>/<name> for hf_local, not {model_id!r}; a path is never "
            "loaded"
        )
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError(
            f"model.revision must be a full commit id (40 lowercase hexadecimal characters) for hf_local, not "
            f"{revision!r}"
        )
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"model.seed must be an integer from 0 to {_SEED_LIMIT - 1} for hf_local, not {seed!r}")
    return parse_device(device, "model.device")


def _checked_checkpoint(checkpoint: Any, sha: Any) -> tuple[Path | None, str | None]:
    """Return model.checkpoint as a Path and model.checkpoint_sha256, both None when neither is set; reads no file."""
    if checkpoint is None and sha is None:
        return None, None
    if checkpoint is None:
        raise ValueError("model.checkpoint_sha256 is set without model.checkpoint, the directory it pins")
    absolute = isinstance(checkpoint, str) and (PurePosixPath(checkpoint).is_absolute()
                                                or PureWindowsPath(checkpoint).is_absolute())
    if not absolute:
        raise ValueError(f"model.checkpoint must be an absolute path to a checkpoint directory, not {checkpoint!r}")
    if not isinstance(sha, str) or _SHA256.fullmatch(sha) is None:
        raise ValueError(
            f"model.checkpoint_sha256 must be the checkpoint's content digest, 64 lowercase hex digits (lassi train "
            f"records it in provenance.json as checkpoint_sha256), not {sha!r}"
        )
    return Path(checkpoint), sha


@register("LLMBackend", "hf_local")
class HFLocalBackend:
    """Serves a model in process on Transformers and PyTorch, loaded once by check() from HF_HOME at a pinned revision.

    Construction checks the settings and that the framework is installed,
    and loads nothing. complete() calls check() first.
    """

    name = "hf_local"
    capabilities = frozenset({"chat", MODEL_CHECK, TAKES_DEVICE})
    # The model-section keys a recipe may set beside model.device; the runner passes them as keyword settings.
    config_keys = frozenset({"revision", "seed", "checkpoint", "checkpoint_sha256"})

    def __init__(
        self,
        model_id: str,
        *,
        device: Mapping[str, Any] | None = None,
        revision: str | None = None,
        seed: int | None = None,
        checkpoint: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        """Keep the checked settings; load nothing.

        Raises FrameworkMissing (an ImportError) when torch or transformers is
        not installed, found without importing either; then ValueError naming
        model.id, model.revision, model.seed, model.device, model.checkpoint,
        or model.checkpoint_sha256 for a missing or malformed setting.
        """
        for module in FRAMEWORK_MODULES:
            if importlib.util.find_spec(module) is None:
                raise FrameworkMissing(framework_missing_message(module))
        spec = _checked_settings(model_id, revision, seed, device)
        self.model_id = model_id
        self.revision = revision
        self.seed = seed
        self.device = spec
        self._torch_device = torch_device(spec)
        self.checkpoint, self.checkpoint_sha256 = _checked_checkpoint(checkpoint, checkpoint_sha256)
        self._loaded: _Loaded | None = None
        self._calls = 0

    def __repr__(self) -> str:
        """Show the class, model_id, and revision only."""
        return f"{type(self).__name__}(model_id={self.model_id!r}, revision={self.revision!r})"

    @staticmethod
    def framework() -> FrameworkBuild:
        """Return torch's build metadata: version, CUDA and HIP versions; FrameworkMissing without the extra.

        It imports torch alone and reads torch.__version__ and torch.version
        only, never a call that initializes a device runtime (OQ-002).
        """
        torch = _import(FRAMEWORK_MODULES[0])
        build = torch.version
        return FrameworkBuild(
            name="torch", version=torch.__version__, cuda=getattr(build, "cuda", None), hip=getattr(build, "hip", None)
        )

    def check(self) -> dict[str, Any]:
        """Load the model once and return a fresh {id, revision, max_position_embeddings}; ServingError if it fails.

        A failure is not cached, so a later call loads again.
        """
        loaded = self._model()
        return {"id": self.model_id, "revision": self.revision, "max_position_embeddings": loaded.context}

    def serving(self) -> dict[str, Any]:
        """Return the serving record provenance.json holds as `serving`; calls check() first.

        It holds model, revision, max_position_embeddings, dtype, device,
        threads (torch's CPU thread count), seed, stop_token_ids, and the
        installed versions of VERSION_NAMES (null when one is not installed),
        plus checkpoint and checkpoint_sha256 when a checkpoint is served.
        Each call returns a fresh record.
        """
        loaded = self._model()
        torch, _ = _import_framework()
        served = {} if self.checkpoint is None else {
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.checkpoint_sha256,
        }
        return {
            "model": self.model_id,
            "revision": self.revision,
            "max_position_embeddings": loaded.context,
            "dtype": loaded.dtype,
            "device": loaded.device,
            "threads": int(torch.get_num_threads()),
            "seed": self.seed,
            "stop_token_ids": list(loaded.stops),
            "versions": {name: _version(name) for name in VERSION_NAMES},
            **served,
        }

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the model's reply to `messages` under `sampling`, with the tokenizer's token counts.

        Calls check() first. Call k (from 0, counted when it starts) samples
        under torch.manual_seed(seed + k). A request whose prompt positions
        (prompt_positions) plus max_tokens pass the context raises
        ContextExceeded before generating;
        nothing is truncated. completion_tokens counts the ids generated, an
        end id included when one stopped generation.
        """
        index = self._calls
        self._calls += 1
        loaded = self._model()
        settings = generation_settings(sampling, loaded.stops, loaded.pad)
        torch, transformers = _import_framework()
        encoded = _encode(loaded.tokenizer, messages)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_positions(prompt_tokens, loaded) + sampling.max_tokens > loaded.context:
            raise ContextExceeded(prompt_positions(prompt_tokens, loaded), sampling.max_tokens, loaded.context)
        encoded = encoded.to(loaded.device)
        torch.manual_seed(self.seed + index)
        with torch.inference_mode():
            output = loaded.model.generate(**encoded, generation_config=transformers.GenerationConfig(**settings))
        new_ids = output[0, prompt_tokens:]
        return Completion(
            text=loaded.tokenizer.decode(new_ids, skip_special_tokens=True),
            prompt_tokens=prompt_tokens,
            completion_tokens=int(new_ids.shape[0]),
        )

    def _model(self) -> _Loaded:
        """Return the loaded model, loading it on the first successful call."""
        if self._loaded is None and self.checkpoint is not None:
            assert self.checkpoint_sha256 is not None
            self._loaded = _load_checkpoint(
                self.model_id, self.revision, self.checkpoint, self.checkpoint_sha256, self._torch_device
            )
        if self._loaded is None:
            self._loaded = _load(self.model_id, self.revision, self._torch_device)
        return self._loaded
