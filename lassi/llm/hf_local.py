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
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from lassi.core.capabilities import MODEL_CHECK, TAKES_DEVICE
from lassi.core.devices import DeviceSpec, FrameworkBuild, parse_device
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
_SEED_LIMIT = 2**32


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
    """A loaded model: tokenizer, model, context, stop ids, pad id, torch device, and dtype."""

    tokenizer: Any
    model: Any
    context: int
    stops: tuple[int, ...]
    pad: int
    device: str
    dtype: str


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
    return _Loaded(tokenizer, model, context, stops, int(pad), device, dtype)


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


@register("LLMBackend", "hf_local")
class HFLocalBackend:
    """Serves a model in process on Transformers and PyTorch, loaded once by check() from HF_HOME at a pinned revision.

    Construction checks the settings and that the framework is installed,
    and loads nothing. complete() calls check() first.
    """

    name = "hf_local"
    capabilities = frozenset({"chat", MODEL_CHECK, TAKES_DEVICE})
    # The model-section keys a recipe may set beside model.device; the runner passes them as keyword settings.
    config_keys = frozenset({"revision", "seed"})

    def __init__(
        self,
        model_id: str,
        *,
        device: Mapping[str, Any] | None = None,
        revision: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Keep the checked settings; load nothing.

        Raises FrameworkMissing (an ImportError) when torch or transformers is
        not installed, found without importing either; then ValueError naming
        model.id, model.revision, model.seed, or model.device for a missing or
        malformed setting.
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
        installed versions of VERSION_NAMES (null when one is not installed).
        Each call returns a fresh record.
        """
        loaded = self._model()
        torch, _ = _import_framework()
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
        }

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the model's reply to `messages` under `sampling`, with the tokenizer's token counts.

        Calls check() first. Call k (from 0, counted when it starts) samples
        under torch.manual_seed(seed + k). A request whose prompt tokens plus
        max_tokens pass the context raises ContextExceeded before generating;
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
        if prompt_tokens + sampling.max_tokens > loaded.context:
            raise ContextExceeded(prompt_tokens, sampling.max_tokens, loaded.context)
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
        if self._loaded is None:
            self._loaded = _load(self.model_id, self.revision, self._torch_device)
        return self._loaded
