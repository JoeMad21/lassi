"""Generate the tt-metalium context pack from the pinned tt-metal tree (owner direction of 2026-10-23).

Usage:

    uv run tools/build_tt_context.py --tree DIR [--out DIR] [--name NAME --limit-bytes N]

`--tree` is tt-metal at commit 5280a9cfb00998fd49667a29523d03aee905c129
(toolchains/tt-metal.pin): the installed tree on the build host
($LASSI_TOOLCHAINS/tt-metal@5280a9cf), or a directory of files fetched from
GitHub at exactly that commit, each at its tree path
(https://raw.githubusercontent.com/tenstorrent/tt-metal/<commit>/<path>). It
must hold every file SOURCES names, every Tier A kernel
assets/bench/tt-pairs-v0.yaml names, and every file TIER_A_EXAMPLE_FILES
names, each with the sha256 recorded there. Anything else is refused with
exit status 1, a message, and no file written.

It writes `context/<NAME>/` under `--out` (default assets/; NAME defaults to
PACK_NAME, tt-metalium-5280a9cf): MANIFEST.yaml (plain ASCII, LF) and one
`<key>.txt` per fragment, which
lassi.prompts joins in manifest order into the pack that upstream's
summarize_context stage sends for the target language tt. No choice of
content below looks at a Tier A item, its programs, or any result on it
(Agent Rule 5); Tier A files are read only to keep their text out.

- metalium_guide.tt: METALIUM_GUIDE.md, the guide at the tree's root. The
  rule, by heading: the guide's title, and each `##` part GUIDE_PARTS names
  (the Executive Summary, and TT-Metalium, the part on the SDK a host
  program calls) with every section under it, except a section
  GUIDE_LEFT_OUT names and the sections under that one (the compute kernel
  internals, which describe code inside a kernel, not a host program).
  Every code block and image is left out; so are the sentences that point
  at text the pack leaves out (GUIDE_DANGLING) and, under Agent Rule 5, the
  guide's vector addition walkthrough of the eltwise_binary example and its
  paragraph on split_work_to_cores, nearly word for word a comment of the
  matmul_multi_core example (GUIDE_CUTS).
- host_api.tt: the host API reference, by fixed rules from the prompt set
  tt-host-v0 and the include walk, with nothing learned from a Tier A item
  or a result on one. Caveat: generate.txt itself was written with Tier A's
  reference host programs in view (docs/BIBLE.md, the Decision Log entry of
  2026-10-06), so the pack adds nothing the prompt does not already show,
  and results with it carry the same label: not held-out results on any
  Tier A item later assigned to eval. The rules:
  1. The walk (include_closure): the ROOTS headers, the tt-metalium
     includes generate.txt already tells the model a tt program uses
     (host_api.hpp, distributed.hpp, tensor_accessor_args.hpp, and
     tilize_utils.hpp, in its order), then the tt-metalium public headers
     they include, directly or transitively, within API_DIR, in
     breadth-first include order from all roots together.
  2. The order (walk_order): the roots; then, for each name prompt_names
     reads from generate.txt at build time (every identifier the template
     uses that some header of the walk defines, in the order of its first
     use there, no name chosen by hand), the first header of the walk whose
     declarations define it; then the rest of the walk.
  3. The text (api_part): every header, the roots included, as
     declarations only, its comments dropped (declarations,
     comments=False). Each part follows its include line. Left out of each:
     every comment, function bodies, constructor initializer lists, members
     that are not public, friend, forward, deprecated, and explicitly
     specialized declarations, `detail` namespaces, includes, pragmas,
     license lines, and Doxygen's main page.
  The fragment's own text, its preamble and the line format before each
  header, is this repository's, in HEAD_FILE; HEAD_FILE and TEMPLATE_FILE
  are its first sources.

Size. The whole pack, the guide and the reference as the loader joins them,
fits LIMIT_BYTES (40 KiB) by a fixed rule over that order, which looks at
nothing but header sizes (fit_reference): the roots are always kept, and
each later header, in order, is kept whole when the pack still fits with it
and skipped otherwise, the fill going on to the next. The manifest names
every skipped header in `cut_by_limit`. A guide and roots that alone pass
the limit are refused, naming the sizes.

A fuller pack (2026-10-25). `--name` and `--limit-bytes` build a second pack
by the same rules from the same tree, head file, and template, under
`context/<NAME>/`, with `--limit-bytes` in place of LIMIT_BYTES: FULL_NAME
with FULL_LIMIT_BYTES (128 KiB) gives the declaration lookup
(lassi.core.stages, pack_declarations) the headers the default pack's limit
cuts. NAME is PACK_NAME, or PACK_NAME followed by one or more
`-<lowercase letters and digits>` parts, so it names the pin; any other
name, or a limit below 1, is refused with exit status 1 and nothing
written. Without the options the tool writes the default pack exactly as
before. The two manifests differ only in limit_bytes and in the
reference's sources, sha256, and cut_by_limit; both are tracked, and both
packs' fragments are gitignored.

Agent Rule 5. No source is a programming example (the Tier A items are
tt-metal's programming examples loopback, eltwise_binary, eltwise_sfpu,
matmul_single_core, and matmul_multi_core). Before writing, every fragment
is checked against every Tier A reference file (tier_a_references: the
tracked Tier A programs, the kernels the suite names, the example host
programs and walkthroughs TIER_A_EXAMPLE_FILES names, and every other file
under each example's directory in the tree): a fragment holding RUN_LINES or
more consecutive non-blank lines of a reference file, each line compared
with its whitespace collapsed, is refused (tier_a_runs).

Only the manifest and HEAD_FILE are tracked; the fragments are gitignored,
as OQ-018 keeps upstream text out of the repository. The same tree and head
file give the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lassi.prompts.assets import PACK_JOINER  # noqa: E402

DEFAULT_OUT = REPO / "assets"
PACK_NAME = "tt-metalium-5280a9cf"
PACK_TREE = f"context/{PACK_NAME}"
# The fuller pack (the docstring's "A fuller pack"): its name and its limit, which the tool takes as options.
FULL_NAME = f"{PACK_NAME}-full"
FULL_LIMIT_BYTES = 128 * 1024
# A pack name the tool accepts: PACK_NAME, then optional `-<lowercase letters and digits>` parts.
_PACK_NAME_RULE = re.compile(rf"{re.escape(PACK_NAME)}(?:-[a-z0-9]+)*")
MANIFEST = "MANIFEST.yaml"
GENERATOR = "tools/build_tt_context.py"
PIN_FILE = REPO / "toolchains" / "tt-metal.pin"
REPO_URL = "https://github.com/tenstorrent/tt-metal.git"
# tt-metal's pinned commit (toolchains/tt-metal.pin COMMIT; bible, Toolchain Pins). SOURCES holds its bytes.
PIN = "5280a9cfb00998fd49667a29523d03aee905c129"
LANGUAGE = "tt"
# The whole pack's size limit: the guide's sections, PACK_JOINER, and the declarations, filled in walk order with
# whole headers, a header that does not fit skipped. The arithmetic is PROJECTED, not measured: no served model's
# tokenizer has counted this text. The bible's Serving Rules set max_model_len >= 32,768 tokens for LASSI-DF
# context packs, and the context recipes bound a reply at max_tokens 4,096 (qwen3-coder-30b, llama31-8b) or 16,384
# (exaone4-32b, a reasoning model), which leaves a prompt 28,672 or 16,384 tokens at that floor. At r bytes per
# token the pack takes 40,960 / r tokens, so the pack alone passes those budgets only when r falls below 1.43 or
# 2.50 bytes per token. At an assumed r of 3.0 it takes 13,654 tokens and leaves 15,018 or 2,730 tokens for the
# rest of the prompt (the request's own text, or a generate prompt's source and kernels); a served window above
# the floor leaves more. The fuller pack's FULL_LIMIT_BYTES (131,072) takes 43,691 tokens at the same assumed r,
# past a 32,768-token window, so a recipe that names it gives its text to the declaration lookup, whose
# lookup.max_chars bounds what a prompt carries, and lists no stage that sends the whole pack (summarize_context).
LIMIT_BYTES = 40 * 1024
# The pack's own text (OQ-018 restricts only upstream text, so it is tracked): the reference's preamble, then the
# line each header's declarations follow, holding {include}. Its sha256 is a source of host_api.tt.
HEAD_FILE = REPO / "assets" / "context" / "tt-metalium-5280a9cf.head.txt"
HEAD_SOURCE = HEAD_FILE.relative_to(REPO).as_posix()
INCLUDE_SLOT = "{include}"
# Agent Rule 5: the Tier A suite, its tracked programs, and the run length that counts as a copy.
TIER_A_MANIFEST = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
TIER_A_TRACKED = REPO / "assets" / "bench" / "tt-pairs-v0"
EXAMPLES = "tt_metal/programming_examples/"
# Example directories the Tier A examples share: both matmul examples include matmul_common's bmm_op.hpp.
SHARED_EXAMPLE_DIRS = ("tt_metal/programming_examples/matmul/matmul_common",)
# The Tier A examples' host programs, walkthroughs, and shared host header, which the Agent Rule 5 check needs
# beside the kernels the suite names; a tree without one, or with other bytes, is refused. Each sha256 was read
# from the file at PIN on GitHub on 2026-10-23, and each directory holds no README.md there.
TIER_A_EXAMPLE_FILES = {
    f"{EXAMPLES}loopback/loopback.cpp": "f7c217730c7ff5a4d518f8806b60424b2c3b34dbc71ad2874c708320d085b2b7",
    f"{EXAMPLES}loopback/dram_loopback.md": "aa610a5a3b814bc924b85e68f2aa3027d253fbf97ced32b0b0872376d0c0cc38",
    f"{EXAMPLES}eltwise_binary/eltwise_binary.cpp": "82b11f7ab766228ff4a7e94a3f26d414ae14ca8c55079e0a3b790fa8b3c0297b",
    f"{EXAMPLES}eltwise_binary/eltwise_binary.md": "b5c634edfd46930924e36453bbf926d690784216d3dfe59661f9312a83a112d3",
    f"{EXAMPLES}eltwise_sfpu/eltwise_sfpu.cpp": "c8a5b835c9c2abb1afe50fadf0cb5416de62fb2fd8d38d3da78b7fddddb7a513",
    f"{EXAMPLES}eltwise_sfpu/eltwise_sfpu.md": "3f4c6f4216ebda7ff80d74c4244ee487a06f8f73d3c5d89a976f13f856334727",
    f"{EXAMPLES}matmul/matmul_single_core/matmul_single_core.cpp":
        "8ebd1c486e14fa01131a199b6e0dc37ac4846008444b0c918552ba2c904bdcd5",
    f"{EXAMPLES}matmul/matmul_single_core/matmul_single_core.md":
        "095c5016f0b8f5a4e9eb44a646d6b611f477131e9bdc1a4b4e3507685e8ade2a",
    f"{EXAMPLES}matmul/matmul_multi_core/matmul_multi_core.cpp":
        "f564c60d483bfc8f30050535092b3bee2baa337a7948ade7b0f36d6184cb7843",
    f"{EXAMPLES}matmul/matmul_multi_core/matmul_multi_core.md":
        "d8dfe4f1fd43e96e4398dc8d7338be1cf4f9e288ae59aeb3160ed19b49cfe33a",
    f"{EXAMPLES}matmul/matmul_common/bmm_op.hpp": "0b924f53a37fd90be5ec858165fa2c8d66199123e3b704847dce6db38e9d00f4",
}
RUN_LINES = 3

GUIDE = "METALIUM_GUIDE.md"
GUIDE_KEY = f"metalium_guide.{LANGUAGE}"
API_KEY = f"host_api.{LANGUAGE}"
# The guide's `##` parts the pack keeps, each with every section under it: the summary of the whole guide, and the
# part on TT-Metalium, the SDK a host program calls. Left out: the table of contents and the hardware overview.
GUIDE_PARTS = ("## Executive Summary", "## TT-Metalium")
# A kept part's sections left out with the sections under them: the compute kernel internals (and their
# `#### Compute APIs`), which describe code inside a compute kernel, not a host program.
GUIDE_LEFT_OUT = ("### Register control and Data Flow within the Compute Kernels",)
# Blocks cut from a kept section, as (first, last) index ranges, inclusive, among the section's paragraphs, images,
# and code blocks at the pin. Agent Rule 5: the vector addition walkthrough is the eltwise_binary programming
# example (Tier A item eltwise_binary): its host program in four parts with the text that narrates each (2 to 10).
# Agent Rule 5: the SPMD section's paragraph on split_work_to_cores (4; METALIUM_GUIDE.md:581) is nearly word for
# word a comment of Tier A's matmul_multi_core.cpp (lines 115 to 120 at the pin).
GUIDE_CUTS = {"### Running code on device": ((2, 10),), "### SPMD in Metalium": ((4, 4),)}
# Sentences of kept paragraphs that point at text the pack leaves out, each removed: the kernels of the hardware
# overview, and the code block an example introduces.
GUIDE_DANGLING = {
    "## TT-Metalium": ("The kernels shown in previous sections are written using Metalium.",),
    "### SPMD in Metalium": ("For example:",),
}
# The host API: the include root and the directory the tt-metalium includes resolve in.
INCLUDE_ROOT = "tt_metal/api/"
API_DIR = f"{INCLUDE_ROOT}tt-metalium/"
# The generate template of the prompt set tt-host-v0, what the prompt already shows the model. Its tt-metalium
# includes are ROOTS (a test checks it), and the names it uses that a header of the walk defines are read from it
# at build time (prompt_names), so neither list is chosen by hand, and the rule adds nothing learned from a Tier A
# item or a result on one (Agent Rule 5). Caveat: the template itself was written with Tier A's reference host
# programs in view (docs/BIBLE.md, the Decision Log entry of 2026-10-06), so the pack adds nothing the prompt does
# not show, and results with it carry the same label: not held-out results on a Tier A item later assigned to eval.
# Its sha256 is a source of host_api.tt.
TEMPLATE_FILE = REPO / "assets" / "prompts" / "tt-host-v0" / "generate.txt"
TEMPLATE_SOURCE = TEMPLATE_FILE.relative_to(REPO).as_posix()
# The root headers of the include walk: the tt-metalium includes TEMPLATE_FILE tells the model a tt program uses,
# in the order it names them.
ROOTS = tuple(f"{API_DIR}{name}" for name in (
    "host_api.hpp",
    "distributed.hpp",
    "tensor_accessor_args.hpp",
    "tilize_utils.hpp",
))
# An identifier of the template: a word that starts with a letter or an underscore, not inside another word.
_IDENTIFIER = re.compile(r"(?<!\w)[A-Za-z_]\w*")
# The sha256 of every source at PIN, read from the files at the commit on GitHub on 2026-10-23: the guide, then
# include_closure(ROOTS) in its order (the Tier A kernels fetched the same way match the sha256 the build host
# read, so the two routes give the same bytes).
SOURCES = {
    GUIDE: "130d1ea0b46e3ed3a225b047fe0a5127a5a95f78e339de4caaf1151176aa8f5f",
    f"{API_DIR}host_api.hpp": "a0d52ebf3c9d6c86b8e3fba06abb14a0eb6e9d36f94f6ea525359f24b3e0a839",
    f"{API_DIR}distributed.hpp": "8bf745877dcb96c789751679052da43a59885b8c8c06387470eeedcd2082ae87",
    f"{API_DIR}tensor_accessor_args.hpp": "f7f57c4988f2dd6c19e56a52b8cf49fdb96cb8074d840606359e3cd07141b5ea",
    f"{API_DIR}tilize_utils.hpp": "9997ec3290e4b66a29feef40eafd19126fc99e447bc19ac8cb6e88beb490650a",
    f"{API_DIR}buffer.hpp": "c42a7a186fe5dafd61aae48a69682f559a3df31d2b932c1e662b14e34e4ae4c1",
    f"{API_DIR}circular_buffer.hpp": "d2633bf17b3b828a220122096390104102a041ab1a00e524bf0385a7d0e1ae45",
    f"{API_DIR}dispatch_core_common.hpp": "10ed4a0574cf6d1ce66d5461970c8ae9d06d347cbf6ad7efaef238046b6ac038",
    f"{API_DIR}runtime_args_data.hpp": "70311b1e15cfd83a8c4696ce2f9d5951d162071d94cf4287d12e24cd601cbb4b",
    f"{API_DIR}program.hpp": "2981c71ec7241b4bcdd32166e89d6b5ec988d09f8f2521535a002625f9a19c79",
    f"{API_DIR}device.hpp": "10bc837e111153999720b83702b247f6c29e3bdd2b807b2c439625e284b16b08",
    f"{API_DIR}sub_device_types.hpp": "f27f05d50c7148bcb43875e5c01bbfc73bd5fa8d77905cebf30844fc6266d18c",
    f"{API_DIR}profiler_types.hpp": "ad18a25377e2ab42df5a6d11d3464e7ed852983a931b29c87951cb4a926928aa",
    f"{API_DIR}profiler_optional_metadata.hpp": "53a02d6249eedac01cd8f228faf2619d0459faba374202a203a5f85d0915de2c",
    f"{API_DIR}kernel_types.hpp": "5ab62b703c24deb40cb99b8bd25713583a8d43edc5564dc23c181e0b799a77e2",
    f"{API_DIR}circular_buffer_config.hpp": "3041963a13c8e42558dfd66569b2e21fcbd751af44bf6686d96d5877a9e5b336",
    f"{API_DIR}mesh_buffer.hpp": "bd2bf16410a161d2483567ea307ab1e45ba4a8917b9d759d265535580fd5ca0e",
    f"{API_DIR}mesh_command_queue.hpp": "757296d0fe6c96a87c1ca07f438b62e5014c7706251f0dea5cb188e6c0f47ea2",
    f"{API_DIR}mesh_coord.hpp": "14c8447d88db22f6f4e827bfe4d7f03c021a495177a0c89fb0f3e90864c972ce",
    f"{API_DIR}mesh_event.hpp": "773f3ae480aaccc743777fb8d6c4e388c5b56462b9572254334bd94cb67f284e",
    f"{API_DIR}experimental/sockets/mesh_socket.hpp":
        "cdaa628382a0aaffe4fd851b5c6d32961c774cb34dbca3c698cdaa8ca8fc7ea8",
    f"{API_DIR}mesh_trace_id.hpp": "fc7bf3ed484c0f06094166a28a974b635219aa9bc26587d5a086b39ebfd3e685",
    f"{API_DIR}mesh_workload.hpp": "f4fd0a158a70057916b814c366afb56449c3cf0254553c6fbfaa185bbc76ca3d",
    f"{API_DIR}bfloat16.hpp": "ee55c50e99fd2fefe4cfd0cfcf06d18e4e580602f25129ae4f6f1b8b08a3deb2",
    f"{API_DIR}buffer_types.hpp": "43615ab8235365c3d0253ad65969c8b3e689550135956050914195e71c750930",
    f"{API_DIR}buffer_distribution_spec.hpp": "c6580590cb33b646232da3121b35cdebabf3c09c73251e2b574600993fe338d6",
    f"{API_DIR}core_coord.hpp": "d43f27b5a41976529ab2a209333471273e843e516a7480e1c5c5eacfeda19778",
    f"{API_DIR}hal_types.hpp": "c22128eb2166497b065fd8d64379e3a338fc0926f297f750f8feea3461698fee",
    f"{API_DIR}buffer_page_mapping.hpp": "fd7834951833f83903ab6768a099043d954e0711dac97126439ecba354d11b73",
    f"{API_DIR}tt_backend_api_types.hpp": "6fae209a25ddf5fdd4132496d5f04076e3c5f4e39856df2ee1bf5bb620fa680d",
    f"{API_DIR}circular_buffer_constants.h": "1c1190293f6a4c95eadb324ec6d3919d219b40eab5c3b3c63bf5f5522b1c2a45",
    f"{API_DIR}data_types.hpp": "54f71d1d97b754a9fb91589f6bd83e975a4c91ecfb3c92fefbaee023be0e15bb",
    f"{API_DIR}base_types.hpp": "7b3a740a1f96c5b968070b4f997b06ac965c75f1d7bcf47a14f55b9cc6dcb64d",
    f"{API_DIR}tile.hpp": "5f60d23bf3cbbc7b64ab9167e8959954bd540bf8736d9d0315b9179e710ead03",
    f"{API_DIR}program_descriptors.hpp": "2efc4f08c2b97fca419a71ca4eb139b518bb3df4da669f366314a8dbc8819d22",
    f"{API_DIR}mesh_device.hpp": "6c4c6f3c64c2e33fbd0216916e3f546d4d71cec855aebf1f4d0d138723402fea",
    f"{API_DIR}mesh_device_view.hpp": "568fde760a39b363bc9cce8ca72b6df1d3dc0f1ccbbf39851782022d0f8e9d51",
    f"{API_DIR}shape2d.hpp": "8061dd77cf0e114b86d747e63eb40d9097170d7a05f5ef79a84d7fa436d831d4",
    f"{API_DIR}distributed_host_buffer.hpp": "f5bb006a5383555e1b0a65bb971ea5f366cf9649baf3b0f3b586daee6a6d129a",
    f"{API_DIR}vector_aligned.hpp": "4c0c8c761e8524142cf81ebe60125a35327bd34eefb05153abf695780bd19d90",
    f"{API_DIR}shape_base.hpp": "f342acee57bcd9100e704e591ea53d563c1c74abaf8739cb5d1c518986aeecc0",
    f"{API_DIR}maybe_remote.hpp": "0251db932cfaf29daa5edb54818fc51805f804c3075dfe1fa03a10bf6b15df15",
    f"{API_DIR}distributed_context.hpp": "d0bfccc2d375e55bd47aee55d618a94e9302da22bc9a582dd740708f87194e1c",
    f"{API_DIR}experimental/fabric/routing_table_generator.hpp":
        "28c4eed3b9b992d900ebdfbff57248d085a2dfe2459c25778c7ff2c20ff876c8",
    f"{API_DIR}shape.hpp": "e81db1410e3ce41b2b7a6bb21ad94fd094ec5f11792d7d41b48ff04bb50df457",
    f"{API_DIR}constants.hpp": "41b3df42358d602a604d660e5e7b75bd6b26c637f465d37dcceb0326e540e8bd",
    f"{API_DIR}math.hpp": "8ae3aebfbd127f88e7ed34d94eeef0cd5d4fa3c3673819a8090881cf6ede956a",
    f"{API_DIR}mesh_config.hpp": "97fdc432dabbebf833ac2986fd8a1562256841369a878b12e56a0dffdd596211",
    f"{API_DIR}host_buffer.hpp": "7bd3c4570ea4d85e0bac727daf5eeff205c71d5f80e63fc43742423c10938b0d",
    f"{API_DIR}experimental/fabric/mesh_graph.hpp": "22486bf5b47dec29182a1336ddec208508abcc79a552e803a677c7b9e49b15c5",
    f"{API_DIR}experimental/fabric/fabric_types.hpp":
        "ad2f185730ef2903e2d9a46c0e86d44028f78456380830c74bebb49de766c2cc",
    f"{API_DIR}memory_pin.hpp": "db6e6a33e04e6e50c14cc9797a292e8c35b9c6a0b40160be2bb48792e6b953c9",
    f"{API_DIR}experimental/fabric/mesh_graph_descriptor.hpp":
        "cb24f651040a0101b9151229f7efa144fe84b6fc53cb91e03293e4341393454d",
}

HEADER = (
    f"# Generated by {GENERATOR} from tt-metal at {PIN} (toolchains/tt-metal.pin).\n"
    "# Only this manifest and the head file are tracked; the fragment files beside it are generated and\n"
    "# gitignored, as OQ-018 keeps upstream text out of the repository. A source under assets/ is this\n"
    "# repository's own text; every other source is a tt-metal path at the commit. No source is a programming\n"
    f"# example, and no fragment holds a run of {RUN_LINES} or more lines of a Tier A file (Agent Rule 5; the\n"
    "# tool's docstring). cut_by_limit names, in fill order, every header skipped as it did not fit limit_bytes.\n"
)

# A tt-metalium include: `#include <tt-metalium/...>` or `#include "tt-metalium/..."` at the start of a line.
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"](tt-metalium/[^>"]+)[>"]', re.M)
# A brace opens a scope (kept, and parsed inside) when its statement starts with one of these before any '('.
_SCOPE = re.compile(r"\b(namespace|class|struct|union|enum)\b|\bextern\b\s*\"")
_ACCESS = re.compile(r"^\s*(public|private|protected)\s*:(?!:)")
_OPERATOR = re.compile(r"\boperator\s*(\(\s*\)|\[\s*\]|[^\s(]+)")
_FORWARD = re.compile(r"(template\s*<.*>\s*)?(class|struct|union)\s+[\w:]+\s*;", re.S)
_SPECIALIZATION = re.compile(r"template\s*<\s*>")
_DETAIL = re.compile(r"namespace\s+(\w+::)*detail\s*\{")
# Doxygen's front page, which describes no declaration.
_MAINPAGE = re.compile(r"/\*\*\s*\\mainpage.*?\*/\n?", re.S)
_TABLE_ROW = re.compile(r"^(\s*(?:\*\s*|//\s*)?)(\|.*\|)\s*$")
# The doc table columns the reference leaves out: the declaration under the table gives each argument's type, and
# its default arguments say which are optional.
TABLE_DROPPED_COLUMNS = ("Type", "Required")


class Refusal(RuntimeError):
    """The tree is not tt-metal at the pin, or the pack would break Agent Rule 5 or the size limit."""


@dataclass(frozen=True)
class Fragment:
    """One fragment of the pack: its manifest key, the source paths it is drawn from, and its text."""

    key: str
    sources: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Head:
    """The reference's own text from HEAD_FILE: its preamble, the include line format, and the file's sha256."""

    preamble: str
    include_line: str
    sha256: str

    def line(self, include: str) -> str:
        """Return the line that heads the declarations of `include` (such as tt-metalium/host_api.hpp)."""
        return self.include_line.replace(INCLUDE_SLOT, include)


# ---------------------------------------------------------------- C++ text


def code_mask(text: str) -> str:
    """Return `text` with comments, string and char literal contents, and preprocessor lines blanked.

    Every blanked character becomes a space except newlines, so offsets and
    line numbers stay those of `text`.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        directive = text[i] == "#" and not text[text.rfind("\n", 0, i) + 1:i].strip()
        if text.startswith("//", i) or directive:
            end = text.find("\n", i)
            end = n if end < 0 else end
            _blank(out, i, end)
            i = end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            _blank(out, i, end)
            i = end
        elif text[i] in "\"'":
            end = i + 1
            while end < n and text[end] not in (text[i], "\n"):
                end += 2 if text[end] == "\\" else 1
            _blank(out, i + 1, min(end, n))
            i = end + 1
        else:
            i += 1
    return "".join(out)


def _blank(chars: list[str], start: int, end: int) -> None:
    """Replace chars[start:end] with spaces, newlines kept."""
    for k in range(start, end):
        if chars[k] != "\n":
            chars[k] = " "


def matching(mask: str, start: int) -> int:
    """Return the offset just past the (), [], or {} group that opens at `start` in a code mask."""
    depth = 0
    for k in range(start, len(mask)):
        if mask[k] in "([{":
            depth += 1
        elif mask[k] in ")]}":
            depth -= 1
            if depth == 0:
                return k + 1
    raise Refusal(f"an unbalanced bracket at offset {start}; the header is not the C++ this tool reads")


def _flat(head: str) -> str:
    """Return `head` with the inside of every top-level (), [], and {} group removed, the brackets kept."""
    out, depth = [], 0
    for ch in head:
        if ch in "([{":
            if depth == 0:
                out.append(ch)
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                out.append(ch)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _without_templates(flat: str) -> str:
    """Return `flat` with each `template <...>` parameter list removed, so a default argument there is no '='."""
    out = flat
    while True:
        found = re.search(r"\btemplate\s*<", out)
        if found is None:
            return out
        depth, k = 0, found.end() - 1
        while k < len(out):
            depth += {"<": 1, ">": -1}.get(out[k], 0)
            if depth == 0:
                break
            k += 1
        out = out[:found.start()] + out[k + 1:]


def _blank_access(head: str) -> str:
    """Return `head` with a leading access specifier (`public:` and the like) blanked, offsets kept."""
    found = _ACCESS.match(head)
    return head if found is None else " " * found.end() + head[found.end():]


def brace_kind(head: str) -> str:
    """Return what a top-level '{' opens, from its statement's code mask `head`: scope, value, or body.

    A scope is a namespace, class, struct, union, enum, or extern block. A
    value is an initializer: the statement has a top-level '=' (operator
    names and template parameter lists aside) or no parameter list. Anything
    else is a function body.
    """
    plain = _without_templates(_OPERATOR.sub("operator", _flat(_blank_access(head))))
    if _SCOPE.search(plain.split("(", 1)[0]):
        return "scope"
    if "=" in plain or ")" not in plain:
        return "value"
    return "body"


def _init_colon(head: str) -> int | None:
    """Return the offset in `head` of a constructor initializer list's ':' (top level, after a ')'), or None."""
    depth, closed = 0, False
    for k, ch in enumerate(head):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            closed = closed or (depth == 0 and ch == ")")
        elif ch == ":" and depth == 0 and closed and head[k - 1] != ":" and head[k + 1:k + 2] != ":":
            return k
    return None


def _body_brace(mask: str, stmt: int, start: int) -> int:
    """Return the offset of the body's '{' after an initializer list: the first one after a ')' or a '}'."""
    k = start
    while k < len(mask):
        if mask[k] == "{":
            if mask[stmt:k].rstrip()[-1:] in (")", "}"):
                return k
            k = matching(mask, k)
        elif mask[k] in "([":
            k = matching(mask, k)
        else:
            k += 1
    raise Refusal(f"no function body follows the initializer list at offset {start}")


def strip_bodies(text: str) -> str:
    """Return `text` with every function body, and any constructor initializer list, replaced by ';'.

    Braces in comments, strings, and parameter lists are not read; scopes
    are kept and read inside; initializers are kept whole.
    """
    mask = code_mask(text)
    cuts: list[tuple[int, int]] = []
    i = stmt = 0
    while i < len(text):
        ch = mask[i]
        if ch in ";}":
            i = stmt = i + 1
        elif ch in "([":
            i = matching(mask, i)
        elif ch != "{":
            i += 1
        elif (kind := brace_kind(mask[stmt:i])) == "scope":
            i = stmt = i + 1
        elif kind == "value":
            i = matching(mask, i)
        else:
            colon = _init_colon(_blank_access(mask[stmt:i]))
            body = i if colon is None else _body_brace(mask, stmt, i)
            start, end = (body if colon is None else stmt + colon), matching(mask, body)
            while start > stmt and text[start - 1] in " \t\n":
                start -= 1
            after = end
            while after < len(text) and mask[after] in " \t":
                after += 1
            end = after + 1 if mask[after:after + 1] == ";" else end
            cuts.append((start, end))
            i = stmt = end
    pieces, last = [], 0
    for start, end in cuts:
        pieces.extend((text[last:start], ";"))
        last = end
    pieces.append(text[last:])
    return "".join(pieces)


@dataclass(frozen=True)
class Unit:
    """A piece of body-free header text: decl (ends at ';'), open ('{' of a scope), close, access, or tail."""

    kind: str
    text: str


def units(text: str) -> list[Unit]:
    """Split body-free header text into Units; each keeps the comments and spaces before it."""
    mask = code_mask(text)
    found: list[Unit] = []
    i = start = 0
    n = len(text)
    while i < n:
        ch = mask[i]
        if ch in "([":
            i = matching(mask, i)
        elif ch == ";":
            found.append(Unit("decl", text[start:i + 1]))
            i = start = i + 1
        elif ch == ":" and _ACCESS.match(mask[max(start, mask.rfind("\n", 0, i) + 1):i + 2]):
            found.append(Unit("access", text[start:i + 1]))
            i = start = i + 1
        elif ch == "{" and brace_kind(mask[start:i]) == "scope":
            found.append(Unit("open", text[start:i + 1]))
            i = start = i + 1
        elif ch == "{":
            i = matching(mask, i)
        elif ch == "}":
            end = i + 1
            while end < n and mask[end] in " \t":
                end += 1
            end += 1 if mask[end:end + 1] == ";" else 0
            line_end = text.find("\n", end)
            line_end = n if line_end < 0 else line_end
            end = line_end if text[end:line_end].strip().startswith("//") else end
            found.append(Unit("close", text[start:end]))
            i = start = end
        else:
            i += 1
    found.append(Unit("tail", text[start:]))
    return found


def _code(text: str) -> str:
    """Return the code of `text` (comments and literals blanked) with its whitespace collapsed."""
    return " ".join(code_mask(text).split())


def _left_out(code: str) -> bool:
    """Return True for a declaration the reference leaves out: deprecated, friend, forward, or a specialization.

    An override stays: the reference may hold a derived class without its
    base (MeshDevice without IDevice), so the overrides are where a caller
    reads those members.
    """
    return ("[[deprecated" in code or code.startswith("friend ") or bool(_FORWARD.fullmatch(code))
            or bool(_SPECIALIZATION.match(code)))


@dataclass
class _Frame:
    """One open scope while public_units reads: the access in force, and whether the scope itself is hidden."""

    access: str
    hidden: bool


def public_units(found: Sequence[Unit]) -> list[Unit]:
    """Return the Units a caller of the API sees: public members only, and none of _left_out's declarations."""
    kept: list[Unit] = []
    scopes: list[_Frame] = []

    def hidden() -> bool:
        """Return True while the text read now lies outside the public part of a shown scope."""
        return bool(scopes) and (scopes[-1].access != "public" or scopes[-1].hidden)

    for unit in found:
        code = _code(unit.text)
        if unit.kind == "open":
            scopes.append(_open_scope(code, hidden()))
            if not scopes[-1].hidden:
                kept.append(unit)
        elif unit.kind == "close":
            frame = scopes.pop() if scopes else _Frame("public", False)
            if not frame.hidden:
                kept.append(unit)
        elif unit.kind == "access":
            if scopes:
                scopes[-1].access = code.rstrip(":").strip()
            if not hidden():
                kept.append(unit)
        elif unit.kind == "tail" or not (hidden() or _left_out(code)):
            kept.append(unit)
    return _without_empty_scopes(kept)


def _open_scope(code: str, outer_hidden: bool) -> _Frame:
    """Return the frame of a scope that opens with `code`: its first access, and whether it is hidden.

    A class starts private and anything else public; a deprecated scope, an
    explicit specialization (a formatter, hash, or JSON converter), and a
    `detail` namespace (tt-metal's internal, unstable APIs) are hidden, and so
    is a scope inside a hidden one.
    """
    head = _flat(code).split("(", 1)[0]
    is_class = bool(re.search(r"\bclass\b", head)) and not re.search(r"\benum\b", head)
    left_out = "[[deprecated" in code or bool(_SPECIALIZATION.match(code) or _DETAIL.match(code))
    return _Frame("private" if is_class else "public", outer_hidden or left_out)


def _without_empty_scopes(found: list[Unit]) -> list[Unit]:
    """Return `found` without any scope left holding nothing (an open Unit followed by a bare close)."""
    out = list(found)
    k = 0
    while k < len(out) - 1:
        if out[k].kind == "open" and out[k + 1].kind == "close" and _code(out[k + 1].text).startswith("}"):
            del out[k:k + 2]
            k = max(k - 1, 0)
        else:
            k += 1
    return out


def _pruned(text: str) -> str:
    """Return header text without its license comment, includes, pragmas, clang-format and @file lines, and mainpage."""
    lines = _MAINPAGE.sub("", text.replace("\r\n", "\n")).split("\n")
    lead = 0
    while lead < len(lines) and lines[lead].lstrip().startswith("//"):
        lead += 1
    if any("SPDX-" in line for line in lines[:lead]):
        lines = lines[lead:]
    drop = re.compile(r"\s*(#\s*(include|pragma)\b|// clang-format (on|off)\s*$|/\*\* @file \*/\s*$)")
    return "\n".join(line for line in lines if not drop.match(line))


def _tidy(text: str) -> str:
    """Return reference text with doc tables compacted, trailing spaces gone, and at most one blank line in a row.

    A table keeps the columns of its header row but TABLE_DROPPED_COLUMNS,
    which the declaration below it states; a separator row and a cell past
    the header row's columns are left out, and cells keep single spaces.
    """
    lines: list[str] = []
    kept_columns: list[int] | None = None
    for line in text.split("\n"):
        row = _TABLE_ROW.match(line)
        if row is None:
            kept_columns = None
            lines.append(line.rstrip())
            continue
        cells = [cell.strip() for cell in row.group(2).strip()[1:-1].split("|")]
        if set(row.group(2)) <= set("|-: "):
            continue
        if kept_columns is None:
            kept_columns = [k for k, name in enumerate(cells) if name not in TABLE_DROPPED_COLUMNS]
        shown = [cells[k] if k < len(cells) else "" for k in kept_columns]
        lines.append(row.group(1) + re.sub(r" {2,}", " ", "| " + " | ".join(shown) + " |"))
    joined = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
    return joined + "\n"


def strip_comments(text: str) -> str:
    """Return `text` with every // and /* */ comment removed; string and char literals and newlines stay."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif text[i] in "\"'":
            end = i + 1
            while end < n and text[end] not in (text[i], "\n"):
                end += 2 if text[end] == "\\" else 1
            out.append(text[i:end + 1])
            i = end + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def declarations(text: str, comments: bool = True) -> str:
    """Return the declarations of one header as the API reference gives them, with their doc comments or without.

    Without comments, the same declarations are extracted and then every
    comment is removed (strip_comments), so only the code is left.
    """
    body_free = strip_bodies(_pruned(text))
    kept = "".join(unit.text for unit in public_units(units(body_free)))
    return _tidy(kept if comments else strip_comments(kept))


# ---------------------------------------------------------------- the host API headers and the reference


def include_closure(texts: Mapping[str, str], roots: Sequence[str] = ROOTS) -> list[str]:
    """Return `roots` and the tt-metalium headers they include, transitively, in breadth-first include order.

    The walk starts from all roots together: the roots come first, in their
    order, then the headers they include, level by level. `texts` maps tree
    paths to header text. Each `#include <tt-metalium/X>` (or
    "tt-metalium/X") line of a header read resolves to INCLUDE_ROOT + X, a
    path within API_DIR; any other include (the standard library, tt_stl,
    umd, fmt) is not followed. Each header is listed once, where its first
    include reaches it: a header's includes are queued in the order its
    lines give them, after those already queued. Raises Refusal when a root
    or an include they reach is not in `texts`, naming the header and the
    file that includes it.
    """
    missing = [root for root in roots if root not in texts]
    if missing:
        raise Refusal(f"SOURCES records no sha256 for the root header {missing[0]}")
    order: list[str] = []
    queue, seen = list(dict.fromkeys(roots)), set(roots)
    while queue:
        path = queue.pop(0)
        order.append(path)
        for name in _INCLUDE.findall(texts[path]):
            target = INCLUDE_ROOT + name
            if target in seen:
                continue
            if target not in texts:
                raise Refusal(f"{path} includes <{name}>, and SOURCES records no sha256 for {target}; fetch it at "
                              f"{PIN} and record it")
            seen.add(target)
            queue.append(target)
    return order


def read_head(path: Path = HEAD_FILE) -> Head:
    """Return the Head in `path`: every line above the one line holding INCLUDE_SLOT, then that line.

    The file is plain ASCII with LF newlines, ends with its include line and
    a newline, and holds INCLUDE_SLOT once; anything else raises Refusal.
    """
    data = path.read_bytes()
    where = f"the head file {path.name}"
    if not data.isascii() or b"\r" in data or not data.endswith(b"\n"):
        raise Refusal(f"{where} must be plain ASCII with LF newlines, ending in a newline")
    text = data.decode("ascii")
    lines = text[:-1].split("\n")
    if text.count(INCLUDE_SLOT) != 1 or INCLUDE_SLOT not in lines[-1] or len(lines) < 2:
        raise Refusal(f"{where} must end with the one line that holds {INCLUDE_SLOT}, below its preamble")
    preamble = "\n".join(lines[:-1]) + "\n"
    return Head(preamble=preamble, include_line=lines[-1], sha256=hashlib.sha256(data).hexdigest())


def api_part(path: str, text: str, head: Head) -> str:
    """Return one header's part of the reference: its include line, a blank line, and its declarations only.

    Every header, a root included, goes in without its comments
    (declarations, comments=False).
    """
    return f"{head.line(path.split(INCLUDE_ROOT, 1)[-1])}\n\n{declarations(text, comments=False)}"


def api_text(headers: Sequence[tuple[str, str]], head: Head) -> str:
    """Return the API reference: the head's preamble, then each (path, text) header's part, a blank line apart."""
    return "\n".join([head.preamble, *(api_part(path, text, head) for path, text in headers)])


def defines(declared: str, name: str) -> bool:
    """Return True when declarations text `declared` defines `name`.

    A definition is a class, struct, union, or enum with a body, after an
    optional base list or underlying type (`class MeshDevice : ... {`,
    `enum class BufferType {`), or a `using name =` alias. A forward or
    opaque declaration (`class MeshDevice;`, `enum class DataFormat :
    uint8_t;`) is not one; comments and literals are not read.
    """
    return _definition(name).search(code_mask(declared)) is not None


def _definition(name: str) -> re.Pattern[str]:
    """Return the pattern that finds a definition of `name` in a code mask (defines)."""
    word = re.escape(name)
    return re.compile(rf"\b(?:class|struct|union|enum(?:\s+(?:class|struct))?)\s+(?:\[\[[^\]]*\]\]\s*)?{word}\b"
                      rf"\s*(?:final\b\s*)?(?::(?!:)[^;{{]*)?\{{|\busing\s+{word}\s*=")


def _defining_header(masks: Mapping[str, str], name: str) -> str | None:
    """Return the first header of `masks` {path: code mask of its declarations}, in order, that defines `name`."""
    pattern = _definition(name)
    return next((path for path, mask in masks.items() if name in mask and pattern.search(mask)), None)


def declared_headers(texts: Mapping[str, str], closure: Sequence[str]) -> dict[str, str]:
    """Return {path: declarations only (comments=False)} of each header of the walk `closure`, in its order."""
    return {path: declarations(texts[path], comments=False) for path in closure}


def prompt_names(template: str, declared: Mapping[str, str]) -> list[str]:
    """Return every identifier `template` uses that a header of the walk defines, in the order of first use.

    The rule is fixed from the prompt and the include walk, with no name
    chosen by hand: the template's identifiers (_IDENTIFIER), each counted
    once where it first appears, kept when the declarations of some header
    in `declared` (declared_headers) define it, as defines() reads them.
    """
    masks = {path: code_mask(text) for path, text in declared.items()}
    return [name for name in dict.fromkeys(_IDENTIFIER.findall(template))
            if _defining_header(masks, name) is not None]


def walk_order(
    declared: Mapping[str, str], roots: Sequence[str] = ROOTS, names: Sequence[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """Return the order the reference fills in, and {name: the header chosen for it}.

    `declared` is declared_headers of the walk, in walk order. The rule is
    fixed from the prompt and the include walk: the roots, then for each of
    `names` (prompt_names) in turn the first header of the walk whose
    declarations define it (when that header is not already listed), then
    the rest of the walk in its order. Raises Refusal when no header of the
    walk defines a name.
    """
    masks = {path: code_mask(text) for path, text in declared.items()}
    chosen: dict[str, str] = {}
    for name in names:
        header = _defining_header(masks, name)
        if header is None:
            raise Refusal(f"no header the include walk reaches defines {name}, which the generate prompt names")
        chosen[name] = header
    first = list(dict.fromkeys([*roots, *chosen.values()]))
    return [*first, *(path for path in declared if path not in first)], chosen


def _reference(head: Head, kept: Sequence[tuple[str, str]]) -> Fragment:
    """Return the reference fragment of the kept (path, part) pairs: the head's preamble, then each part."""
    return Fragment(API_KEY, (HEAD_SOURCE, TEMPLATE_SOURCE, *(path for path, _ in kept)),
                    "\n".join([head.preamble, *(part for _, part in kept)]))


def fit_reference(
    guide: Fragment, headers: Sequence[tuple[str, str]], head: Head, limit: int = LIMIT_BYTES, keep: int = 1,
) -> tuple[Fragment, tuple[str, ...]]:
    """Return the reference that fills `limit` in the order of `headers`, and the paths of the headers skipped.

    The rule is fixed over the include walk: every header goes in as
    declarations only (api_part); the first `keep` headers (the roots) are
    always kept; each later header, in order, is kept when the pack with
    `guide` still fits `limit` with it and skipped otherwise, the fill going
    on to the next header. The skipped paths keep their order. Raises
    Refusal, naming the sizes, when the guide and the roots alone leave the
    pack over the limit.
    """
    floor = max(keep, 1)
    if len(headers) < floor:
        raise Refusal(f"the reference needs at least {floor} headers, and {len(headers)} were given")
    parts = [(path, api_part(path, text, head)) for path, text in headers]
    roots_size = pack_size([guide, _reference(head, parts[:floor])])
    if roots_size > limit:
        raise Refusal(f"the guide ({len(guide.text.encode('utf-8'))} bytes) and the first {floor} headers "
                      f"{[path for path, _ in parts[:floor]]} make a {roots_size}-byte pack, over the "
                      f"{limit}-byte limit")
    kept, skipped = list(parts[:floor]), []
    for path, part in parts[floor:]:
        if pack_size([guide, _reference(head, [*kept, (path, part)])]) <= limit:
            kept.append((path, part))
        else:
            skipped.append(path)
    return _reference(head, kept), tuple(skipped)


# ---------------------------------------------------------------- the guide


def guide_blocks(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return the guide as (heading line, its blocks) per section; a block is (kind, text).

    The kinds are para (a run of non-blank lines), image (an `<img` line),
    and code (a fenced block). Text before the first heading belongs to the
    heading "".
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    sections: list[tuple[str, list[tuple[str, str]]]] = [("", [])]
    k = 0
    while k < len(lines):
        line = lines[k]
        if not line.strip():
            k += 1
        elif line.startswith("#"):
            sections.append((line, []))
            k += 1
        elif line.lstrip().startswith("```"):
            end = k + 1
            while end < len(lines) and not lines[end].lstrip().startswith("```"):
                end += 1
            sections[-1][1].append(("code", "\n".join(lines[k:end + 1])))
            k = end + 1
        elif line.lstrip().startswith("<img"):
            sections[-1][1].append(("image", line))
            k += 1
        else:
            end = k
            while end < len(lines) and lines[end].strip() and not lines[end].startswith("#") \
                    and not lines[end].lstrip().startswith(("```", "<img")):
                end += 1
            sections[-1][1].append(("para", "\n".join(lines[k:end])))
            k = end
    return sections if sections[0][1] else sections[1:]


def _level(heading: str) -> int:
    """Return a Markdown heading's level: the number of '#' it starts with (0 for the text before any heading)."""
    return len(heading) - len(heading.lstrip("#"))


def kept_headings(headings: Sequence[str], parts: Sequence[str], left_out: Sequence[str]) -> list[str]:
    """Return the headings the guide rule keeps, in order: each title, and each heading of a part in `parts`.

    A title is a level-1 heading; a part is a level-2 heading and the deeper
    headings that follow it. A heading in `left_out` is dropped with the
    deeper headings that follow it.
    """
    kept: list[str] = []
    in_part, skip_below = False, 0
    for heading in headings:
        level = _level(heading)
        if skip_below and level > skip_below:
            continue
        skip_below = 0
        if level <= 1:
            in_part = False
            kept.extend([heading] if level == 1 else [])
            continue
        if level == 2:
            in_part = heading in parts
        if in_part and heading in left_out:
            skip_below = level
        elif in_part:
            kept.append(heading)
    return kept


def _without_sentence(paragraphs: list[str], sentence: str, heading: str) -> list[str]:
    """Return `paragraphs` with `sentence` and the space before (or after) it removed, and any paragraph left empty.

    Raises Refusal unless the sentence occurs exactly once among them.
    """
    count = sum(paragraph.count(sentence) for paragraph in paragraphs)
    if count != 1:
        raise Refusal(f"{GUIDE}: expected the sentence {sentence!r} once in {heading!r}, found it {count} times")
    out = []
    for paragraph in paragraphs:
        if sentence in paragraph:
            variant = next(form for form in (f" {sentence}", f"{sentence} ", sentence) if form in paragraph)
            paragraph = paragraph.replace(variant, "", 1)
        if paragraph.strip():
            out.append(paragraph)
    return out


def _check_guide_names(
    headings: Sequence[str], kept: Sequence[str], named: Mapping[str, Sequence[str]],
) -> None:
    """Raise Refusal unless each named heading is in the guide once and the cut and dangling ones are kept.

    `named` maps "parts", "left_out", "cuts", and "dangling" to the headings
    each names.
    """
    for role, wanted in named.items():
        for heading in wanted:
            if headings.count(heading) != 1:
                raise Refusal(f"{GUIDE}: expected one heading {heading!r} ({role}), found {headings.count(heading)}")
            if role in ("cuts", "dangling") and heading not in kept:
                raise Refusal(f"{GUIDE}: {heading!r} ({role}) is not a section the guide rule keeps")


def guide_text(
    text: str,
    parts: Sequence[str] = GUIDE_PARTS,
    left_out: Sequence[str] = GUIDE_LEFT_OUT,
    cuts: Mapping[str, Sequence[tuple[int, int]]] = GUIDE_CUTS,
    dangling: Mapping[str, Sequence[str]] = GUIDE_DANGLING,
) -> str:
    """Return the guide fragment: each kept section's heading and paragraphs (kept_headings), the rest left out.

    Images, code blocks, the `cuts` of a section, and its `dangling`
    sentences are left out. Raises Refusal when a heading named here is not
    in the guide exactly once, a cut or dangling heading is not kept, a cut
    lies outside its section's blocks, or a dangling sentence is not in its
    section's kept paragraphs exactly once.
    """
    parsed = guide_blocks(text)
    headings = [heading for heading, _ in parsed]
    kept = kept_headings(headings, parts, left_out)
    _check_guide_names(headings, kept, {"parts": parts, "left_out": left_out, "cuts": tuple(cuts),
                                        "dangling": tuple(dangling)})
    out = []
    for heading, blocks in parsed:
        if heading not in kept:
            continue
        cut: set[int] = set()
        for low, high in cuts.get(heading, ()):
            if not 0 <= low <= high < len(blocks):
                raise Refusal(f"{GUIDE}: the cut {low} to {high} of {heading!r} lies outside its {len(blocks)} blocks")
            cut.update(range(low, high + 1))
        paragraphs = [body for index, (kind, body) in enumerate(blocks) if kind == "para" and index not in cut]
        for sentence in dangling.get(heading, ()):
            paragraphs = _without_sentence(paragraphs, sentence, heading)
        out.append("\n\n".join([heading, *paragraphs]))
    return "\n\n".join(out) + "\n"


# ---------------------------------------------------------------- Agent Rule 5 and the size limit


def _run_lines(text: str) -> list[tuple[int, str]]:
    """Return (line number, line with its whitespace collapsed) for every non-blank line of `text`."""
    lines = text.replace("\r\n", "\n").split("\n")
    return [(number, " ".join(line.split())) for number, line in enumerate(lines, 1) if line.strip()]


def tier_a_runs(fragment: str, references: Mapping[str, str], width: int = RUN_LINES) -> list[str]:
    """Return one message per place where `fragment` holds `width` consecutive non-blank lines of a reference.

    Lines are compared with their whitespace collapsed; blank lines on
    either side are skipped, so a run need not be contiguous in the file.
    """
    runs: dict[tuple[str, ...], str] = {}
    for label, text in references.items():
        lines = [line for _, line in _run_lines(text)]
        for k in range(len(lines) - width + 1):
            runs.setdefault(tuple(lines[k:k + width]), label)
    numbered = _run_lines(fragment)
    hits = []
    for k in range(len(numbered) - width + 1):
        run = tuple(line for _, line in numbered[k:k + width])
        if run in runs:
            hits.append(f"line {numbered[k][0]}: {width} lines of {runs[run]}, from {run[0][:60]!r}")
    return hits


def _pinned_files(tree: Path, files: Mapping[str, str], what: str) -> dict[str, str]:
    """Return {path: text} of `files` {path: sha256} in `tree`; raise Refusal naming each one missing or off the pin."""
    texts: dict[str, str] = {}
    problems = []
    for path, digest in sorted(files.items()):
        file = tree / path
        if not file.is_file():
            problems.append(f"{path} is missing")
        elif hashlib.sha256(data := file.read_bytes()).hexdigest() != digest:
            problems.append(f"{path} has another sha256 than at the pin")
        else:
            texts[path] = data.decode("utf-8", errors="replace")
    if problems:
        raise Refusal(f"{tree} lacks {what} the Agent Rule 5 check reads; fetch each at {PIN}: {'; '.join(problems)}")
    return texts


def tier_a_references(tree: Path) -> dict[str, str]:
    """Return {label: text} of every Tier A reference file: the tracked programs and the tree's example files.

    The tracked programs are every file under assets/bench/tt-pairs-v0/.
    From the tree: each kernel the suite names and each file
    TIER_A_EXAMPLE_FILES names (all required, each checked against its
    sha256), and every other file under each example's directory (the
    directory above `kernels/` in a kernel's path) and SHARED_EXAMPLE_DIRS.
    Raises Refusal when the suite is at another commit or a required file
    is missing or not the pinned bytes, naming each such file.
    """
    references = {path.relative_to(REPO).as_posix(): _read_text(path)
                  for path in sorted(TIER_A_TRACKED.rglob("*")) if path.is_file()}
    suite = yaml.safe_load(TIER_A_MANIFEST.read_text(encoding="utf-8"))
    if suite.get("commit") != PIN:
        raise Refusal(f"{TIER_A_MANIFEST.name} is at {suite.get('commit')}, not the pin {PIN}")
    kernels = {spec["upstream"]: spec["sha256"] for item in suite["items"].values()
               for language in item["languages"].values()
               for spec in language.get("support", {}).values() if "upstream" in spec}
    references.update(_pinned_files(tree, {**kernels, **TIER_A_EXAMPLE_FILES}, "Tier A files"))
    folders = sorted({path.split("/kernels/", 1)[0] for path in kernels} | set(SHARED_EXAMPLE_DIRS))
    for folder in folders:
        for path in sorted((tree / folder).rglob("*")) if (tree / folder).is_dir() else []:
            if path.is_file():
                references.setdefault(path.relative_to(tree).as_posix(), _read_text(path))
    return references


def _read_text(path: Path) -> str:
    """Return a file's text, undecodable bytes replaced, for comparison only."""
    return path.read_bytes().decode("utf-8", errors="replace")


def pack_size(fragments: Sequence[Fragment]) -> int:
    """Return the bytes of the pack the loader joins from `fragments` (lassi.prompts.assets PACK_JOINER)."""
    return len(PACK_JOINER.join(fragment.text for fragment in fragments).encode("utf-8"))


def check_pack(fragments: Sequence[Fragment], references: Mapping[str, str], limit: int = LIMIT_BYTES) -> None:
    """Raise Refusal when a fragment draws on a programming example or holds a Tier A run, or the pack is too big."""
    for fragment in fragments:
        examples = [path for path in fragment.sources if path.startswith(EXAMPLES)]
        if examples:
            raise Refusal(f"{fragment.key} draws on programming examples {examples} (Agent Rule 5)")
        hits = tier_a_runs(fragment.text, references)
        if hits:
            raise Refusal(f"{fragment.key} holds {RUN_LINES} or more consecutive lines of a Tier A file "
                          f"(Agent Rule 5): {'; '.join(hits[:3])}")
    size = pack_size(fragments)
    if size > limit:
        raise Refusal(f"the pack is {size} bytes, over the {limit}-byte limit; skip whole headers rather than "
                      "cut one")


# ---------------------------------------------------------------- reading, building, writing


def check_pin_file() -> None:
    """Raise Refusal unless toolchains/tt-metal.pin's COMMIT is PIN, the commit SOURCES was read at."""
    found = re.search(r"^COMMIT=(\S+)$", PIN_FILE.read_text(encoding="utf-8"), flags=re.M)
    if found is None or found.group(1) != PIN:
        raise Refusal(f"{PIN_FILE.name} pins {found.group(1) if found else 'no COMMIT'}, but this tool reads "
                      f"{PIN}; update PIN and SOURCES for the new pin")


def read_sources(tree: Path) -> dict[str, str]:
    """Return {path: text} of every source in `tree`; raise Refusal naming each file missing or off the pin."""
    texts: dict[str, str] = {}
    problems = []
    for path, digest in SOURCES.items():
        file = tree / path
        if not file.is_file():
            problems.append(f"{path} is missing")
            continue
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            problems.append(f"{path} has another sha256 than at the pin")
            continue
        texts[path] = data.decode("utf-8")
    if problems:
        raise Refusal(f"{tree} is not tt-metal at {PIN}: {'; '.join(problems)}")
    return texts


def headers_of(sources: Mapping[str, str]) -> list[str]:
    """Return include_closure of the sources, refusing when SOURCES lists a header the closure does not reach."""
    closure = include_closure(sources)
    unreached = [path for path in sources if path != GUIDE and path not in closure]
    if unreached:
        raise Refusal(f"SOURCES records {unreached}, which the includes of ROOTS do not reach")
    return closure


@dataclass(frozen=True)
class Pack:
    """A built pack: its fragments, the sha256 of this repository's own sources, its limit, and what that skipped."""

    fragments: tuple[Fragment, ...]
    own: Mapping[str, str]
    cut: tuple[str, ...]
    limit: int = LIMIT_BYTES


def read_template(path: Path = TEMPLATE_FILE) -> tuple[str, str]:
    """Return (text, sha256) of the generate template; raise Refusal when it is not UTF-8."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8"), hashlib.sha256(data).hexdigest()
    except UnicodeDecodeError as error:
        raise Refusal(f"the generate template {path.name} is not UTF-8 ({error})") from error


def build(
    tree: Path, head_file: Path = HEAD_FILE, template_file: Path = TEMPLATE_FILE, limit: int = LIMIT_BYTES,
) -> Pack:
    """Return the pack built from `tree`, `head_file`, and `template_file` to `limit` bytes, checked by check_pack.

    The names the template uses that the walk defines (prompt_names) put
    their defining headers right after the roots (walk_order); the check is
    against tier_a_references(tree).
    """
    check_pin_file()
    sources = read_sources(tree)
    head = read_head(head_file)
    template, template_sha256 = read_template(template_file)
    guide = Fragment(GUIDE_KEY, (GUIDE,), guide_text(sources[GUIDE]))
    declared = declared_headers(sources, headers_of(sources))
    order, _ = walk_order(declared, names=prompt_names(template, declared))
    reference, cut = fit_reference(guide, [(path, sources[path]) for path in order], head, limit, keep=len(ROOTS))
    fragments = (guide, reference)
    check_pack(fragments, tier_a_references(tree), limit)
    return Pack(fragments, {HEAD_SOURCE: head.sha256, TEMPLATE_SOURCE: template_sha256}, cut, limit)


def manifest_text(
    fragments: Sequence[Fragment], hashes: Mapping[str, str], cut: Sequence[str] = (), limit: int = LIMIT_BYTES,
) -> str:
    """Return MANIFEST.yaml: the header, the pin and `limit`, each entry's key, sources, and sha256, and the cut."""
    entries = [{
        "key": fragment.key,
        "sources": [{"path": path, "sha256": hashes[path]} for path in fragment.sources],
        "sha256": hashlib.sha256(fragment.text.encode("utf-8")).hexdigest(),
    } for fragment in fragments]
    document = {"generator": GENERATOR, "repo": REPO_URL, "commit": PIN, "language": LANGUAGE,
                "limit_bytes": limit, "entries": entries, "cut_by_limit": list(cut)}
    return HEADER + yaml.safe_dump(document, sort_keys=False, default_flow_style=False, width=120)


def pack_tree(name: str = PACK_NAME) -> str:
    """Return the assets-relative directory of the pack `name`: context/<name>."""
    return f"context/{name}"


def check_options(name: str, limit: int) -> None:
    """Raise Refusal unless `name` follows _PACK_NAME_RULE and `limit` is at least 1, naming the option."""
    if not _PACK_NAME_RULE.fullmatch(name):
        raise Refusal(f"--name {name!r} is not {PACK_NAME} or {PACK_NAME} with -<lowercase letters and digits> "
                      f"parts, such as {FULL_NAME}")
    if limit < 1:
        raise Refusal(f"--limit-bytes {limit} is below 1; the default pack's is {LIMIT_BYTES}, the fuller pack's "
                      f"{FULL_LIMIT_BYTES}")


def write_pack(out: Path, pack: Pack, name: str = PACK_NAME) -> Path:
    """Write the fragments and MANIFEST.yaml under `out`/context/`name`, each file only if it changed; return it."""
    tree = out / pack_tree(name)
    tree.mkdir(parents=True, exist_ok=True)
    files = {f"{fragment.key}.txt": fragment.text.encode("utf-8") for fragment in pack.fragments}
    hashes = {**SOURCES, **pack.own}
    files[MANIFEST] = manifest_text(pack.fragments, hashes, pack.cut, pack.limit).encode("ascii")
    for name, data in files.items():
        path = tree / name
        if not (path.is_file() and path.read_bytes() == data):
            path.write_bytes(data)
    return tree


def main(argv: list[str] | None = None) -> int:
    """Build the pack from --tree and write it under --out; return 0, or 1 on a refusal (nothing written)."""
    parser = argparse.ArgumentParser(description="Generate the tt-metalium context pack from tt-metal at the pin.")
    parser.add_argument("--tree", type=Path, required=True, help="tt-metal at the pin: an install or fetched files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="assets root to write the pack under")
    parser.add_argument("--name", default=PACK_NAME, help=f"the pack's name (default {PACK_NAME}; or {FULL_NAME})")
    parser.add_argument("--limit-bytes", type=int, default=LIMIT_BYTES,
                        help=f"the whole pack's byte limit (default {LIMIT_BYTES}; {FULL_LIMIT_BYTES} for {FULL_NAME})")
    args = parser.parse_args(argv)
    try:
        check_options(args.name, args.limit_bytes)
        pack = build(args.tree, limit=args.limit_bytes)
    except Refusal as refusal:
        print(f"build_tt_context: refused: {refusal}", file=sys.stderr)
        return 1
    tree = write_pack(args.out, pack, args.name)
    sizes = ", ".join(f"{fragment.key} {len(fragment.text.encode('utf-8'))} bytes" for fragment in pack.fragments)
    print(f"{tree.as_posix()}: {sizes}; the pack {pack_size(pack.fragments)} of {pack.limit} bytes; "
          f"{len(pack.cut)} headers cut by the limit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
