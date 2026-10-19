"""The adversary stage: search for in-contract inputs on which the candidate diverges from the target reference.

Bible: Adversarial Setups (item 1; the adversary rule), Component Interfaces
(Stage; Executor; Oracle; the adversary bullet), Result Record (Trial;
Output files and statistics; Storage), Oracles (binary_io rules), Sandbox
(attempt run limits; model-generated code), Agent Rules 1, 6, 10, and 14.
Plan: plans/p16-adversarial.md, P16.5, and its planning decision "Confirmed
divergence (P16.5)". [DESIGN]

The stage is registered as `adversary` and declares RUNS_PROPOSALS, which
the runner reads (never the stage's name) to pair it with the recipe's
adversary section: a recipe lists it after the oracle stage and binds the
adversary at adversary.kind with budget.inputs, the most input sets the
adversary may propose in one trial (lassi.core.runner refuses, before any
directory exists, the section without the stage, the stage without the
section, and every other choice it cannot carry out). It is built as
factory(context=<RunContext>) once per trial, like every stage.

Which trials it takes: a trial whose standing attempt (lassi.core.record
standing_attempt) ran clean (S5) and passed binary_io on the item's
held-out inputs (an Alignment with statistics per output and mean 1.0)
(taken_attempt). Any other trial comes back unchanged, with Trial.adversary
None: the adversary is not asked and nothing runs.

For a trial it takes:

1. It builds the adversary from the run's registry (RunContext.registry) as
   factory(**config), the config being the adversary section's keys other
   than kind, and builds two programs once each, with the target language's
   toolchain and the item's support files for that language, as
   compile_loop builds an attempt: the target reference from the item's
   target files in <trial>/adversary/reference-build (the baseline keeps no
   program for a later stage), and the candidate from the standing
   attempt's files in <trial>/adversary/candidate-build. The candidate is
   never copied from the attempt's build directory, where run_loop ran it
   with that directory writable, so a program that changed its own file
   there cannot change what the stage runs.
2. It asks the adversary while budget is left and its last reply proposed
   something: each AdversaryView holds the item's InputContract, the source
   program's files, the standing attempt's files, the budget left
   (budget.inputs less the proposals made so far, refused ones included),
   and the earlier proposals' outcomes in order, never a held-out input, the
   item's seed, or a reference output. A reply of more proposals than the
   budget left is refused with a ValueError naming the Agent and its budget,
   and an item that is not exactly a Proposal with a TypeError, before any
   proposal of that reply is checked or runs.
3. Each proposal is checked before any run (Adversarial Setups, the
   adversary rule): its arrays are written, each to its own file in array
   order under <trial>/adversary/proposal<NN>/proposed, and read by
   lassi.harness check_inputs against the contract. An array lassi_io cannot
   write (a bad name or dtype, a negative dim, data that is not its shape's
   size) is refused as unreadable, and one whose shape holds more than
   MAX_INPUT_ELEMENTS elements as too-many-elements, before it is written,
   array by array in order. A refused proposal runs nothing; its outcome
   code is the refusal's reason, and AdversaryRecord.refusals counts it.
4. An accepted set runs the target reference, then the candidate (the
   standing attempt's program, as the stage built it), each once, on the
   target language's bound executor with run_loop's limits
   (lassi.core.stages attempt_limits). Each run has a fresh workdir,
   <trial>/adversary/proposal<NN>/<reference or candidate>, holding the
   program's files, the item's support files for the target language, a
   copy of the built program at its place in its build directory, and
   @inputs written from the proposal by lassi.harness write_inputs (a
   program file under @inputs is dropped first, as stage_inputs drops it);
   no earlier run's outputs are there, so an executor that lists only new
   files (lassi.executors.native) lists each run's own. The
   arguments are program_args, as for the held-out inputs. A backend that
   declares unload_before_run is asked to unload before each run. The
   stage starts no process itself (Agent Rule 6): the executor, which must
   declare runs_code and sandboxed, runs every program, and the runs are not
   profiled.
5. A target reference run that fails is a contract finding (outcome code
   contract-finding; ContractFinding): a simulator gap, a kernel JIT error,
   a hang, undefined behavior, an exit status other than 0 or none, an
   incomplete workdir, or output files the oracle cannot compare against
   (reference_problem). The candidate then does not run, and a finding is
   never a divergence. Otherwise the oracle compares the candidate's output
   files with the reference's: when binary_io fails an output (a missing
   one included, as from a candidate that crashed), the divergence is
   confirmed (outcome code diverged; Divergence, with binary_io's
   OutputStats and candidate_failure, how the candidate's run failed, read
   as the reference's run is read (failed_run), or None when it ran
   clean), and otherwise the outcome code is agreed. The threshold is
   the recipe's oracle metric and number; under threshold from_baseline it
   is the item's declared tolerance (Oracle.with_tolerance), since the
   references' agreement was measured on the held-out inputs only.
6. Each divergence and finding records its inputs as name -> sha256 of the
   proposed lassi_io file in the run's binary store (lassi.core.store
   BlobStore). Trial.adversary (AdversaryRecord) holds the agent's name, its
   resolved config, the proposal count, the refusals by reason, the program
   runs made, the stage's own wall time (from its first build to its last
   run; Final.wall_s, the trial's pipeline time, includes it), the
   divergences, and the findings, both in proposal order.

The stage changes no attempt's stage, alignment, or score, and nothing in
the trial but Trial.adversary (Agent Rule 14: a divergence counts only when
binary_io confirms it, and an adversary's claim is never recorded).

Known cases, not complete:

- A candidate run that fails (a crash or a hang) but whose outputs all
  pass binary_io is agreed, since a divergence is binary_io's verdict (the
  planning decision).
- By the same rule, a candidate run that stopped at a simulator gap, or
  whose workdir came back incomplete (outputs cut by the copy-back cap),
  is a confirmed divergence when an output it left missing or cut fails
  binary_io, as oracle_stage reads an attempt's run. The input may only
  have met a gap or passed a cap, not a wrong value, so the Divergence's
  candidate_failure names how the run ended, and such a divergence is
  never read as an output that merely differs.
- Under metric pcc, binary_io gives pcc 0.0 when either side is constant
  over the compared elements and the two are not equal element for
  element, so a constant or boundary proposal (the fuzzer makes both) can
  turn a one-ULP difference into a divergence; max_abs or ulp is the
  metric for a recipe that binds an adversary.
- A reference run whose workdir came back incomplete is a contract
  finding, though the input may only have passed a cap.
- The rebuilt target reference and candidate are the baseline's and the
  standing attempt's files and toolchain, not their very artifacts.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lassi.bench.registry import INPUTS_DIR, SuiteItem, program_args
from lassi.core.capabilities import ALIGNS_OUTPUT_FILES, OutputFileOracle, declares, unload_before_run
from lassi.core.interfaces import AdversaryView, Executor, Limits, Proposal, ProposalOutcome, RunResult
from lassi.core.record import AdversaryRecord, Attempt, ContractFinding, Divergence, Trial, standing_attempt
from lassi.core.registry import DEFAULT_REGISTRY, Binding, register
from lassi.core.stages import (
    JIT,
    PURPOSE,
    RAN_CLEAN,
    RUNS_CODE,
    SANDBOXED,
    RunContext,
    attempt_limits,
    executor_for,
    output_file_oracle,
)
from lassi.core.store import BlobStore
from lassi.executors.workdir import build_dir
from lassi.harness import InputContract, InputContractError, LassiIOError, check_inputs, write_array, write_inputs
from lassi.harness.contract import MAX_INPUT_ELEMENTS
from lassi.harness.lassi_io import SUFFIX, element_count

# The capability of a stage that runs an adversary's proposals; the runner pairs such a stage with the recipe's
# adversary section and refuses either without the other (lassi.core.runner).
RUNS_PROPOSALS = "runs_proposals"
# The recipe section that binds the adversary, and the binding path of its Agent (lassi.core.recipe _bindings).
ADVERSARY_SECTION = "adversary"
ADVERSARY_KIND = f"{ADVERSARY_SECTION}.kind"
# The outcome codes of a proposal that ran (lassi.core.interfaces ProposalOutcome); a refused proposal's code is its
# check_inputs reason (lassi.harness CONTRACT_REASONS).
AGREED = "agreed"
DIVERGED = "diverged"
CONTRACT_FINDING = "contract-finding"
# The directories under the trial's directory: the stage's root, the reference build, and per proposal its staged
# arrays and the two runs' workdirs.
ADVERSARY_DIR = "adversary"
REFERENCE_BUILD = "reference-build"
CANDIDATE_BUILD = "candidate-build"
PROPOSED = "proposed"
REFERENCE_ROLE = "reference"
CANDIDATE_ROLE = "candidate"
# A path that starts with a drive letter, which is absolute on Windows.
_DRIVE = re.compile(r"[A-Za-z]:")


def adversary_binding(bindings: Sequence[Binding]) -> Binding | None:
    """Return the recipe's adversary binding (the Agent at adversary.kind), or None when it binds none."""
    return next((item for item in bindings if item.interface == "Agent" and item.where == ADVERSARY_KIND), None)


def budget_inputs(config: Mapping[str, Any]) -> int:
    """Return budget.inputs of the adversary section's config: an int of at least 1; ValueError names the key."""
    budget = config.get("budget")
    inputs = budget.get("inputs") if isinstance(budget, Mapping) else None
    if isinstance(inputs, bool) or not isinstance(inputs, int) or inputs < 1:
        raise ValueError(
            f"{ADVERSARY_SECTION}.budget.inputs is {inputs!r}, but the adversary stage asks within it: set an int of "
            "at least 1, the most input sets the adversary may propose in one trial (a required choice, no default)"
        )
    return inputs


def taken_attempt(trial: Trial) -> Attempt | None:
    """Return the standing attempt when the stage takes the trial: it ran clean (S5) and passed binary_io (mean 1.0)."""
    standing = standing_attempt(trial)
    if standing is None or standing.stage_reached != RAN_CLEAN:
        return None
    if standing.alignment.outputs is None or standing.alignment.mean != 1.0:
        return None
    return standing


def failed_run(run: RunResult, where: str, limits: Limits) -> str | None:
    """Return how a run failed, as a message that opens with `where`, or None when it ran clean.

    `where` names the run, such as "the cpp target reference run": the
    message is a contract finding's (ContractFinding.message) for the target
    reference and a divergence's candidate_failure for the candidate. Read
    in run_loop's order: a kernel JIT error first, then a simulator gap, a hang,
    undefined behavior, an exit status other than 0 or none, then an
    incomplete workdir, whose output files may be cut.
    """
    status = "no exit status" if run.exit_code is None else f"exit status {run.exit_code}"
    if run.sim_gap is not None:
        return f"{where} stopped at a simulator gap, {run.sim_gap}"
    if any(item.stage == JIT and item.severity == "error" for item in run.diagnostics):
        return f"{where} reported a kernel JIT error and ended with {status}"
    if run.hang:
        return f"{where} hung past its wall limit of {limits.wall_s:g} s"
    if run.sim_ub is True:
        return f"{where} reported undefined behavior and ended with {status}"
    if run.exit_code != 0:
        return f"{where} ended with {status}"
    if run.workdir_incomplete:
        return f"{where}'s workdir came back incomplete, so its output files may be cut"
    return None


def staged_inputs(arrays: Sequence[Any], directory: Path) -> list[Path]:
    """Write each array to its own file under the new directory `directory`, in order; return the files.

    The files are named by position, so check_inputs reads a repeated name
    as repeated-input. An array whose shape holds more than
    MAX_INPUT_ELEMENTS elements raises InputContractError too-many-elements
    before it is written, and one lassi_io cannot write raises it as
    unreadable; arrays before it stay written.
    """
    directory.mkdir()
    paths = []
    for index, array in enumerate(arrays):
        path = directory / f"{index:02d}{SUFFIX}"
        if min(array.shape, default=0) >= 0 and element_count(array.shape, MAX_INPUT_ELEMENTS) is None:
            raise InputContractError(
                "too-many-elements",
                f"array {index} ({array.name!r}) holds more than {MAX_INPUT_ELEMENTS} elements, the input cap",
                name=array.name,
                path=str(path),
            )
        try:
            write_array(path, array.name, array.dtype, array.shape, array.data)
        except LassiIOError as error:
            raise InputContractError(
                "unreadable", f"array {index} cannot be written as a lassi_io file: {error}", path=str(path)
            ) from None
        paths.append(path)
    return paths


@dataclass(frozen=True)
class _Program:
    """One program the stage runs: its role, its built artifact, the build directory it lies in, and its files.

    `files` holds the program's files and the item's support files for the
    target language (relative path -> text), as its build directory held them.
    """

    role: str
    artifact: Path
    base: Path
    files: Mapping[str, str]

    def copied(self, root: Path) -> Path:
        """Write the files under the new directory `root`, copy the artifact to its place there, and return the copy.

        Each file's path must be relative with no '..' segment, as its build
        checked it; anything else raises ValueError before it is written.
        """
        root.mkdir()
        for path, text in self.files.items():
            if not path or path.startswith("/") or _DRIVE.match(path) or "\\" in path or ".." in path.split("/"):
                raise ValueError(f"the {self.role} file path {path!r} is not a relative path inside its workdir")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        try:
            relative = self.artifact.relative_to(self.base)
        except ValueError:
            relative = Path(self.artifact.name)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.artifact, target)
        return target


@dataclass
class _Search:
    """One taken trial's search: what the stage reads, and the outcomes, runs, and findings it gathers.

    `source` and `attempt` are the source program's files and the standing
    attempt's files, the two an AdversaryView shows.
    """

    context: RunContext
    item: SuiteItem
    contract: InputContract
    oracle: OutputFileOracle
    executor: Executor
    limits: Limits
    root: Path
    reference: _Program
    candidate: _Program
    source: Mapping[str, str]
    attempt: Mapping[str, str]
    runs: int = 0
    outcomes: list[ProposalOutcome] = field(default_factory=list)
    refusals: dict[str, int] = field(default_factory=dict)
    divergences: list[Divergence] = field(default_factory=list)
    findings: list[ContractFinding] = field(default_factory=list)

    def ask(self, agent: Any, binding: Binding, budget: int) -> None:
        """Ask `agent` for proposals within `budget` and take each (module docstring, steps 2 to 6)."""
        while len(self.outcomes) < budget:
            left = budget - len(self.outcomes)
            view = AdversaryView(
                contract=self.contract,
                source=self.source,
                candidate=self.attempt,
                budget_left=left,
                outcomes=tuple(self.outcomes),
            )
            reply = list(agent.propose(view))
            if len(reply) > left:
                raise ValueError(
                    f"Agent {binding.name!r} ({binding.where}) proposed {len(reply)} input sets with {left} of its "
                    f"budget left ({ADVERSARY_SECTION}.budget.inputs {budget}); an adversary proposes at most "
                    "view.budget_left, so none of them ran"
                )
            if not reply:
                return
            for position, proposal in enumerate(reply):
                if type(proposal) is not Proposal:
                    raise TypeError(
                        f"Agent {binding.name!r} ({binding.where}) returned a {type(proposal).__name__} at position "
                        f"{position} of its reply; an adversary returns Proposals, which are data"
                    )
            for proposal in reply:
                code = self.take(len(self.outcomes), proposal)
                self.outcomes.append(ProposalOutcome(proposal=proposal, code=code))

    def take(self, index: int, proposal: Proposal) -> str:
        """Check proposal `index`, run it when accepted, and return its outcome code."""
        folder = self.root / f"proposal{index:02d}"
        folder.mkdir()
        try:
            staged = staged_inputs(proposal.arrays, folder / PROPOSED)
            check_inputs(self.contract, staged)
        except InputContractError as error:
            self.refusals[error.reason] = self.refusals.get(error.reason, 0) + 1
            return error.reason
        target = self.context.direction.target
        reference = self.run(self.reference, proposal, folder)
        problem = failed_run(reference, f"the {target} target reference run", self.limits)
        if problem is None:
            problem = self.oracle.reference_problem(reference.output_files, side=f"{target} target reference")
        if problem is not None:
            self.findings.append(ContractFinding(proposal=index, inputs=self.stored(proposal, staged), message=problem))
            return CONTRACT_FINDING
        candidate = self.run(self.candidate, proposal, folder)
        stats = self.oracle.compare(reference.output_files, candidate.output_files)
        if all(entry.passed for entry in stats):
            return AGREED
        failure = failed_run(candidate, f"the {target} candidate run", self.limits)
        inputs = self.stored(proposal, staged)
        self.divergences.append(Divergence(proposal=index, inputs=inputs, stats=stats, candidate_failure=failure))
        return DIVERGED

    def run(self, program: _Program, proposal: Proposal, folder: Path) -> RunResult:
        """Run `program` on `proposal` in a fresh workdir under `folder`, through the executor; count the run.

        As stage_inputs does, a program file under @inputs is removed before
        the proposal's inputs are written there.
        """
        artifact = program.copied(folder / program.role)
        inputs = artifact.parent / INPUTS_DIR
        if inputs.is_dir():
            shutil.rmtree(inputs)
        elif inputs.exists():
            inputs.unlink()
        inputs.mkdir()
        write_inputs(proposal.arrays, inputs)
        unload_before_run(self.context.backend)
        run = self.executor.run(artifact, program_args(self.item), self.limits)
        self.runs += 1
        return run

    def stored(self, proposal: Proposal, staged: Sequence[Path]) -> dict[str, str]:
        """Put each staged input file in the run's binary store; return input name -> its sha256."""
        blobs = BlobStore(self.context.store.root)
        return {array.name: blobs.put(path.read_bytes()) for array, path in zip(proposal.arrays, staged, strict=True)}

    def record(self, binding: Binding, wall_s: float) -> AdversaryRecord:
        """Return the trial's AdversaryRecord, with `wall_s` as the stage's own wall time."""
        return AdversaryRecord(
            agent=binding.name,
            config=dict(binding.config),
            proposals=len(self.outcomes),
            refusals=dict(self.refusals),
            runs=self.runs,
            wall_s=wall_s,
            divergences=list(self.divergences),
            contract_findings=list(self.findings),
        )


@register("Stage", "adversary")
class AdversaryStage:
    """Asks the bound adversary for input sets, checks each, and runs the target reference and the candidate on it.

    The module docstring states the rules. It declares RUNS_PROPOSALS, which
    the runner reads to pair it with the adversary section, and requires an
    Oracle that declares aligns_output_files, since a divergence counts only
    when binary_io confirms it, and the Toolchain that builds the target
    reference and the candidate.
    """

    name = "adversary"
    capabilities = frozenset({RUNS_PROPOSALS})
    requires = {"Toolchain": frozenset({"diagnostics"}), "Oracle": frozenset({ALIGNS_OUTPUT_FILES})}

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with Trial.adversary set when the stage takes it (taken_attempt), else `trial` itself.

        Raises ValueError when the recipe binds no adversary, its budget is
        not an int of at least 1, or a reply passes the budget left, and
        TypeError for an item of a reply that is not a Proposal.
        """
        standing = taken_attempt(trial)
        if standing is None:
            return trial
        binding = adversary_binding(self.context.recipe.bindings)
        if binding is None:
            raise ValueError(
                f"{trial.trial_id}: the adversary stage asks the Agent bound at {ADVERSARY_KIND}, but the recipe "
                "binds none; set the adversary section"
            )
        budget = budget_inputs(binding.config)
        started = time.monotonic()
        search = self._search(trial, standing)
        search.ask(self._agent(binding), binding, budget)
        return dataclasses.replace(trial, adversary=search.record(binding, time.monotonic() - started))

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        binding = adversary_binding(self.context.recipe.bindings)
        agent = "no bound" if binding is None else repr(binding.name)
        return (
            f"adversary: ask the {agent} adversary for input sets within {ADVERSARY_SECTION}.budget.inputs, refuse "
            f"each outside the item's input contract, and run the {self.context.direction.target} target reference "
            "and the standing attempt on each accepted set; binary_io confirms each divergence"
        )

    def _agent(self, binding: Binding) -> Any:
        """Build the adversary from the run's registry as factory(**config); TypeError when it has no propose()."""
        registry = DEFAULT_REGISTRY if self.context.registry is None else self.context.registry
        agent = registry.get("Agent", binding.name).factory(**binding.config)
        if not callable(getattr(agent, "propose", None)):
            raise TypeError(f"Agent {binding.name!r} ({binding.where}) has no propose(view); it cannot be an adversary")
        return agent

    def _search(self, trial: Trial, standing: Attempt) -> _Search:
        """Return the search over one taken trial: its contract, oracle, executor, limits, and the two programs."""
        context = self.context
        item = context.suite.item(context.item, purpose=PURPOSE)
        if item.contract is None:
            raise ValueError(
                f"{context.suite.name}/{context.item} declares no held-out inputs, so it has no input contract to "
                "check a proposal against"
            )
        executor = self._executor(trial)
        root = build_dir(context.build_root, trial.trial_id, 0).parent.parent / ADVERSARY_DIR
        root.mkdir(parents=True)
        target = context.direction.target
        support = context.suite.support_files(context.item, context.sources_root, purpose=PURPOSE, language=target)
        source = context.suite.source_files(context.item, context.direction, context.sources_root, purpose=PURPOSE)
        return _Search(
            context=context,
            item=item,
            contract=item.contract,
            oracle=self._oracle(item),
            executor=executor,
            limits=attempt_limits(context, trial),
            root=root,
            reference=self._reference(root, support),
            candidate=self._built(CANDIDATE_ROLE, standing.files, support, root / CANDIDATE_BUILD),
            source=source,
            attempt=standing.files,
        )

    def _executor(self, trial: Trial) -> Executor:
        """Return the target language's executor; ValueError unless it runs programs and declares sandboxed."""
        executor = executor_for(self.context, self.context.direction.target)
        if not declares(executor, RUNS_CODE):
            raise ValueError(
                f"{trial.trial_id}: the target language's executor runs no program ({RUNS_CODE!r} not declared), "
                "so the adversary stage cannot run a proposal"
            )
        if not declares(executor, SANDBOXED):
            raise ValueError(
                f"{trial.trial_id}: the executor runs programs but does not declare {SANDBOXED!r}; the adversary "
                "stage runs the candidate, model-generated code, only in the sandbox (Agent Rule 6)"
            )
        return executor

    def _oracle(self, item: SuiteItem) -> OutputFileOracle:
        """Return the recipe's output-file Oracle, bound to the item's declared tolerance under from_baseline."""
        oracle = output_file_oracle(self.context)
        if oracle is None:
            raise ValueError(f"the adversary stage needs an Oracle that declares {ALIGNS_OUTPUT_FILES!r}")
        if oracle.agreement_setting is None:
            return oracle
        if item.tolerance is None:
            raise ValueError(
                f"{oracle.agreement_setting}: the adversary stage judges a proposal against the item's declared "
                f"tolerance, and {self.context.suite.name}/{item.name} declares none"
            )
        return oracle.with_tolerance(item.tolerance)

    def _reference(self, root: Path, support: Mapping[str, str]) -> _Program:
        """Build the item's target reference once in <root>/reference-build (_built)."""
        context = self.context
        files = context.suite.reference_target(context.item, context.direction, context.sources_root, purpose=PURPOSE)
        return self._built(REFERENCE_ROLE, files, support, root / REFERENCE_BUILD)

    def _built(self, role: str, files: Mapping[str, str], support: Mapping[str, str], workdir: Path) -> _Program:
        """Build `files` in the new directory `workdir` with the target toolchain, as compile_loop builds an attempt.

        The item's support files go in as the toolchain's `harness` argument,
        only when there are some. Raises ValueError when no toolchain is
        bound for the target language or the build gives no program, though
        the baseline or compile_loop built the same files with it.
        """
        target = self.context.direction.target
        toolchain = self.context.toolchains.get(target)
        if toolchain is None:
            raise ValueError(f"no toolchain is bound for the target language {target!r}; set toolchain.{target}")
        workdir.mkdir()
        result = toolchain.build(files, workdir, harness=support) if support else toolchain.build(files, workdir)
        if result.artifact is None:
            errors = "; ".join(item.message for item in result.diagnostics if item.severity == "error")
            raise ValueError(
                f"the {target} {role} did not build in {workdir}, though an earlier stage built the same files with "
                f"the same toolchain: {errors or 'no error was reported'}"
            )
        return _Program(role, Path(result.artifact), workdir, {**support, **files})
