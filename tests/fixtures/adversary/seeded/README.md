# Seeded faulty candidates of adv-seeded-v0 (SYNTHETIC)

Every file here is a synthetic fixture of task P16.8 (plans/p16-adversarial.md, the planning decisions "Gate
subjects (P16.8)", "Freeze (P16.10)", and "Gate reading (P16.G)"). Each candidate was written by hand from its
item's reference program (assets/bench/adv-seeded-v0/<item>/cpp/<item>.cpp) with one fault seeded in it. None is
model output, none is upstream code or prompt text (OQ-018), and no value here is a measurement. The candidates
are replay fixtures outside every suite: no suite lists them and no held-out input comes from them (Agent Rule 5).

## Files

- `candidates/<item>/<id>.cpp`: each candidate's one file, the `<item>.cpp` it returns as its reply. Its first line
  names it SYNTHETIC; the rest is the reference with exactly one change.
- `seeded.json`: the replay backend's recording for the seeded recipes (projects/adversary-seeded/seeded-fuzzer.yaml
  and seeded-llm.yaml, `model.recording`): reply text only, never `messages`, one completion per trial in the
  runner's trial order (the one direction cpp -> cpp, the items sorted, run 1 then run 2), each completion the
  candidate's FILE block (lassi.core.files render_file_blocks). The recipes set loop.max_corrections to 0, so no
  trial asks for a correction and each takes exactly one completion, as long as every trial reaches generate: a
  trial that ends before it (a baseline failure) takes none and shifts every later reply, since one backend
  serves the run.
- `index.json`: one entry per candidate, in recording order: its id, item, run (its trial's run number), class,
  fault (what changed), trigger (the condition its witness was written from), witness (an in-contract input set in
  the llm adversary's input form, lassi.agents.llm), file, and sha256 (of its completion, the reply).
- `break-contract.json`: the llm adversary's recording for projects/adversary-seeded/seeded-llm.yaml: one reply of
  eight proposals, each outside every adv-seeded-v0 item's contract (no input; an input named zz and no x; x as
  f128, as f64, of shape [3], of shape [2000000], given twice, and filled with 200000000), so check_inputs refuses
  every one before any run, then a reply proposing nothing more.

## Fault classes (known cases, not complete)

Two candidates each. Every fault fires only on inputs the item's held-out inputs never hold and its contract
allows. Most fire at a value the fuzzer's strategies reach as a constant array or in its specials pattern (0 and
-0, +1, a negative value, a large magnitude, a NaN beside a number); precision-blocks fires only on mixed values
(the fuzzer's fresh draws, sweeps, and seeded mixes), since a constant array keeps its block sums exact. So the
gate can hold the fuzzer to finding each (P16.G), which task P16.10's runs first show. Each trigger below states
the fault's condition, known cases, not complete: some inputs that meet it still stay within the oracle's
threshold (for sign-abs and sign-unary, a negative x of magnitude below about 2.5e-7; for arith-divide, x within
about 1.7e-4 of -1); each witness was checked by hand to diverge.

- boundary: relational operator replacement (ROR), one of the mutation operators found sufficient in Offutt, Lee,
  Rothermel, Untch, and Zapf, "An Experimental Determination of Sufficient Mutant Operators", ACM TOSEM 5(2):99-118,
  1996. piecewise_gate: the gate at 0 becomes `x >= 0`; the gate at 1 becomes `x > 1`.
- arithmetic: arithmetic operator replacement (AOR), from the same set. piecewise_ops: in the negative branch, the
  subtraction becomes an addition; the multiplication becomes a division.
- sign: an absolute value inserted and a unary minus dropped, the kin of the ABS and UOI operators of the mutation
  testing literature (DeMillo, Lipton, and Sayward, "Hints on Test Data Selection: Help for the Practicing
  Programmer", IEEE Computer 11(4):34-41, 1978, began it). signed_root: sqrt(fabs(x)) for every x; sqrt(-x) for x < 0.
- precision: an accumulator narrower than the reference's, the kind of floating-point variation randomized
  differential testing of numerical programs looks for (Laguna, "Varity: Quantifying Floating-point Variations in
  HPC Systems Through Randomized Testing", IPDPS 2020, 622-633; McKeeman, "Differential Testing for Software",
  Digital Technical Journal 10(1):100-107, 1998). prefix_sum: an f32 running sum; f32 blocks of 16.
- narrowing: an integer product formed in a narrower type. scale_i64: 32-bit wraparound; an f32 product.
- special-value: a NaN not propagated. pair_max: `a > b ? a : b`; `std::fmax(a, b)`.

The bibliographic details above were checked on 2026-10-20 against secondary records only, not the papers' full
texts, as plans/spikes/p16-seeded-classes.md logs (its searches, what each returned, and the sources); that ROR and
AOR are among Offutt et al.'s sufficient operators is as citing papers report it. The narrowing and special-value
classes cite no source. The classes are this task's choice.

## Witnesses and the freeze

Each witness was written from its candidate's trigger condition, never from the fuzzer's strategies or proposals
(the Freeze decision). Nothing here has been built or run yet: task P16.10 builds and runs every reference,
candidate, and witness natively in the sandbox, checks that each candidate passes binary_io on its item's held-out
inputs and diverges on its witness, and then writes the freeze record that fixes these files' sha256.
