# Adversary replay recordings (SYNTHETIC, task P16.4)

Every file here is a scripted, synthetic recording in the replay backend's
format (lassi/llm/replay.py), labeled `"synthetic": true`, for the llm
adversary (lassi.agents.llm) bound with `backend: replay` and
`adversary.recording`. The replies were written by hand for the tests. They
are not model output: no model is called before OQ-068 is answered. Nothing
here is upstream source or upstream prompt text (OQ-018), and no value here
is a measurement.

Each recording holds reply text only, never `messages`: the messages the
adversary sends are its rendered prompts, and none is stored here (the P16
plan's Constraints, Replay recordings line; a test over this tree checks it).

The target contract is adv-stage's input x (f32, shape [4], each element in
[-1000, 1000), no NaN, no infinity; tests/core/adversary_support.py).

## Files

- `in-contract.json`: one reply proposing two in-contract sets (x filled
  with 500; x with the values 1, 2, 3, 4), with SYNTHETIC prose outside its
  block that claims a divergence (the adversary ignores it and nothing
  records it), then `{"proposals": []}`, which stops the search.
- `break-contract.json`: one reply whose eight proposals each break the
  contract, in this order: x as i32 (dtype), x of shape [3] (shape), x
  filled with 5000 (out-of-range), x filled with "nan" (nan), x plus an
  extra input w (extra-input), no inputs (missing-input), x of shape
  [2000000] filled with 0 (too-many-elements), and x as dtype "f128"
  (unreadable); then `{"proposals": []}`. Every one of them is refused by
  check_inputs before any run, under the reason named; the phase gate runs
  it too.
- `malformed.json`: six replies that each break the reply form (no block;
  two blocks; JSON that does not parse; an unknown top-level key; an i8
  value of 300; a float of 1e39 for f32), then `{"proposals": []}`.

All files are plain ASCII with LF line ends. They were written by a one-off
script from literals; edit them by hand, and keep each file's description
here true.
