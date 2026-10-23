# Fixture programs for the adversary stage's remote tests (SYNTHETIC)

Every file here is a synthetic fixture of task P16.10 (plans/p16-adversarial.md, the third acceptance bullet: remote
tests of the stage on fixture programs outside adv-seeded-v0). Each program was written by hand; none is model
output or upstream code, and no value here is a measurement. No suite under assets/bench lists them: the tests in
tests/core/test_adversary_stage_remote.py write their own one-item manifest at run time and read this directory as
its bench root, so no held-out input of an eval or unassigned item comes from here (Agent Rule 5). This directory
lies outside every directory the freeze record (tests/fixtures/adversary/freeze.yaml) covers.

## Files

- `scale/cpp/scale.cpp`: the item's cpp target reference, y = 2x for an f32 x, built with gcc-native against
  assets/harness/c/lassi_io.h. The tests also hand it back as its own candidate, the control.
- `candidates/scale/large-zero.cpp`: the reference with one fault, y = 0 where |x| >= 100. The held-out inputs lie
  in [0, 10), where it agrees with the reference; the item's contract lets x reach 1000 in magnitude, where it
  diverges.
