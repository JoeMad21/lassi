# Tier A TT references on Wormhole silicon (2026-10-23)

[MEASURED] Each of the five tt-pairs-v0 TT reference programs ran once on one Tenstorrent Wormhole card. Each one's outputs passed the binary_io oracle against its C++ counterpart under the item's declared tolerance. This is a smoke check of the pinned stack on silicon: one run per item and one card, so it is not a rate and not a timing.

- Host: alpha01, Ubuntu 22.04.5 LTS, kernel 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+.
- Device: Tenstorrent Wormhole n300 card 0, serial 01000146118320F8. Firmware bundle 19.5.0.0, ARC 2.37.0.0, ETH 7.5.0, read from sysfs. tenstorrent driver 2.8.0 (UMD logged "KMD version: 2.8.0").
- Commit: 982e71b202cc733cdea966e5ab5912cf037106ed, clean in the run's slot. Each item's provenance.json gives `dirty: false`, and each record.json gives the same.
- Pins: tt-metal 5280a9cfb00998fd49667a29523d03aee905c129 (toolchains/tt-metal.pin), gcc 12.3.0 (toolchains/gcc.pin). ttsim was not used on the silicon side.
- Date: 2026-10-23, 04:19:10 to 04:20:41 UTC (00:19 to 00:20 EDT), in the owner's window of 00:00 to 06:00 EDT.
- Approval: the owner enabled the gate's tt_silicon class and told the agent to go, both in the working session (Agent Rule 9).

## What ran

`uv run python plans/runs/tt-silicon/smoke.py <item> --device 0`, one rx run per item, in turn. The script follows tests/bench/test_tt_pairs_remote.py:

- It fetches and verifies the suite's pinned files.
- It builds the TT reference with ttmetal-host and the C++ counterpart with gcc-native.
- It stages the item's held-out inputs and runs the C++ side through the native executor, in the sandbox.
- It runs the TT side once on card 0 and compares every output with binary_io.

How the TT side ran on the card:

- The environment was the ttsim row's, without TT_METAL_SIMULATOR and with TT_VISIBLE_DEVICES=0.
- UMD logged "Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {1}" in every run: card 0's PCIe chip and that n300's second chip, which it reaches over ethernet, and no other card.
- The program ran outside the sandbox, because the sandbox gives a program no device node. Every program here is the item's tracked upstream reference, never model output (Agent Rule 6).
- No card was reset, flashed, or reconfigured, and nothing was left holding /dev/tenstorrent/0 after the runs (`fuser`).

Before the silicon runs, the same script passed loopback on ttsim from the same commit (rx 20261022-211744-tt-silicon-f12e).

## Results

| Item | rx run | Exit, hang | Output | Metric | Value | Threshold | Passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loopback | 20261022-211857-tt-silicon-869d | 0, no | output | max_abs | 0.0 (pcc 1.0) | 0 | yes |
| eltwise_binary | 20261022-211920-tt-silicon-72a5 | 0, no | c | max_abs | 0.002197265625 (pcc 0.9999989) | 0.01 | yes |
| eltwise_sfpu | 20261022-211940-tt-silicon-49a1 | 0, no | result | max_abs | 0.015625 (pcc 0.9999595) | 0.05 | yes |
| matmul_single_core | 20261022-212000-tt-silicon-96c5 | 0, no | c | pcc | 0.9798845 (max_abs 6.0) | 0.97 | yes |
| matmul_multi_core | 20261022-212024-tt-silicon-daa2 | 0, no | c | pcc | 0.9999084 (max_abs 0.28125) | 0.97 | yes |

Each item's directory holds:

- `provenance.json`: the gate's run record, from `rx pull <id> --into`.
- `record.json`: the script's record, fetched with `rx exec cat`. Its sha256 matched the host file's sha256sum. `rx pull --path` on a single file copies nothing.

The wall times in record.json (about 1.8 to 1.9 s per program, as the script's timer reads them) are not performance.

## Notes

- matmul_single_core passes with pcc 0.9799 against a threshold of 0.97, the narrowest margin of the five. Its max_abs of 6.0 is large because bf16 accumulation over a whole matmul differs from the C++ float result. The item judges by pcc for that reason.
- tt-metal wrote two files into the pinned tree:
  - `generated/fabric/physical_chip_mesh_coordinate_mapping_1_of_1.yaml`
  - `generated/fabric/asic_to_fabric_node_mapping_rank_1_of_1.yaml`

  Under the sandbox the tree is read-only, and these writes only warn. lassi-install.txt and the CPM source list, which the install check reads, are unchanged.
- The program is not model output. Running a model's TT candidates on silicon needs a sandboxed silicon executor, which exposes exactly one card's device node and the hugepages to the run. It is not built yet (Execution Backends, the tt_silicon row).
