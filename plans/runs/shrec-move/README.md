# The move to /mnt/nvme10/shrec-shared (task P17.63)

The owner's direction of 2026-11-11 and the choices made in that session are OQ-124 (plans/OWNER-QUEUE.md).

- shrec-move.sh: the move script exactly as it ran on alpha01 (sent to `rx exec` as base64; it renames on one
  filesystem, leaves a link at each old path, repairs the worktree slots, and deletes nothing). Two of its lines run
  past 120 characters; it is kept byte for byte as run.
- list-rx-20261110-215100-exec-7c8d.txt: its list-only pass (32 entries to move, 0 absolute links in the HF stores).
- move-rx-20261110-215116-exec-abc4.txt: the move itself (32 moved, 5 links in the new layout, 40 slots repaired).
  Both outputs are the gate's job logs, read back with `rx job tail`.
- move-test.sh: a test of the script on a fake tree, no host contacted; move-test-output.txt is its output from
  the workstation's Ubuntu-24.04 WSL distro on 2026-11-11 (the earlier copy of this test that preceded the move
  checked the same cases with a scratch tree inside this folder).

Read-only inventories before the move: rx 20261110-212139-exec-b4e4, 20261110-212242-exec-838b, and
20261110-212520-exec-fb83 (sizes, home folder, HF stores, venvs, files naming the old root per toolchain).
