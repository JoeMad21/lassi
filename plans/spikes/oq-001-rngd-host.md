# Spike OQ-001: the Furiosa RNGD cards on alpha01

- Owner-queue item: OQ-001 in `plans/OWNER-QUEUE.md`, answered by the owner on 2026-09-23.
- Date: 2026-09-23. Times are alpha01 host times (UTC-07:00) unless they end in Z.
- Host: alpha01 (I/ONX), reached only through `uv run tools/rx.py`.
- No repository code took part: `rx devcheck`, `rx doctor`, and `rx exec` run from the gate's scratch root without a checkout.

## Question

OQ-001 asked which I/ONX host holds the RNGD cards, and whether they can return to alpha01 or be reached from it. The bible's Environment State (2026-09-22 check) recorded `lspci -d 1ed2:` as empty and `furiosa-smi info` as listing zero devices.

The owner answered: "Alpha01 can directly interface with the Furiosa cards. You can run furiosa-smi on the host and get status."

This spike checks that answer with read-only commands.

## Classification

Factual once answered. Read-only status commands on alpha01 settle it. No card was claimed, no model was served, and no project code opened a device.

## Method

- `rx devcheck`: the gate's read-only inventory (`DEVCHECK` in `tools/server/gate.py`). It runs `lspci -d 1e52:`, `lspci -d 1ed2:`, `ls -l /dev/tenstorrent*`, `furiosa-smi info`, `id -nG`, `ls -l /dev/kfd` plus a Python `open()` of it, and `nvidia-smi -L` when that command exists.
- `rx exec` of `furiosa-smi ps`, the check Agent Rule 8 requires before any claim.
- `rx doctor` for the gate's device classes (`devices_enabled`).
- `rx job status <id>` and `rx job tail <id>` to read back the exec records.

## Commands and Outputs

### 1. rx devcheck

Command: `uv run tools/rx.py devcheck`. It returned checked_at 2026-09-23T11:55:29-07:00 and host alpha01. The inventory has no rx run id.

Output as recorded when the check ran:

- The `furiosa_pci` and `furiosa_smi_info` probes are shown with two edits: the row separators between devices are omitted, and the degree sign that `furiosa-smi` prints is written as " C" to keep this file ASCII.
- The last four probes are summaries, not verbatim output.

```
furiosa_pci (lspci -d 1ed2:), rc 0:
8b:00.0 Processing accelerators: FuriosaAI, Inc. RNGD (rev 01)
8c:00.0 Processing accelerators: FuriosaAI, Inc. RNGD (rev 01)
8d:00.0 Processing accelerators: FuriosaAI, Inc. RNGD (rev 01)
8e:00.0 Processing accelerators: FuriosaAI, Inc. RNGD (rev 01)
94:00.0 Processing accelerators: FuriosaAI, Inc. RNGD (rev 01)
95:00.0 Processing accelerators: FuriosaAI, Inc. RNGD (rev 01)
96:00.0 Processing accelerators: FuriosaAI, Inc. RNGD (rev 01)
97:00.0 Processing accelerators: FuriosaAI, Inc. RNGD (rev 01)

furiosa_smi_info (furiosa-smi info), rc 0:
+------+--------+-------------------+---------+---------+--------------+
| Arch | Device | Firmware          | Temp.   | Power   | PCI-BDF      |
+------+--------+-------------------+---------+---------+--------------+
| rngd | npu0   | 2026.3.0, 2d3f72a | 28.65 C | 37.00 W | 0000:8b:00.0 |
| rngd | npu1   | 2026.3.0, 2d3f72a | 28.27 C | 37.00 W | 0000:8c:00.0 |
| rngd | npu2   | 2026.3.0, 2d3f72a | 30.97 C | 37.00 W | 0000:8d:00.0 |
| rngd | npu3   | 2026.3.0, 2d3f72a | 28.97 C | 37.00 W | 0000:8e:00.0 |
| rngd | npu4   | 2026.3.0, 2d3f72a | 29.70 C | 37.00 W | 0000:94:00.0 |
| rngd | npu5   | 2026.3.0, 2d3f72a | 28.67 C | 38.00 W | 0000:95:00.0 |
| rngd | npu6   | 2026.3.0, 2d3f72a | 29.11 C | 38.00 W | 0000:96:00.0 |
| rngd | npu7   | 2026.3.0, 2d3f72a | 27.52 C | 37.00 W | 0000:97:00.0 |
+------+--------+-------------------+---------+---------+--------------+

tenstorrent_pci: rc 0, empty. tenstorrent_dev: no /dev/tenstorrent*.
kfd: /dev/kfd is crw-rw---- root render; open() gives PermissionError (groups: joseph_ufl).
nvidia: no nvidia-smi.
```

### 2. furiosa-smi ps

Command, as `rx job status 20260923-115543-exec-6dba` records it: `date -Is; furiosa-smi ps 2>&1 | head -40`, run through `rx exec`. rx id 20260923-115543-exec-6dba, rc 0.

Output, verbatim from `uv run tools/rx.py job tail 20260923-115543-exec-6dba`:

```
2026-09-23T11:55:43-07:00
+-----+----------+-----+
| PID | NPU core | CMD |
+-----+----------+-----+
[rx] id=20260923-115543-exec-6dba state=done rc=0
```

The table has its header and no rows, so at 11:55:43 no process held an NPU core.

### 3. rx doctor: gate device classes

The state of the gate's `rngd` class changed during 2026-09-23:

- Earlier on 2026-09-23, the `rx doctor` run that went with the devcheck reported `devices_enabled` as rngd false, tt_silicon false, rocm_gpu false, and nvidia_gpu false. Its exact time was not recorded. The P0.6 spike recorded the same four values at 04:13:06-07:00 (`plans/spikes/p0-nvcc.md`, step 1).
- Two later `rx doctor` runs reported rngd true:
  - one at 2026-09-23T19:06:12Z (12:06:12-07:00), taken from the local clock just before the call, because `rx doctor` prints no host date;
  - one just before rx 20260923-120658-exec-b063, whose `date -Is` printed 2026-09-23T12:06:58-07:00.

Excerpt from both later runs:

```
  "devices_enabled": {
    "rngd": true,
    "tt_silicon": false,
    "rocm_gpu": false,
    "nvidia_gpu": false
  },
```

The rngd class went from false to true between the earlier doctor run and 12:06:12-07:00. J reported on 2026-09-23 that he enabled it in the gate config on alpha01; this spike did not change it, and agents never change device classes. Later read-only `rx doctor` runs, at about 2026-09-23T19:14Z and at 2026-09-23T19:20:17Z (local clock), still showed rngd true and tt_silicon, rocm_gpu, and nvidia_gpu false.

## Sources

- `tools/server/gate.py`, `DEVCHECK` and `verb_devcheck`: the probe commands listed under Method.
- `plans/OWNER-QUEUE.md`, OQ-001: the question and the owner's answer.
- `docs/BIBLE.md`, Environment State and Host Facts: the 2026-09-22 check and the last known RNGD configuration.

## What It Shows

1. The eight RNGD cards are on alpha01 [MEASURED 2026-09-23, rx 20260923-123527-exec-4e1b for the count and firmware; the temperature and power readings below come from the unlogged rx devcheck]:
   - `lspci -d 1ed2:` lists 8 FuriosaAI RNGD (rev 01) devices, at 8b:00.0 to 8e:00.0 and 94:00.0 to 97:00.0.
   - `furiosa-smi info` on the host lists them as npu0 to npu7, all with firmware 2026.3.0 (2d3f72a). Temperatures range from 27.52 C to 30.97 C and power from 37 W to 38 W.
   - This matches the owner's answer and the bible's last known configuration (8 cards npu0-npu7, firmware 2026.3.0).
2. `furiosa-smi` runs on the host and reports status. Both `info` and `ps` returned rc 0.
3. At 11:55:43 no process held an NPU core, npu0 included (rx 20260923-115543-exec-6dba).
4. The same devcheck matches the bible's other Environment State rows:
   - no Tenstorrent PCI device and no /dev/tenstorrent*;
   - /dev/kfd refuses open() for joseph_ufl, whose only group is joseph_ufl;
   - no nvidia-smi.

## Logged Re-Check

`rx devcheck` logs no run id, so the same inventory was repeated through a logged run: rx 20260923-123527-exec-4e1b (2026-09-23T12:35:27-07:00, `date -Is; lspci -d 1ed2: ; furiosa-smi info; furiosa-smi ps`, rc 0). It again listed 8 RNGD devices at 8b-8e and 94-97, npu0-npu7 at firmware 2026.3.0, 2d3f72a, and an empty `furiosa-smi ps` table. The bible cites this id for the card count and firmware [MEASURED].

## What It Does Not Show

- That the project can use the cards. No card was claimed, no model was served, and nothing ran on an NPU. Serving and target execution follow Agent Rules 8 and 13 and need the gate's rngd class to be enabled. A class can change at any time, so check `rx doctor` before any RNGD work.
- Which gate state held at the moment of the devcheck. Section 3 gives only a doctor run of unrecorded time (rngd false) and later runs (rngd true from 12:06 -07:00).
- Software versions:
  - The Furiosa SDK and furiosa-llm versions. The venv `/mnt/nvme10/joseph_ufl/furiosa-venv` was not inspected.
  - The driver version and the `furiosa-smi` version, which were not queried.
- Why the cards did not enumerate in the 2026-09-22 check.
- That npu0 is free. An empty `ps` at one instant does not change its tenancy, and Agent Rule 8 still forbids claiming it.

## Consequences

- The bible records the result under Environment State (the RNGD row and Host Facts), Execution Backends (`furiosa_silicon`), Build Roadmap (the P3 and P14 Blocked-by cells), and Risks And Questions (the RNGD risk and question 1), with a Decision Log entry dated 2026-09-23.
- Agent Rule 8 is unchanged.
- The phases that need RNGD (P3, P6, P7, P14) keep the state `plans/STATUS.md` gives them until the owner clears the blocker in the phase Note (AGENTS.md, Work Order).
