# Wormhole cards, host memory, and the IOMMU on alpha01

This note (2026-10-26) explains one host-security problem that stands between LASSI and running model-generated
programs on the Tenstorrent cards, what fixes it, what the fix costs, and what has to be decided. Evidence and source citations are in plans/spikes/tt-card-dma.md and
plans/spikes/tt-silicon-executor.md; the decision is OQ-078 in plans/OWNER-QUEUE.md.

## In one paragraph

On alpha01 the one Wormhole card whose IOMMU group was read, card 0 (PCI 0000:e8:00.0, group 135), runs with the
IOMMU in pass-through mode: probe rx 20261022-214312-exec-59e3 read its group type as "identity", the type that
`iommu=pt` on the kernel command line gives every group. The other cards' groups and the kernel command line were
not read. In that mode a card can be told to read or write host physical addresses untranslated (the low 4 GiB with
certainty, higher addresses plausibly; inferred from the register semantics, never tried). Any program that holds a
card's device node can tell it to: it maps the card's BAR2 register window, which every tt-metal program must do
anyway, and writes a host address into the card's PCIe DMA engine or its outbound address-translation (iATU)
registers. Those are ordinary memory stores, not system calls, so the LASSI sandbox's system-call filter cannot see
them, and the driver has no option that stops them. A model-generated program running on a card could therefore
overwrite or read alpha01's memory: other users' processes, the rx gate, the kernel. Switching each card's IOMMU
group to translation (DMA) mode is expected to close this: the host IOMMU then checks every transaction the card
makes and refuses any address the driver did not map for it (inferred: the standard meaning of DMA remapping,
applied here; not run on alpha01). That is a root-level host change, which agents never make.

## Why this matters now

- The owner's direction of 2026-10-26: "The silicon must be in the correction loop in future tests." Today every
  TT program LASSI generates runs on ttsim, a simulator. In the correction loop a model's candidate is built, run,
  judged, and corrected up to 8 times per trial; the night of 2026-10-25 ran 131 trials (1073 attempts) unattended
  (results/night-20261025/night/trials.tsv). Putting the cards in that loop means every candidate that builds
  running on silicon, all night, with nobody watching.
- Those programs are not trusted code. Most are wrong in ordinary ways: on the night of 2026-10-25, 52 of
  Llama-3.3-70B's 60 C++ -> TT trials ended at the correction cap with a build error
  (results/night-20261025/summary.md; ttsim, development results). A program does not have to be hostile to cause
  harm here: a host program that passes a wrong buffer address, or a kernel that programs a DMA descriptor from an
  uninitialized value, is enough. On a translating IOMMU such a mistake would fault instead of reaching memory
  (inferred: the standard meaning of DMA remapping, applied here; not run on alpha01). Under pass-through it would
  land on whatever host page that number names.
- Simulator results have limits the hardware does not: ttsim cannot run some constructs (simulator gaps), reports no
  power, and its timing is not performance (Agent Rule 2). Silicon in the loop gives real pass/fail on the hardware
  the project targets, and, with the Tenstorrent power backend (task P9.5), chip power for LASSI-EE.

## Background: what the IOMMU does

A PCIe device such as a Wormhole card can read and write host memory directly (DMA), without the CPU. The IOMMU sits
between devices and memory and translates the addresses devices use (IO virtual addresses) into physical addresses,
using page tables the kernel keeps per device group. The kernel can run each group in one of two kinds of domain:

- Identity (pass-through). Device addresses are host physical addresses, unchecked. Fast, and common on hosts that
  trust their devices; `iommu=pt` on the kernel command line makes it the default for every group.
- DMA (translation). The kernel maps, per group, only the pages the driver has given the device, and the IOMMU
  translates every transaction from the group's devices. A device access to any other address is meant to fault and
  be reported in the kernel log instead of reaching memory (the standard meaning of DMA remapping; for these cards
  inferred, not run on alpha01). "DMA" translates every transaction; "DMA-FQ" does too but removes translations
  lazily with batched invalidation, for less overhead at the cost of reduced memory protection; identity is no
  translation, "zero protection" (Linux v6.6 Documentation/ABI/testing/sysfs-kernel-iommu_groups).

On alpha01, probe rx 20261022-214312-exec-59e3 read card 0's group type (0000:e8:00.0, group 135 per probe rx
20261022-213441-exec-32e2) as identity (plans/spikes/tt-silicon-executor.md, Commands and outputs 4 and 8, and
Finding 2). The other cards' groups and /proc/cmdline were not read, so whether the command line carries `iommu=pt`
is not known.

## What a program holding a card can do under identity (read from the pinned sources)

At the project's pins (tt-kmd 2.8.0, UMD 0450f1be, tt-metal 5280a9cf):

1. The driver gives any holder of /dev/tenstorrent/<n> a mapping of the card's BAR0 and BAR2 register windows
   (the QUERY_MAPPINGS ioctl and mmap). No capability check guards this; tt-metal needs it for all device access, so
   no executor can withhold it.
2. Through BAR2 the program can write the card's embedded PCIe DMA engine registers (source, destination, size,
   doorbell) or one of its 16 outbound iATU regions. For the DMA engine, UMD itself does this for its own
   transfers, with the destination a caller-chosen 64-bit host address and no software bound. The iATU is
   different: on this host the driver programs it (tt-kmd 2.8.0 maps buffers to the NOC itself), and UMD's own iATU
   path goes through the ARC mailbox, not the registers; the registers in BAR2 are still directly writable
   (plans/spikes/tt-card-dma.md, Findings 1 to 3).
3. Under an identity domain that address is a host physical address, and the card would read or write it
   untranslated (inferred from the register semantics, never tried; see Read versus inferred below). At least the
   low 4 GiB of host physical memory is reachable with certainty; higher addresses plausibly.
4. These are CPU store instructions to a mapped page, not system calls. The LASSI sandbox's seccomp filter, which
   refuses the driver's reset ioctl (OQ-075), cannot intercept them, and no tt-kmd 2.8.0 module parameter disables
   them.
5. With every card exposed to a run (the all-card exposure of OQ-075), a program can also map another card's BAR
   into a card's address space (the peer-BAR ioctl): card-to-card reach, which a translating IOMMU would not remove
   (inferred: the ioctl has the driver map the peer's BAR into the card's DMA address space with dma_map_resource,
   so translation carries that mapping; plans/spikes/tt-card-dma.md, Finding 4).

The driver's own pinned-buffer path is safe: when tt-metal asks the driver to map a buffer for the card, the driver
programs the window with the kernel-chosen address of the caller's pinned pages. The exposure is that nothing stops a
program from going around that path and writing the registers itself.

Read versus inferred: the register paths, the BAR mapping, the absence of checks, and card 0's identity domain are
read from the pinned sources and the probe. That a hand-programmed register actually steers the
hardware is inferred from the register semantics (UMD and the driver program the same registers for normal
transfers); it was not tried, and must not be. That a translating domain blocks such an address is inferred as
well (the standard meaning of DMA remapping, applied here; not run on alpha01).

## The fix: put the cards' IOMMU groups in DMA mode

With a translating domain, every transaction a card issues, from the DMA engine or through any iATU window, is
translated by the host IOMMU against the mappings the driver created for that card. A hand-written address that
the driver never mapped should fault instead of reaching memory. Because this check is on the host side, downstream
of the card, it should hold however the card's internal registers are programmed. This protection is inferred: it is
the standard meaning of DMA remapping, applied to these cards, and was not run on alpha01
(plans/spikes/tt-card-dma.md, Finding 5).

The software supports it: tt-kmd 2.8.0 is past the versions that added IOMMU support and NOC buffer mapping, UMD
detects an IOMMU when the group type starts with "DMA", and it then takes its IOMMU path (anonymous memory mapped
through the driver, no hugepages) automatically.

Two ways to apply it, both root-only (agents run neither):

- Per card, with no reboot. For each card: find its PCI address and IOMMU group
  (`readlink /sys/bus/pci/devices/<bdf>/iommu_group`), make sure nothing is using the card, unbind the tenstorrent
  driver from it (`echo <bdf> > /sys/bus/pci/drivers/tenstorrent/unbind`; every device in the group must be
  unbound), write the type (`echo DMA > /sys/kernel/iommu_groups/<group>/type`), and rebind
  (`echo <bdf> > /sys/bus/pci/drivers/tenstorrent/bind`). The kernel's documented rule (sysfs-kernel-iommu_groups,
  Linux 6.6) is that the type can change only while no driver is bound. This lasts until reboot, so it needs a boot
  script to persist. It is meant to change only the cards, not the RNGD cards, network cards, or anything else, but
  each card's IOMMU group members were not read (only card 0's group link was), so root lists
  /sys/kernel/iommu_groups/<group>/devices first: every member of a group changes with it.
- Host-wide, with a reboot. If the kernel command line carries `iommu=pt` (check /proc/cmdline; it was not read),
  remove it, so every group, the RNGD cards and network adapters included, defaults to a translating domain.
  Simpler and permanent, but it changes every device on a shared host, and its effect on the FuriosaAI cards and on
  network throughput is unmeasured.

The per-card route is recommended: it changes only the cards' groups (with any other members they hold, which were
not read).

After the change, check: the group's type file reads DMA; the kernel log shows no IOMMU faults during a run of the
tracked reference programs; and UMD reports the IOMMU path. A human-started smoke of the tracked references on one
card (as on 2026-10-23) then shows whether tt-metal at the pin runs end to end under translation, which is not yet
verified.

## What the fix costs and does not cover

- Performance: translation adds IOTLB lookups and mapping work on the card's host-memory traffic. For LASSI's small
  test programs this should not matter; it is unmeasured.
- tt-metal under the IOMMU path at the pin is untested end to end (UMD supports it). The first smoke run answers it.
- It would not remove card-to-card reach through the peer-BAR ioctl (inferred, as in item 5 above). Exposing only
  the card a run is placed on, not all cards, would; that reopens OQ-075's all-card choice.
- It does not address Agent Rule 9's other concern: looped device opens have twice taken alpha01's network down.
  Silicon in the correction loop needs pacing (one placement at a time, one open per program run, a pause between
  runs) and a stop on the first card fault.

## What has to be decided

1. OQ-078: apply the DMA-mode change (recommended: per card), keep model output off silicon until then, or accept
   the exposure (not recommended). Who applies it, and when.
2. OQ-099: the owner's direction that silicon be in the correction loop changes Agent Rule 9, which today says
   silicon runs are evaluation runs a human starts and RL rewards never execute on silicon. The proposed rule text,
   its prerequisites (OQ-078 applied and verified, the gate's tt_silicon class enabled, pacing), and what stays
   human-started are in that item.
3. Whether a run exposes all cards (OQ-075) or only its placed card, which decides the card-to-card exposure.
