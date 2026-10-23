# Spike: can a sandboxed program make a Wormhole card DMA to arbitrary host memory?

- Owner item: drafted below as OQ-078 (the remediation is a choice). Bible: Sandbox (the Tenstorrent-access
  bullet), Execution Backends (the tt_silicon row), Agent Rules 6, 7, and 9. Related: OQ-072, OQ-075, task P17.15,
  plans/spikes/tt-silicon-executor.md.
- Status: the pinned sources are read from GitHub. Nothing ran on alpha01 and no device command ran anywhere; the
  only host facts used are those plans/spikes/tt-silicon-executor.md already measured and cited (the card's IOMMU
  domain type "identity" from probe rx 20261022-214312-exec-59e3, and the driver version 2.8.0). No rx command of
  any kind ran.
- Date: 2026-10-23.
- Pins: UMD 0450f1be3a21cbf17adb97517038586a6a3af4c0 (the submodule of tt-metal 5280a9cf, confirmed by
  plans/spikes/tt-silicon-executor.md probe 73fc); tt-metal 5280a9cfb00998fd49667a29523d03aee905c129; tt-kmd tag
  ttkmd-2.8.0.
- Citations:
  - `umd <file>:<line>`: tenstorrent/tt-umd at 0450f1be, paths under device/.
  - `kmd <file>:<line>`: tenstorrent/tt-kmd at tag ttkmd-2.8.0.
  - `metal <file>:<line>`: tenstorrent/tt-metal at 5280a9cf.
  - `sandbox.py`, repository files: working tree of branch p16-adversarial (the Tenstorrent paths of
    lassi/executors/sandbox.py are the P17.15 code).
- Conventions: upstream code is cited by file and line and paraphrased, never quoted (as the sibling spike does).
  Register offsets are named by their source symbol, not reproduced as a write sequence.

## Question

On alpha01, can a user-space program that holds a Wormhole n300 card's /dev/tenstorrent node make the card DMA to
arbitrary host physical memory? The card's IOMMU group type is "identity", from the kernel cmdline iommu=pt
(plans/spikes/tt-silicon-executor.md, Finding 2). If it can, a sandboxed model-generated program (Agent Rule 6)
could read or write all of host memory, and the sandbox (lassi/executors/sandbox.py, TenstorrentExposure) would not
isolate the host.

Why it matters:

- OQ-075 is answered: the sandbox exposes every Wormhole card to a silicon run, with a seccomp filter that refuses
  the driver's reset ioctl as the only device-level mitigation (OQ-075 part 4a). This spike tests whether that
  mitigation, and the sandbox as designed in P17.15, actually confine a run to the card or whether model code can
  reach host memory through the card.
- The tt_silicon executor (task P17.16) runs model-generated TT programs on silicon. If the hole is open, every such
  run can read or write the host's RAM, including other tenants' and the gate's.

## Classification

- The capability question (can model code reach arbitrary host memory at these pins and this host configuration) is
  factual: it is answered by reading the pinned sources and the already-measured host facts. Findings 1 to 5 below.
- The remediation is a choice: it needs a host-level change only the owner or root can make (set the card's IOMMU
  group to a DMA domain, or drop iommu=pt), or the owner's acceptance of the risk. Drafted as OQ-078.

## Commands and outputs

No remote or device commands ran. The pinned trees were fetched to a scratch directory and read locally:

```
curl -sSL https://codeload.github.com/tenstorrent/tt-kmd/tar.gz/refs/tags/ttkmd-2.8.0
curl -sSL https://codeload.github.com/tenstorrent/tt-umd/tar.gz/0450f1be3a21cbf17adb97517038586a6a3af4c0
```

tt-metal 5280a9cf was read through its GitHub pages and through the facts plans/spikes/tt-silicon-executor.md already
established for it (that tt-metal calls no reset or warm reset, probe c1f7, and uses UMD's sysmem path with two 1 GiB
channels for an n300). The kernel ABI for the IOMMU group type was read from the Linux v6.6 documentation (Sources).

## Findings

### 1. How the card's PCIe outbound window (iATU) to host memory is configured

The Wormhole PCIe outbound iATU (the unit that remaps a NOC-initiated transaction to a host PCIe address) is
configured through two different code paths at these pins, and the host-side target address is validated in only one
of them:

- KMD path (the one taken on this host). UMD calls `WormholeTTDevice::configure_iatu_region` only when the KMD is
  too old to map a buffer to the NOC itself: `LocalChip::start_device` runs `init_pcie_iatus()` only inside
  `if (!is_mapping_buffer_to_noc_supported())` (umd chip/local_chip.cpp:190-193), and
  `is_mapping_buffer_to_noc_supported()` is `kmd_version >= KMD_MAP_TO_NOC` where `KMD_MAP_TO_NOC` is version 2.0.0
  (umd pcie/pci_device.cpp:569; umd api/umd/device/utils/kmd_versions.hpp:23). The host driver is 2.8.0, so this is
  true and UMD does not program the iATU for sysmem; instead `map_hugepage_to_noc` issues `TENSTORRENT_IOCTL_PIN_PAGES`
  with the NOC_DMA flag (umd pcie/pci_device.cpp:617-667) and the KMD programs the window. In the KMD, that reaches
  `wormhole_configure_outbound_atu` (kmd wormhole.c:1082-1108) by way of `setup_noc_dma` and `configure_outbound_iatu`
  (kmd memory.c:158-187, :123-156). There the target is the DMA/physical address the kernel just assigned to the
  pinned pages (kmd memory.c:649, :669, :182), not a user-chosen value; the only bound checked is `limit > U32_MAX`
  (kmd wormhole.c:1096-1097). So through the KMD ioctl the target is confined to the caller's own pinned pages.

- User-space ARC path. `WormholeTTDevice::configure_iatu_region` takes a `uint64_t target`, splits it into a low and a
  high 32-bit word (umd tt_device/wormhole_tt_device.cpp:100-102), writes those plus the region id and size into the
  ARC CSM mailbox through `bar_write32` (which targets BAR0, umd tt_device/tt_device.cpp:272-279), and sends the ARC
  message SETUP_IATU_FOR_PEER_TO_PEER (message type 0x97; umd tt_device/wormhole_tt_device.cpp:115-120, umd
  api/umd/device/arch/wormhole_implementation.hpp:104, :371-372). UMD performs no validation of `target`. The base
  `TTDevice::configure_iatu_region` throws "not implemented" (umd tt_device/tt_device.cpp:236-237); the header comment
  states plainly that programming the iATU from user space is "architecturally incorrect" because "Multiple processes
  can corrupt each other's iATU configurations" (umd api/umd/device/tt_device/tt_device.hpp:223-226). Whether the ARC
  firmware validates the target is not in these sources (firmware is not public here); it is moot for the security
  question, because the iATU registers are also directly writable (Finding 3) and the embedded DMA engine (below)
  needs no iATU at all.

- Is the target any 64-bit value chosen in user space? Through the KMD ioctl, no (kernel-chosen pinned-page address).
  Through either the ARC path or a direct register write (Finding 3), yes: the register pair is a full 64-bit host
  address with no software bound. Under an identity IOMMU domain that host address is a host physical address
  (Finding 2).

Read vs inferred: the code paths and the absence of a target check in the user-space path are read. That the ARC
firmware does or does not validate the target is unknown (not read).

Confidence: high that the two paths exist as described and that the KMD path confines the target while the user-space
path does not.

### 2. What the KMD exposes for DMA, and what the device sees under an identity domain

- The pin-pages ioctl `TENSTORRENT_IOCTL_PIN_PAGES` pins the caller's pages for the lifetime of the pinning
  (`pin_user_pages_fast_longterm`, kmd memory.c:596) and returns a host address. Which address depends on the IOMMU
  domain: `is_iommu_translated` returns true only when a domain exists and its type is not IOMMU_DOMAIN_IDENTITY (kmd
  memory.c:511-515). On alpha01 the domain type is "identity" (plans/spikes/tt-silicon-executor.md, Finding 2), so the
  function returns false and the KMD takes the non-IOMMU branch (kmd memory.c:658-677): it requires the pages to be
  physically contiguous and returns `page_to_phys(pages[0])` (kmd memory.c:669). So the address the device is given is
  a raw host physical address, and the device issues PCIe transactions to physical addresses with no translation in
  between.

- Can the NOC-to-PCIe path reach addresses outside the pinned pages when the iATU is reprogrammed? Yes. Pinning only
  decides what physical address the KMD writes into an iATU region when the KMD programs it. It does not constrain the
  hardware: the outbound iATU register pair and the embedded DMA engine (Finding 3) accept any address. Once a region
  (or the DMA engine's address register) is reprogrammed from user space to a different host physical address, the
  card's transactions go there. The pinned pages are not a hardware fence; they are just the addresses the KMD's own
  bookkeeping uses.

- The embedded PCIe DMA engine is the most direct primitive and needs no iATU and no ioctl. UMD's
  `WormholeTTDevice::dma_d2h_transfer` and `dma_h2d_transfer` program the engine's registers by plain MMIO stores to
  BAR2 (via a local `bar2` pointer to `pci_device_->bar2_uc`), setting a source address, a destination address, a
  size, and a doorbell, then polling a completion word (umd tt_device/wormhole_tt_device.cpp:134-211 and :213-289).
  The host-side destination of a device-to-host transfer is written into the engine's destination-address register
  pair (low and high words; umd tt_device/wormhole_tt_device.cpp:194-195), and UMD passes that value through as a
  caller argument (`dst`). `dma_h2d_zero_copy` even passes a raw user pointer straight into the host-address register
  (umd tt_device/wormhole_tt_device.cpp:324-328). The register pair has a high word, so addresses above 4 GiB can be
  expressed; at least the low 4 GiB of host physical memory is reachable with certainty, and higher addresses
  plausibly (the device's `dma_address_bits` is 32 in the KMD, kmd enumerate.c:328 / wormhole.c:1152, but that mask
  governs the kernel's own coherent allocations, not a destination address a program writes into the engine by hand).

Read vs inferred: the pin-pages branch and the DMA engine register programming are read. That a reprogrammed
register actually steers the hardware to the new address is inferred from the register semantics and from the fact
that the KMD and UMD both program these same registers for legitimate transfers; it was not run.

Confidence: high that the device sees host physical addresses under identity and that the DMA engine's host-address
register is caller-controlled; medium on reachability strictly above 4 GiB.

### 3. Can a program write the iATU registers directly through the mapped BARs, without the ARC?

Yes.

- The outbound iATU registers live in BAR2 at a fixed offset (IATU_BASE 0x1200 relative to the start of BAR2, 16
  outbound regions at stride 0x100, with named low/high base, limit, and low/high target offsets; kmd wormhole.c:46-62).
  The KMD writes them with `iowrite32` to its BAR2 mapping (kmd wormhole.c:375-381, :1099-1105).
- BAR2 is mapped into the program's address space. UMD obtains the BAR mapping offsets from
  `TENSTORRENT_IOCTL_QUERY_MAPPINGS` (umd pcie/pci_device.cpp:418-456) and `mmap`s BAR0 and, for Wormhole, BAR2 UC
  read-write, MAP_SHARED, from the char device (umd pcie/pci_device.cpp:462-468, :479-490). On the KMD side,
  `tenstorrent_mmap` multiplexes BAR0/2/4 by mmap offset and maps them with `vm_iomap_memory` (kmd memory.c:1230-1281,
  :1086-1120). No ioctl and no mmap handler in the KMD checks any capability (the dispatch at kmd chardev.c:465-567
  gates only the reset ioctl, with a lock, and has no `capable()`/CAP_SYS check; the only `capable()` in the files read
  is in the debugfs listing). So any holder of the node maps BAR2 and can store directly into the iATU registers and
  into the DMA engine registers, bypassing both the ARC and the KMD.
- UMD's own DMA engine code (Finding 2) is the proof that user-space direct BAR2 register programming of a
  host-addressed transfer is the normal, intended mechanism, not a hypothetical.

A crucial consequence for the sandbox: writes to a mapped BAR are ordinary CPU store instructions, not system calls.
The sandbox's seccomp filter can refuse syscalls (and it refuses the reset ioctl, request 0xFA06 = _IO(0xFA,6), kmd
ioctl.h:20; sandbox.py RESET_IOCTL and confine_program), but it cannot intercept a store to an mmap. The only syscall
in the DMA/iATU path is the initial `mmap` of BAR2, which UMD needs for all device access and which the executor must
allow. So no seccomp rule in the design reaches the DMA/iATU programming.

Read vs inferred: the register location, the BAR2 mapping, and the absence of a capability check are read. That a
direct store to those offsets produces the same effect as the KMD's `iowrite32` is inferred (same MMIO window, same
offsets); it was not run.

Confidence: high.

### 4. The peer-to-peer BAR ioctl

`TENSTORRENT_IOCTL_MAP_PEER_BAR` (kmd memory.c:754-863) takes a file descriptor of another open card (`peer_fd`),
requires that peer to be a different device of the same class (kmd memory.c:793-801), and maps a window of the
*peer card's* PCI BAR into the calling card's DMA address space via `dma_map_resource`, returning a DMA address (kmd
memory.c:819-835). It maps device MMIO (a peer card's BAR), not arbitrary host RAM. With every card's node bound
(the OQ-075 all-card exposure), a program can open two cards and call this ioctl, so one card's program can reach
another card. Combined with the iATU/DMA primitive (Findings 1 to 3) a program can already reach host memory without
peers, so peer-BAR mainly widens card-to-card reach rather than adding a new host-memory path.

Confidence: high that it needs a second open card and maps a peer BAR; the all-card exposure supplies the second node.

### 5. What closes the hole

- An IOMMU in DMA (translation) mode closes it. If the card's group runs a translating domain instead of identity,
  every PCIe transaction the card emits (from the DMA engine or through any iATU window) is checked by the host IOMMU
  against the IOVA mappings the KMD established, and an address with no mapping faults instead of reaching host RAM.
  This is downstream of the card on the host side, so it holds no matter how the card's internal iATU or DMA engine is
  programmed. The driver fully supports this mode: 2.8.0 is past both `KMD_IOMMU` (1.29.0) and `KMD_MAP_TO_NOC`
  (2.0.0) (umd api/umd/device/utils/kmd_versions.hpp:15, :23), and `detect_iommu` reports IOMMU on only when
  iommu_group/type begins with "DMA" (umd pcie/pci_device.cpp:102-108, :361). When UMD sees an IOMMU it takes its
  no-hugepage path: `SiliconSysmemManager::pin_or_map_sysmem_to_device` branches on `is_iommu_enabled`
  (umd chip_helpers/silicon_sysmem_manager.cpp:45-51), and `init_iommu` mmaps anonymous memory and maps it through the
  KMD, requiring an IOMMU (umd chip_helpers/silicon_sysmem_manager.cpp:260-291). So with a DMA-mode IOMMU UMD does not
  use hugepages at all.
  - How to set it (owner or root, host level): the Linux IOMMU group type is writable. Per the kernel ABI, "A
    privileged user could request kernel to change the group type by writing to this file", and "The default domain
    type of a group may be modified only when ... The device in the group is not bound to any device driver. So, the
    users must unbind the appropriate driver before changing the default domain type." Accepted values include DMA,
    DMA-FQ, identity, and auto (Linux v6.6 Documentation/ABI/testing/sysfs-kernel-iommu_groups). So root can write
    "DMA" to /sys/kernel/iommu_groups/135/type (the card's group, per plans/spikes/tt-silicon-executor.md Finding 2)
    while the tenstorrent module is unbound from that device, or the host can drop iommu=pt from the kernel cmdline so
    every group defaults to a DMA domain. Either is a host policy change outside the sandbox and outside an agent's
    reach.
  - Whether tt-metal at the pin runs end to end under the IOMMU path is unmeasured. UMD supports it; the full metal
    stack was not run this way here.

- A KMD option does not close it. The module has only two parameters, max_devices and dma_address_bits (kmd
  module.c:35-40); neither gates BAR mmap or the DMA engine, and no ioctl checks a capability (Finding 3). The KMD
  does validate the target in its own NOC_DMA path (kernel-chosen address), but it cannot stop a direct BAR write.

- Nothing in user space closes it. Refusing the program UMD's iATU and ARC entry points at build or link time does
  not help: the DMA engine primitive is a direct BAR2 register write (Finding 3), and arbitrary model C++ can compute
  the BAR2 pointer (UMD exposes it, or the program issues QUERY_MAPPINGS and mmap itself) and store to the engine's
  registers without calling any UMD iATU or ARC function. The executor cannot refuse the BAR2 mapping either, because
  UMD needs it for all device access. The reset-ioctl seccomp filter (OQ-075 part 4a) refuses one syscall and does not
  touch MMIO.

Read vs inferred: the UMD IOMMU path and the driver version gates are read; the kernel ABI for the group type is read
from the v6.6 documentation. That a translating IOMMU domain actually blocks a maliciously programmed DMA/iATU is the
standard meaning of DMA remapping, applied here (inference, not run on alpha01). That tt-metal runs cleanly under the
IOMMU path at this pin is unverified.

Confidence: high that a DMA-mode IOMMU is the only thing among the three that closes the hole, and that it needs a
root/host change; medium that the TT stack runs under it at the pin without further work.

## Plain answer

Yes. On alpha01 as configured (the card's IOMMU group type is "identity" from iommu=pt), a user-space program that
holds a card's /dev/tenstorrent node can make the card read or write arbitrary host physical memory (at least the low
4 GiB with certainty), by programming the card's embedded PCIe DMA engine or an outbound iATU region through the
mmap'd BAR2, with no ioctl, no ARC dependency, and no KMD or seccomp check on the host address. Inside the sandboxed
tt_silicon executor as designed (P17.15, OQ-075), the bound card node gives model code exactly this interface; the
reset-ioctl seccomp filter does not reach it, because the programming is MMIO stores, not syscalls. The sandbox
therefore does not isolate the host from a silicon run on this host configuration. The hole is closed only by a
host-level IOMMU in DMA (translation) mode for the card's group (root writes "DMA" to the group's type with the
driver unbound, or the host drops iommu=pt); no KMD option and no user-space or link-time measure closes it.

## Sources

Read on 2026-10-23 from GitHub and the Linux kernel docs.

- https://github.com/tenstorrent/tt-umd/tree/0450f1be3a21cbf17adb97517038586a6a3af4c0: device/tt_device/wormhole_tt_device.cpp,
  device/tt_device/tt_device.cpp, device/api/umd/device/tt_device/tt_device.hpp,
  device/api/umd/device/arch/wormhole_implementation.hpp, device/chip/local_chip.cpp, device/pcie/pci_device.cpp,
  device/pcie/ioctl.h, device/chip_helpers/silicon_sysmem_manager.cpp, device/api/umd/device/utils/kmd_versions.hpp.
- https://github.com/tenstorrent/tt-kmd/tree/ttkmd-2.8.0: wormhole.c, memory.c, chardev.c, enumerate.c, module.c,
  ioctl.h.
- https://github.com/tenstorrent/tt-metal/tree/5280a9cfb00998fd49667a29523d03aee905c129 (the UMD submodule is the
  commit above; the tt-metal facts are those plans/spikes/tt-silicon-executor.md established).
- https://github.com/torvalds/linux/tree/v6.6: Documentation/ABI/testing/sysfs-kernel-iommu_groups (the group type
  attribute and its write conditions). alpha01 runs 6.6.29 with vendor patches.
- plans/spikes/tt-silicon-executor.md (the host facts: identity domain, driver 2.8.0, the card's PCI group 135, the
  two sysmem channels; probes 59e3, 7aeb, 73fc).

## Confidence summary

- High: the capability exists at these pins (the DMA engine and iATU register programming through the user-mapped
  BAR2, the identity-domain pass-through, the absence of a capability or seccomp check on the host address, that only
  the KMD ioctl path confines the target). Every line was read at the pinned commits.
- Medium: reachability strictly above 4 GiB of host physical memory; that a translating IOMMU domain integrates
  cleanly with tt-metal 5280a9cf end to end; that the module built on the host matches tag ttkmd-2.8.0 (assumed, as
  the sibling spike notes).
- Not known: whether the ARC firmware validates an iATU target (moot, since direct BAR writes bypass it).

## Consequences for the plan

- The finding is factual and lands as a Decision Log entry and a sharpening of the Sandbox Tenstorrent-access bullet
  and the Execution Backends tt_silicon row (the proposed text below). The current Sandbox bullet says the sandbox
  "isolates the host, not the cards"; this spike shows that on the identity-domain host the cards are a path to the
  host's memory, so the sandbox does not isolate the host either for a silicon run.
- The remediation is a choice for the owner (OQ-078): run the cards under a DMA-mode IOMMU (a root/host change), or
  accept that a silicon run has full host-memory access. OQ-075 was answered with the reset-ioctl filter as the only
  device-level mitigation, which this spike shows does not confine host memory; OQ-078 puts that back to the owner.
- Until the owner decides, task P17.16 (the tt_silicon executor that runs model output) should not run model-generated
  code on silicon: build the executor and its sandbox wiring, keep silicon runs to tracked references a human starts
  (as the 2026-10-23 smoke did), and gate model output on silicon behind OQ-078. Tracked references are not model
  output (Agent Rule 6), so the smoke is unaffected.

## Proposed bible edit and Decision Log entry

Factual edit. In docs/BIBLE.md, Sandbox, the Tenstorrent-access bullet currently reads (abridged): "a bound card node
gives the program the driver's whole interface to that card (its BARs, DMA, and the card and link resets; the driver
checks no capability), and the reset filter refuses only the reset ioctl, not a reset through a register, so the
sandbox isolates the host, not the cards." Append to that bullet:

```
On a host whose card sits in an identity IOMMU domain (alpha01's iommu=pt; plans/spikes/tt-silicon-executor.md,
Finding 2), that interface reaches host memory, not only the card: the program maps BAR2 and programs the card's
embedded PCIe DMA engine or an outbound iATU region with a host address it chooses, and under the identity domain
that address is a host physical address the card reads or writes with no translation or check (plans/spikes/
tt-card-dma.md, at UMD 0450f1be, tt-metal 5280a9cf, tt-kmd 2.8.0). The programming is MMIO stores, not syscalls, so
the reset-ioctl seccomp filter does not reach it, and no KMD option or user-space or link-time measure closes it; only
a translating (DMA-mode) IOMMU for the card's group does, which is a host change root makes, not an agent. So on this
host a silicon run is not isolated from host memory; model output runs on silicon only behind the owner's decision
(OQ-078), and the 2026-10-23 smoke's human-started tracked references are not model output (Agent Rule 6).
```

Decision Log: insert at the top, newest first, and raise the count sentence (currently "five on 2026-10-23" and "One
hundred and sixty-two decisions") to six on 2026-10-23 and one hundred and sixty-three.

```
| 2026-10-23 | A sandboxed program on a Wormhole card can DMA to arbitrary host memory on an identity-IOMMU host (Sandbox, the Tenstorrent-access bullet; Execution Backends, the tt_silicon row; plans/spikes/tt-card-dma.md; OQ-075; OQ-078). At the pins (UMD 0450f1be, tt-metal 5280a9cf, tt-kmd 2.8.0) a holder of a card's /dev/tenstorrent node maps BAR2 and programs the card's embedded PCIe DMA engine or an outbound iATU region with a host address it chooses; under the card's identity IOMMU domain (iommu=pt) that is a host physical address the card reads or writes untranslated. The KMD validates the target only in its own pin-pages NOC_DMA path (kernel-chosen address); a direct BAR2 register write bypasses the KMD and the ARC, and no ioctl checks a capability. The writes are MMIO stores, not syscalls, so the reset-ioctl seccomp filter (OQ-075 part 4a) does not reach them. Only a DMA-mode IOMMU for the card's group closes it (root unbinds the driver and writes "DMA" to the group type, or the host drops iommu=pt); no KMD option or user-space or link-time measure does | The tt_silicon executor (P17.16) runs model output on silicon through a bound card node (OQ-075), and model output may run only where the sandbox isolates the host (Agent Rule 6). Read from the pinned sources: the DMA engine and iATU register programming through the user-mapped BAR2 (umd tt_device/wormhole_tt_device.cpp, umd pcie/pci_device.cpp), the identity-domain pin-pages branch returning a raw physical address (kmd memory.c:511-515, :658-677), the KMD's target validation only on its own path (kmd wormhole.c:1082-1108, kmd memory.c:123-187), and UMD's IOMMU no-hugepage path (umd chip_helpers/silicon_sysmem_manager.cpp:45-51, :260-291). The host facts (identity domain, driver 2.8.0) are from plans/spikes/tt-silicon-executor.md; nothing ran on alpha01. The remediation is the owner's (OQ-078) |
```

## Owner-queue draft

```
## OQ-078 tt_silicon And Host Memory: A Sandboxed Card Run Can DMA To Arbitrary Host RAM Under iommu=pt
State: OPEN
Kind: decision
Blocks: P17.16 (the tt_silicon executor running model output on silicon); any silicon run of model-generated code
Evidence: plans/spikes/tt-card-dma.md (UMD 0450f1be, tt-metal 5280a9cf, tt-kmd 2.8.0); plans/spikes/tt-silicon-executor.md, Finding 2 (the card's IOMMU domain type "identity" from iommu=pt; rx 20261022-214312-exec-59e3); OQ-075 (the all-card exposure and the reset-ioctl filter)
Question: OQ-075 exposes every Wormhole card to a sandboxed silicon run, with a seccomp filter refusing the driver's reset ioctl as the only device-level mitigation. Reading the pinned sources shows that on alpha01, where each card sits in an identity IOMMU domain (iommu=pt), a program holding a card's node can map BAR2 and program the card's PCIe DMA engine or an outbound iATU region with a host address it chooses, and the card reads or writes that host physical address untranslated. The programming is MMIO stores, not syscalls, so the reset-ioctl filter does not reach it, and no KMD option or user-space or link-time measure closes it. So a sandboxed silicon run of model-generated code (Agent Rule 6) is not isolated from host memory: it can read or write all of host RAM, including other tenants' and the gate's. How should silicon runs of model output proceed?
Options:
- (a) Run the cards under a translating (DMA-mode) IOMMU before any model output runs on silicon. Root unbinds the tenstorrent driver from each card and writes "DMA" to the card's /sys/kernel/iommu_groups/<n>/type, or the host boots without iommu=pt so every group defaults to a DMA domain. Then every card transaction is checked by the host IOMMU and arbitrary host memory is unreachable (plans/spikes/tt-card-dma.md, Finding 5). The driver supports this (2.8.0); UMD switches to its no-hugepage IOMMU path automatically. Consequence: a host change only you or root can make, a reboot or a per-card unbind/rebind, and tt-metal end to end under the IOMMU path is unmeasured at the pin (a qualification step).
- (b) Keep iommu=pt and run model output on silicon anyway, accepting that each silicon run can read or write host memory. Consequence: the sandbox does not isolate the host for these runs; a hostile or buggy candidate can corrupt the host, other tenants, or the gate. Not recommended.
- (c) Never run model output on silicon under iommu=pt: build the tt_silicon executor, but let it run only tracked references a human starts (as the 2026-10-23 smoke did), and keep model-generated TT candidates on ttsim until (a) is in place. Consequence: no host-memory exposure from model code; silicon numbers for model output wait on (a).
Recommendation: (a) if you will make the host change before P17.16 runs model output, else (c) until then; never (b). Either way the tt_silicon executor and its sandbox wiring can be built and tested with stand-ins now (P17.15 is done); only running model output on silicon waits on this.
Answer:
```
