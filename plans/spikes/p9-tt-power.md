# Spike: Tenstorrent power telemetry at the pins

- Task: P9.3 (plans/p9-ee.md). Bible: Execution Backends (the tt_silicon row, the pins), Component Interfaces (the
  Profiler paragraph), Agent Rule 9. Feeds P9.5 (backend `tt`, lassi/profilers/tt_smi.py).
- Status: read at the pins from source fetched from GitHub. No rx command of any kind ran, nothing touched alpha01,
  and no device command ran anywhere. Nothing here is [MEASURED]; every finding is read at the pin, and the
  statements about alpha01's cards that need a probe are listed as not known.
- Date: 2026-10-25.
- Pins (toolchains/tt-metal.pin; plans/spikes/tt-silicon-executor.md, probe 73fc, for UMD; bible Environment State
  for the driver):
  - tt-kmd tag ttkmd-2.8.0, an annotated tag (tag object d6dffeaca67b3fde63908cc7d085ad23652146d5, dated
    2026-04-06) on commit 0b8b5e7c0c44436740ca9ec545e2d08a5c2dfdf1. alpha01 reports driver 2.8.0; that its module
    was built from this tag is assumed, as the sibling spikes assume it.
  - UMD 0450f1be3a21cbf17adb97517038586a6a3af4c0, the submodule of tt-metal 5280a9cfb00998fd49667a29523d03aee905c129.
- Tools read, not pinned by the project (alpha01 has no tt-smi installed; bible Environment State, line for
  2026-10-23). The newest releases on 2026-10-25:
  - tt-smi v6.8.0, commit e25a46520ab915db9ee823a652406b9f44ed8cdd (published 2026-10-22). Its pyproject.toml:30-32
    requires tt_tools_common~=1.6.0, pyluwen~=0.10.0, and tt-umd==0.9.12.
  - luwen v0.10.0, commit c5e623971efa55c6d84aee9ea5d38998178a51ef (published 2026-10-02), the release pyluwen~=0.10.0
    names.
  - tt-umd v0.9.12, commit aa04e17b7ac1bb2de957a6b8ebf482a9a91a1451, the version tt-smi v6.8.0 pins (one file read).
- Citations:
  - `kmd <file>:<line>`: tenstorrent/tt-kmd at ttkmd-2.8.0 (0b8b5e7c).
  - `umd <file>:<line>`: tenstorrent/tt-umd at 0450f1be, paths under device/.
  - `umd-0.9.12 <file>:<line>`: tenstorrent/tt-umd at v0.9.12 (aa04e17b).
  - `smi <file>:<line>`: tenstorrent/tt-smi at v6.8.0 (e25a4652).
  - `luwen <file>:<line>`: tenstorrent/luwen at v0.10.0 (c5e62397).
  - `linux <file>:<line>`: torvalds/linux at v6.6 (alpha01 runs 6.6.29 with vendor patches; v6.6 stands in, as in
    plans/spikes/tt-silicon-executor.md).
- Conventions: upstream code is cited by file and line and paraphrased, never quoted, as the sibling spikes do.

## Question

At the project's pins, which file or call reports Wormhole n300 board power (and energy, if any), in what unit and at
what update interval? Does reading it open /dev/tenstorrent/<n> or message the ARC firmware? Does tt-smi (or luwen,
its library) read it from sysfs hwmon or through a device message? And so: is sampling it every 10 ms during a run a
looped device access under Agent Rule 9 ("one placement at a time, never looped device opens"), or a read of a
kernel sysfs file that opens no device? What must a Tenstorrent energy backend therefore do?

Why it matters:

- P9.5 builds backend `tt` from what this spike names ("reads what P9.3 names"). If the only power source needs a
  device open, the backend would breach Agent Rule 9 by design, since a 10 ms sampler is a loop.
- Every LASSI-EE energy figure for Tenstorrent silicon (P9.6's power protocol) inherits this source's unit,
  resolution, scope (which chip, which rail), and refresh interval.

## Classification

- Which file or call, its unit, whether a read opens the node or messages the ARC, and what tt-smi and luwen read:
  factual, answered by reading the pinned sources (Findings 1 to 6).
- The firmware's telemetry update interval on Wormhole: factual but not answerable from source at the pins, since the
  Wormhole ARC firmware is not in any repository read here; it needs one read-only probe (Finding 5).
- Whether that hwmon directory exists on alpha01's eight cards: factual, needs one read-only sysfs listing (Finding 7).
- Whether LASSI-EE accepts the one power that sysfs gives (the PCIe chip's core power, not n300 board input power) as
  its Tenstorrent measure: a choice (scope of the metric). Drafted below as an owner-queue item.

## Commands and outputs

Every command ran on the workstation (Windows, Git Bash), into a session scratch directory, each archive in its own
empty directory. Nothing ran from inside an extracted tree.

```
curl -sSL -o kmd.tgz https://codeload.github.com/tenstorrent/tt-kmd/tar.gz/refs/tags/ttkmd-2.8.0
  115305 bytes, sha256 cfbed94234b4930513164164d599e44ab8f26ab6a92cbc9d79e9ec1880569376
curl -sS https://api.github.com/repos/tenstorrent/tt-kmd/git/refs/tags/ttkmd-2.8.0
  "object": {"sha": "d6dffeaca67b3fde63908cc7d085ad23652146d5", "type": "tag"}
curl -sS https://api.github.com/repos/tenstorrent/tt-kmd/git/tags/d6dffeaca67b3fde63908cc7d085ad23652146d5
  "date": "2026-04-06T18:56:48Z", "object": {"sha": "0b8b5e7c0c44436740ca9ec545e2d08a5c2dfdf1", "type": "commit"}

curl -sSL -o umd.tgz https://codeload.github.com/tenstorrent/tt-umd/tar.gz/0450f1be3a21cbf17adb97517038586a6a3af4c0
  sha256 b1c19dfdaf7c0f846597d3f22f289cd9cdc9e1cb30ae1d266d46071d24e53e1c
  (tar refused one symlink, compile_commands.json -> build/compile_commands.json; no source file is affected)

curl -sS "https://api.github.com/repos/tenstorrent/tt-smi/releases?per_page=5"
  "tag_name": "v6.8.0", "published_at": "2026-10-22T18:56:32Z"   [... four older releases trimmed ...]
curl -sSL -o smi.tgz https://codeload.github.com/tenstorrent/tt-smi/tar.gz/refs/tags/v6.8.0
  369913 bytes, sha256 2e2a8922d48e691e87eb35549f260bc3c5a3e7fe0dd7a95e3887a19a84010364
curl -sS https://api.github.com/repos/tenstorrent/tt-smi/git/refs/tags/v6.8.0
  "object": {"sha": "e25a46520ab915db9ee823a652406b9f44ed8cdd", "type": "commit"}

curl -sS "https://api.github.com/repos/tenstorrent/luwen/releases?per_page=5"
  "tag_name": "v0.10.0", "published_at": "2026-10-02T16:04:44Z"   [... bh-mod releases trimmed ...]
curl -sSL -o luwen.tgz https://codeload.github.com/tenstorrent/luwen/tar.gz/refs/tags/v0.10.0
  sha256 7e8b16084488a452c318568d21a7a334af6ab63d5e3ce6655c3aa641b0dde6bf
curl -sS https://api.github.com/repos/tenstorrent/luwen/git/refs/tags/v0.10.0
  "object": {"sha": "c5e623971efa55c6d84aee9ea5d38998178a51ef", "type": "commit"}

curl -sS https://api.github.com/repos/tenstorrent/tt-umd/git/refs/tags/v0.9.12
  "object": {"sha": "aa04e17b7ac1bb2de957a6b8ebf482a9a91a1451", "type": "commit"}
curl -sSL https://raw.githubusercontent.com/tenstorrent/tt-umd/aa04e17b7ac1bb2de957a6b8ebf482a9a91a1451/device/arc/smbus_arc_telemetry_reader.cpp
curl -sSL https://raw.githubusercontent.com/tenstorrent/tt-umd/v0.9.12/device/api/umd/device/types/wormhole_telemetry.hpp

curl -sSL https://raw.githubusercontent.com/torvalds/linux/v6.6/drivers/hwmon/hwmon.c
curl -sSL https://raw.githubusercontent.com/torvalds/linux/v6.6/Documentation/ABI/testing/sysfs-class-hwmon
curl -sSL https://raw.githubusercontent.com/torvalds/linux/v6.6/Documentation/hwmon/sysfs-interface.rst
```

Then grep and awk over the extracted trees (for example `grep -rn -i -E "hwmon|power|energy" *.c *.h` in the driver,
`grep -rn -i -E "hwmon|/sys/|power1|sysfs" --include=*.py .` in tt-smi, and `grep -rn "/dev/tenstorrent"` in luwen).
The one arithmetic check:

```
uv run --no-project python -I -c "print((0xFFFF*1000000) % 2**32, 2**32//1000000)"
1110490560 4294
```

## Sources

All read on 2026-10-25.

- https://github.com/tenstorrent/tt-kmd/tree/ttkmd-2.8.0: hwmon.c, hwmon.h, wormhole.c, telemetry.h, chardev.c,
  enumerate.c, module.c.
- https://github.com/tenstorrent/tt-umd/tree/0450f1be3a21cbf17adb97517038586a6a3af4c0:
  device/arc/smbus_arc_telemetry_reader.cpp, device/arc/arc_telemetry_reader.cpp,
  device/arc/wormhole_arc_telemetry_reader.cpp, device/api/umd/device/arc/arc_telemetry_reader.hpp,
  device/api/umd/device/types/wormhole_telemetry.hpp, device/api/umd/device/types/telemetry.hpp,
  device/api/umd/device/arch/wormhole_implementation.hpp, device/arc/wormhole_arc_messenger.cpp,
  device/pcie/pci_device.cpp, device/firmware/firmware_info_provider.cpp,
  device/firmware/wormhole_18_3_firmware_info_provider.cpp.
- https://github.com/tenstorrent/tt-umd/tree/aa04e17b7ac1bb2de957a6b8ebf482a9a91a1451 (v0.9.12):
  device/arc/smbus_arc_telemetry_reader.cpp, device/api/umd/device/types/wormhole_telemetry.hpp.
- https://github.com/tenstorrent/tt-smi/tree/v6.8.0: pyproject.toml, tt_smi/backend.py, tt_smi/tt_smi.py,
  tt_smi/frontend.py, tt_smi/constants.py, tt_smi/utils.py, tt_smi/latest_releases.py.
- https://github.com/tenstorrent/luwen/tree/v0.10.0: crates/luwen-api/src/chip/wormhole.rs,
  crates/luwen-api/src/arc_msg.rs, crates/luwen-kmd/src/lib.rs, apps/prometheus-exporter/src/main.rs.
- https://github.com/torvalds/linux/tree/v6.6: drivers/hwmon/hwmon.c, Documentation/ABI/testing/sysfs-class-hwmon,
  Documentation/hwmon/sysfs-interface.rst.
- https://docs.tenstorrent.com/tt-system-firmware/services/telemetry/index.html (version label TT-System-Firmware
  19.17.99, read 2026-10-25): the telemetry table is updated every 100 ms by a Zephyr worker thread.
- https://docs.tenstorrent.com/tt-system-firmware/doxygen/group__telemetry__tags.html (TT-System-Firmware APIs
  19.17.99, read 2026-10-25): TAG_UPDATE_TELEM_SPEED 5, update interval for telemetry in milliseconds; TAG_VCORE 6,
  millivolts; TAG_TDP 7, watts; TAG_TDC 8, amperes; TAG_BOARD_POWER_LIMIT 53, watts; TAG_INPUT_POWER 54, input power in
  watts; no tag names energy.

## Findings

### 1. The file: the driver's hwmon power1_input, one per PCIe-attached Wormhole chip

- The driver registers a hwmon device for each Wormhole chip it binds: `wormhole_hwmon_init` (kmd wormhole.c:712-740)
  calls `hwmon_device_register_with_info` with the PCI device as parent and the chip name "wormhole" (kmd
  wormhole.c:715, :727). It runs from `init_sysfs_and_hwmon_telemetry` (kmd wormhole.c:753-765), right after the
  telemetry sysfs group (the tt_* attributes) is added, at probe (kmd wormhole.c:837-856; kmd enumerate.c:380-381) or
  from a delayed retry of up to 120 one-second tries while the firmware comes up (kmd wormhole.c:767-782, :791).
- The channels are temp, in, curr, and power (kmd wormhole.c:699-705); power has input, label, and max. The label is
  "power1" (kmd wormhole.c:691-697). There is no energy channel and no update_interval attribute (the chip info
  declares none, kmd wormhole.c:699-710).
- The Linux hwmon core names the device hwmon<k>, in class hwmon, with the PCI device as parent (linux
  drivers/hwmon/hwmon.c:832-839, :31). So the file is
  `/sys/bus/pci/devices/<bdf>/hwmon/hwmon<k>/power1_input`, the same directory as `/sys/class/hwmon/hwmon<k>`, whose
  `name` reads "wormhole". The tenstorrent class device's `device` link points at that PCI device (the listing of
  `/sys/class/tenstorrent/tenstorrent!0` in plans/spikes/tt-silicon-executor.md, probe 59e3, shows
  `device -> ../../../0000:e8:00.0`), so `/sys/class/tenstorrent/tenstorrent!<n>/device/hwmon/hwmon<k>/power1_input`
  reaches it from the card number. The k is assigned at registration and is not stable across boots.
- The attributes are world-readable: the visibility hook returns S_IRUGO for every listed attribute and label (kmd
  hwmon.c:11-29).

### 2. What it reports, and its unit

- A read calls `tt_hwmon_read` (kmd hwmon.c:31-49): one `ioread32` of `telemetry_base + reg_offset`, then shift, mask,
  multiply, divide, all in a u32. For power1_input the entry is offset 0x80, shift 0, mask bits 15:0, multiplier
  1000000, divisor 1 (kmd wormhole.c:686); power1_max is the same word's bits 31:16 (kmd wormhole.c:687).
- The hwmon ABI unit of powerY_input is the microwatt (linux Documentation/ABI/testing/sysfs-class-hwmon:744-750). So
  the driver reports the low 16 bits of the telemetry word, read as whole watts, times 10^6: a value in microwatts at
  1 W resolution.
- What the word is. `telemetry_base` is the address the firmware returns for message 0x2C,
  WH_FW_MSG_GET_TELEMETRY_OFFSET (kmd wormhole.c:118, :243-253, :720-725), the legacy SMBus telemetry block. UMD names
  the same message GET_SMBUS_TELEMETRY_ADDR, 0x2C (umd api/umd/device/arch/wormhole_implementation.hpp:96), and its
  tag list for that block puts TDP at index 32 (umd api/umd/device/types/wormhole_telemetry.hpp:42), byte offset
  32 * 4 = 0x80. The neighbouring hwmon offsets match the same list: VCORE 28 = 0x70, ASIC_TEMPERATURE 29 = 0x74, TDC
  33 = 0x84, VDD_LIMITS 34 = 0x88, THM_LIMITS 35 = 0x8c (kmd wormhole.c:680-687; umd wormhole_telemetry.hpp:38-45).
  luwen reads TDP at word 32 of the same block (luwen crates/luwen-api/src/chip/wormhole.rs:964-966).
- tt-smi shows that word's low 16 bits as the chip's "power" (smi tt_smi/backend.py:720) under the column "Core Power
  (W)", and a separate "board_power" from firmware tag 54, INPUT_POWER, under "Board Power (W)" (smi
  tt_smi/backend.py:721-724, :361-366; tt_smi/constants.py:114-123, :207-217, :138-139). Its limit column for "power"
  is the word's high 16 bits, "tdp_limit" (smi tt_smi/backend.py:809-811). So power1_input is the chip's core power
  (TDP), not the board's input power.
- Scope on an n300. Each /dev/tenstorrent/<n> and each hwmon device belongs to one PCI function. alpha01 has eight
  n300 cards and eight PCI functions, e8:00.0 to eb:00.0 and f1:00.0 to f4:00.0 (bible Environment State, from rx
  20261010-154619-exec-4750), so the driver binds one chip per card, the PCIe-attached one. The card's second chip is
  reached over ethernet (tt-smi treats such chips as remote, smi tt_smi/backend.py:731), has no PCI function, and has
  no hwmon device. power1_input therefore reports one of the n300's two chips, and no sysfs file at the pin reports
  n300 board power.
- No energy. The driver exposes no energy1_input (Finding 1), the firmware tag list names no energy tag (Sources,
  the tag page), and neither UMD's tag list nor luwen's Wormhole telemetry struct has one. Energy must be integrated
  from power samples, which is what the bible's Profiler already does (trapezoid rule over the samples).

### 3. A read opens no device node and sends no ARC message

- Reading the sysfs file is a kernfs read: the hwmon core's show function calls the driver's read callback each time
  and formats the result (linux drivers/hwmon/hwmon.c:314-330). It does not reach the tenstorrent character device,
  so `tt_cdev_open` and `tt_cdev_release` (kmd chardev.c:629-674, :676-725) never run.
- The callback does one `ioread32` through the driver's own BAR4 mapping (kmd hwmon.c:37; the mapping is made at init,
  kmd wormhole.c:802, and `telemetry_base` points into it, kmd wormhole.c:725). That is one PCIe MMIO read from the
  kernel to the card's ARC memory. It takes no lock, allocates no TLB, and sends no message.
- The ARC is messaged once, at registration, for the block's address (message 0x2C, kmd wormhole.c:720 into :243-253,
  which writes the message to scratch register 5 and raises the ARC's IRQ, kmd wormhole.c:207-236). Every later read
  uses the cached address.
- No cache. The hwmon core keeps no value between reads (linux drivers/hwmon/hwmon.c:314-330), and neither does the
  driver (kmd hwmon.c:31-49). Each sample is a fresh read of whatever the firmware last wrote.
- For contrast, what a node open does at 2.8.0: `tt_cdev_open` allocates per-file state, takes a device reference,
  joins the open-file list, and, for a client that does not open with O_APPEND, calls the aggregated power-state path
  (kmd chardev.c:629-674); release frees the file's TLBs, pinnings, and locks, and re-aggregates power when the
  power_policy parameter is on, its default (kmd chardev.c:676-725; kmd module.c:50-52). The Wormhole power-state
  message itself is compiled out at this tag, with a comment that the firmware can take over a second to process it
  and blocks open and close (kmd wormhole.c:1126-1147). A sysfs read does none of this.
- Teardown: the hwmon and tt_* files are removed before the BARs are unmapped, waiting for reads in flight (kmd
  enumerate.c:420-424), so a read never touches an unmapped BAR.

### 4. tt-smi and luwen read power through the device, not sysfs

- tt-smi v6.8.0 has no hwmon read. Its only sysfs reads are the PCI link properties, the driver version, and the
  device index (smi tt_smi/backend.py:525-553; tt_smi/utils.py:69-71, :587-601; tt_smi/latest_releases.py:54); a grep
  for hwmon, power1, or /sys/class over its Python finds nothing else.
- Default path, UMD (smi tt_smi/tt_smi.py:143-146, the --use_luwen flag calls luwen deprecated; :289-296):
  `TopologyDiscovery.discover` opens the cards, and for Wormhole the backend builds a `SmBusArcTelemetryReader` and
  reads every legacy tag, TDP included, at each refresh (smi tt_smi/backend.py:343-367, :385-390). At the pin's UMD
  that reader sends GET_SMBUS_TELEMETRY_ADDR to the ARC when built (umd arc/smbus_arc_telemetry_reader.cpp:17-34),
  under UMD's ARC_MSG lock (umd arc/wormhole_arc_messenger.cpp:70), then reads each tag with `read_from_device`, a NOC
  read through the opened node (umd arc/smbus_arc_telemetry_reader.cpp:36-48); UMD opens the node with `open` on
  /dev/tenstorrent/<n> (umd pcie/pci_device.cpp:353-355). In tt-umd v0.9.12, which tt-smi pins, the reader takes the
  address from its caller and reads through the device protocol (umd-0.9.12 arc/smbus_arc_telemetry_reader.cpp:25-59),
  and its tags stop below 50 (:61-63; umd-0.9.12 types/wormhole_telemetry.hpp:42, :60), so tag 54, board input power,
  is not available on that path for Wormhole and tt-smi prints N/A for it (smi tt_smi/backend.py:361-366, :721-724).
- luwen path: `get_telemetry` sends GetSmbusTelemetryAddr (0x2C) once and caches the address (luwen
  crates/luwen-api/src/chip/wormhole.rs:843-862; crates/luwen-api/src/arc_msg.rs:70), then reads each word, TDP at word
  32, with `axi_read32` through the chip interface (luwen wormhole.rs:864-966), which is the opened node: luwen-kmd
  opens /dev/tenstorrent/<id> read and append (luwen crates/luwen-kmd/src/lib.rs:235-239). luwen's own Prometheus
  exporter samples that way in a sleep loop at a chosen interval (luwen apps/prometheus-exporter/src/main.rs:249-265).
- tt-smi's refresh interval is 0.1 s: its telemetry worker calls the backend's update and sleeps GUI_INTERVAL_TIME
  (smi tt_smi/frontend.py:941-946; tt_smi/constants.py:188).
- tt-metal at the pin uses UMD's reader, which for firmware bundle 18.4.0 or newer is the new-style tag table reader
  (umd arc/arc_telemetry_reader.cpp:20, :26-40; arc/wormhole_arc_telemetry_reader.cpp:15-43), also a device read.

### 5. Update interval: not known at the pins for Wormhole

- The driver defines no update_interval (Finding 1), so the hwmon reports none. The value refreshes when the firmware
  rewrites the word; neither the kernel nor the driver adds a period.
- The firmware publishes its interval as tag 5, UPDATE_TELEM_SPEED, in milliseconds (the firmware tag page, Sources;
  umd api/umd/device/types/telemetry.hpp:16), in the new-style table. UMD treats it as a static entry, read once
  (umd api/umd/device/arc/arc_telemetry_reader.hpp:59-64). The driver exposes no sysfs file for it: its Wormhole
  sysfs attributes are the clocks, serial, card type, firmware versions, ASIC id, and heartbeat (kmd
  wormhole.c:389-403; kmd telemetry.h:10-29). So the interval can be read only through a device open.
- The firmware documentation's "every 100 ms" (Sources) describes the Zephyr worker of tt-system-firmware 19.17.99,
  and nothing read here ties that statement to the Wormhole ARC firmware of alpha01's bundle 19.5.0.0. It is not
  adopted as Wormhole's interval.
- Consequence: a 10 ms sampler may read the same value up to ten times between firmware updates (at 1 W
  resolution). That does no harm to the trapezoid rule, but it adds no information either, and it does not resolve
  power changes faster than the firmware's update.

### 6. Failure image and range

- A hung card or a dead link returns all ones on an MMIO read; the driver itself treats 0xFFFFFFFF as a possible hang
  (kmd wormhole.c:182-185). In `tt_hwmon_read` all ones becomes 0xFFFF after the mask, and 0xFFFF * 1000000 wraps in
  the u32 to 1110490560 (the arithmetic check above). So a hung card reads power1_input = 1110490560 (about 1110.49 W)
  and the same for power1_max, not an error.
- Any honest reading above 4294 W would also wrap, far beyond any n300 chip.
- If the firmware never published its telemetry, or message 0x2C failed, the hwmon device is absent (kmd
  wormhole.c:720-721, :738-739, :781), so a missing directory means no power telemetry, not zero power.

### 7. Not known without a probe

- Whether alpha01's cards carry the hwmon directory. Probe 59e3 (plans/spikes/tt-silicon-executor.md) shows card 0's
  tt_* files, so `telemetry_probe` succeeded on that card (kmd wormhole.c:756-761), but it did not list the PCI
  device's hwmon directory, and `wormhole_hwmon_init` can still fail on message 0x2C after the tt_* group is added
  (kmd wormhole.c:758-762, :720-721).
- Whether bundle 19.5.0.0 still refreshes the legacy SMBus block. UMD at the pin reads the new-style table for that
  firmware (Finding 4), while the 2.8.0 driver (2026-04) and tt-smi v6.8.0 (2026-10) still read the legacy block, which
  suggests it is maintained. That is inferred, not read.
- The firmware's refresh interval on Wormhole (Finding 5).

One read-only probe in an owner window would settle the first two and bound the third, with no device open: list
`/sys/bus/pci/devices/0000:e8:00.0/hwmon/*/` and read `name`, `power1_input`, `power1_max`, and `power1_label` for
each of the eight PCI addresses; then read one card's power1_input every 1 ms for 5 s and log when the value changes.
The gate's tt_silicon patterns (tools/server/config.default.json:34) name /dev/tenstorrent and tt-smi, not /sys paths.
A card at rest may hold one value for the whole 5 s at 1 W resolution, so the change log can only bound the interval
from above.

## Verdict on Agent Rule 9

Sampling `power1_input` every 10 ms is a read of a kernel sysfs file. It opens no /dev/tenstorrent node, so none of
the open or release work of Finding 3 runs, and it sends no ARC message; each read is one MMIO load the driver issues
through its own BAR4 mapping. It is therefore not a looped device open under Agent Rule 9. A Tenstorrent energy
backend must never reach power the way tt-smi or luwen do (an open of the node, an ARC message, and a NOC read per
refresh). Sampled at 10 ms, that path would be a device access in a loop, beside the placement the run holds.

What the backend must do (P9.5):

- Resolve the card's hwmon directory from the card number, without opening anything: readlink
  `/sys/class/tenstorrent/tenstorrent!<n>/device`, then list `<that>/hwmon/`, and require exactly one `hwmon<k>` whose
  `name` reads "wormhole". Never take a fixed `/sys/class/hwmon/hwmon<k>` path, since k is not stable.
- Read `power1_input` as integer microwatts and convert to watts (int(text) / 1e6). Values come in whole watts.
- Treat a missing directory, a missing file, or an unreadable value as TelemetryUnavailable when built and as a failed
  sample later (the Profiler paragraph's rules). Treat 1110490560 (the all-ones image, Finding 6) as a failed sample and stop sampling that card for the rest of the window, so a hung card is not read every 10 ms;
  and any value above `power1_max` read at build time as suspect.
- Sample every card a run may use. The sandbox gives a silicon run every card (OQ-072, OQ-075 (2a)), so a run's power
  is at most the sum over the cards it placed on.
- Name what it measures: the core power (TDP word) of each card's PCIe-attached chip, at 1 W resolution and the
  firmware's refresh interval. This is not n300 board power. It omits the second chip and board input power (tag 54),
  which only a device open reaches.
- Declare supports_power and takes_device. Refuse it beside ttsim (Agent Rule 2) and outside a human-started
  `--silicon` run (Agent Rule 9), as P9.5's acceptance says. Start no tool and import no tt-smi, luwen, or UMD.

## Confidence

- High, read at the pin: the hwmon device, its path, its unit and resolution, the offset-to-TDP mapping, that a read
  does no open and sends no message, that nothing caches, and that tt-smi and luwen read through the device. Every line
  above was read at the named commits.
- Medium: that TDP is the PCIe chip's core power rather than board power. This rests on tt-smi's column labels and
  its separate tag-54 board power, not on firmware source.
- Not known: the hwmon directory on alpha01's cards, the legacy block's refresh at bundle 19.5.0.0, and the
  Wormhole interval (Finding 7).

Side note for plans/spikes/tt-card-dma.md: it says the module has only two parameters (kmd module.c:35-40). At the
tag there are five: max_devices, dma_address_bits, reset_limit, auto_reset_timeout, and power_policy (kmd
module.c:34-52). None gates BAR mapping or DMA, so that spike's Finding 5 stands.

## Consequences for the plan

- P9.5 can build backend `tt` on a sysfs source with fakes, the way rocm_smi reads power1_average. Its synthetic tree
  is `sys/class/tenstorrent/tenstorrent!<n>/device` -> `sys/bus/pci/devices/<bdf>`, holding `hwmon/hwmon<k>/{name,
  power1_input, power1_max, power1_label}`. It needs no device kind beyond the one the tt_silicon executor already
  names, unless the owner picks option (b) below.
- P9.6's power protocol must label Tenstorrent energy as the PCIe chip's core energy integrated at the firmware's
  refresh, never as n300 board energy.
- The owner-window batch (P9.2's probe, P9.7, P9.12) can carry the read-only sysfs probe of Finding 7. It needs no
  device class and opens no node, but it is an rx command, so it waits for the owner's yes (Agent Rule 17).

## Proposed bible edit (Execution Backends, a paragraph after the executor table, before ttsim Facts)

> Tenstorrent power telemetry, read at the pins (task P9.3; plans/spikes/p9-tt-power.md; tt-kmd 2.8.0, UMD 0450f1be,
> tt-metal 5280a9cf): the tenstorrent driver registers one hwmon device, named wormhole, for each PCIe-attached
> Wormhole chip, as a child of the card's PCI device. Its `power1_input`, reached as
> `/sys/class/tenstorrent/tenstorrent!<n>/device/hwmon/hwmon<k>/power1_input`, reports that chip's power in microwatts
> at 1 W resolution: the low 16 bits of the firmware's TDP telemetry word, which tt-smi shows as core power, times
> 1,000,000. `power1_max` is the TDP limit from the high 16 bits. Each read is one MMIO read by the driver through its
> own BAR4 mapping: no cache, no lock, no ARC message (the driver asks the firmware for the block's address once, at
> probe), and no open of `/dev/tenstorrent/<n>`. The file is world-readable. A hung card reads all ones, which the
> driver's 32-bit arithmetic turns into 1110490560. There is no energy counter and no update_interval attribute. n300
> board input power (firmware tag 54, which tt-smi 6.8.0 shows as N/A on Wormhole and which bundle 19.5.0.0 may not
> publish there) and the card's second, ethernet-attached chip are reachable, if at all, only through a device open,
> never through sysfs. Sampling `power1_input` every 10 ms is a sysfs read that opens no device and sends no firmware
> message, not a looped device open under Agent Rule 9; whether reads at that rate beside a run are harmless to the
> card is untested. Which Tenstorrent power LASSI-EE reports is the owner's choice (OQ-093); until it is answered, a
> Tenstorrent power backend reads only this file and never opens the node or starts a tool, and a path that opens the
> node would need an owner ruling under Agent Rule 9. [OPEN] until a read-only probe in an owner window: whether
> alpha01's cards (firmware bundle 19.5.0.0) carry the hwmon directory, whether that bundle still refreshes the legacy
> telemetry block power1_input reads (inferred from tt-kmd 2.8.0 and tt-smi 6.8.0 still reading it), and the
> firmware's telemetry refresh interval, which no source at the pins states for Wormhole.

## Proposed Decision Log entry (top of the table; add one to the total and to 2026-10-25's count)

| 2026-10-25 | Tenstorrent power telemetry is the tenstorrent driver's hwmon `power1_input`, read from sysfs (Execution Backends, the Tenstorrent power telemetry paragraph; Agent Rule 9; task P9.3; plans/spikes/p9-tt-power.md). At tt-kmd 2.8.0 each PCIe-attached Wormhole chip has a hwmon device named wormhole under its PCI device; `power1_input` is the firmware's TDP word's low 16 bits in microwatts, 1 W resolution, which tt-smi labels core power, not n300 board power (which power LASSI-EE reports: OQ-093). A read is one driver MMIO load with no node open and no ARC message, so a 10 ms sampler is not a looped device open; there is no energy counter, and the refresh interval, whether bundle 19.5.0.0 still refreshes the legacy block, and the directory's presence on alpha01's cards are [OPEN] until a read-only probe | Backend `tt` (P9.5) needs a power source that a 10 ms sampler can poll without breaching Agent Rule 9. Read at the pins: the hwmon registration and read path (kmd wormhole.c:679-740, hwmon.c:11-49), the TDP index 32 = offset 0x80 (umd types/wormhole_telemetry.hpp:42), the hwmon core's uncached show (linux v6.6 drivers/hwmon/hwmon.c:314-330) and microwatt unit (sysfs-class-hwmon:744-750), and the node open and release work a sysfs read avoids (kmd chardev.c:629-725). tt-smi 6.8.0 (e25a4652) and luwen 0.10.0 (c5e62397) read the same word through an opened node after an ARC message (smi tt_smi/backend.py:343-367; luwen chip/wormhole.rs:843-966), which a sampler must not copy. No rx command ran |

## Owner-queue item (filed as OQ-093 on 2026-10-25)

```
## OQ-093 What Tenstorrent Power LASSI-EE Reports
State: OPEN
Kind: decision
Blocks: P9.6 (the Tenstorrent half of the power protocol); P9.5 builds option (a) behind a config value with no default
Evidence: plans/spikes/p9-tt-power.md
Question: At the pins, the only Tenstorrent power that a sampler can read without opening a card's /dev/tenstorrent
node is the driver's hwmon power1_input: the core power (TDP word) of each n300's PCIe-attached chip, in whole watts,
refreshed at an interval no source at the pins states for Wormhole. It omits the n300's second chip and the board's
input power. Those (firmware tag 54, and the remote chip's telemetry) are reachable only by opening the node,
messaging the ARC, and reading over the NOC, as tt-smi and luwen do where the firmware publishes them (tt-smi 6.8.0 shows tag 54 as N/A on Wormhole). Which Tenstorrent power should LASSI-EE report?
Options:
- (a) The sysfs core power of each placed card's PCIe chip, labeled exactly so ("PCIe-chip core power, 1 W
  resolution"), summed over the cards a run places on. Consequence: no node open, no ARC message, and Agent Rule 9
  holds; but the figure under-reports n300 board energy (second chip, board overhead), so it compares only
  Tenstorrent runs with each other, never Tenstorrent against GPU board power.
- (b) Board input power and both chips, read through UMD in a separate process that opens the node once per run and
  reads telemetry over the NOC every interval. Consequence: a second open of the node during the placement, NOC
  traffic beside the run, an ARC message at start, and a sampler in a loop on an open device. That needs an owner
  ruling under Agent Rule 9, and it is unverified whether bundle 19.5.0.0 publishes tag 54 on Wormhole.
- (c) No Tenstorrent energy in LASSI-EE: backend tt declares it cannot measure, like cerebras. Consequence: no
  Tenstorrent energy results until a board-level meter exists.
Recommendation: (a). It is the only source that keeps Agent Rule 9 by construction. Its scope is stated, so no figure
is mislabeled, and the read-only sysfs probe (spike, Finding 7) can confirm it in the next owner window.
Answer:
```
