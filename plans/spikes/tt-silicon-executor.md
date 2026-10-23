# Spike: what a sandboxed tt_silicon executor needs

- Owner item: OQ-070 (answered 2026-10-23; "Still to apply: ... the sandboxed tt_silicon executor for model output
  (plans/spikes/tt-silicon-executor.md)"). Bible: Execution Backends (the tt_silicon row and its paragraph), Sandbox,
  Agent Rules 6, 7, and 9.
- Status: the pinned sources are read and the host state was checked read-only on alpha01. Nothing ran in the sandbox
  with a device, so every statement about a sandboxed silicon run is from reading, not measured.
- Date: 2026-10-23. Eight read-only probes, rx 20261022-213241-exec-45d6 to rx 20261022-214312-exec-59e3, ran from
  21:32 to 21:43 on the alpha01 clock (UTC-07:00), which is 00:32 to 00:43 EDT on 2026-10-23. That is inside the owner's
  window of 00:00 to 06:00 EDT, with the gate's tt_silicon class enabled (OQ-070). Each probe ran from the scratch root
  without a checkout and used only grep, sed, ls, readlink, stat, readelf, dpkg-query, and cat of procfs, sysfs, and
  DMI counters. No probe ran a program that opens a device or used tt-smi, and /dev/tenstorrent was only listed with
  `ls -l`.
- Pins: tt-metal 5280a9cfb00998fd49667a29523d03aee905c129 (toolchains/tt-metal.pin). Its UMD submodule is
  0450f1be3a21cbf17adb97517038586a6a3af4c0: probe 73fc printed that commit from `git rev-parse HEAD` in the pinned
  tree's tt_metal/third_party/umd.
- Host: alpha01 has eight Wormhole n300 cards, the tenstorrent driver 2.8.0, and firmware bundle 19.5.0.0. These facts
  come from the smoke record (results/tt-silicon-smoke-20261023/summary.md) and were not read again here.
- Citations:
  - `umd <file>:<line>`: tenstorrent/tt-umd at 0450f1be (paths under device/).
  - `metal <file>:<line>`: tenstorrent/tt-metal at 5280a9cf.
  - `kmd <file>:<line>`: tenstorrent/tt-kmd at tag ttkmd-2.8.0.
  - `linux <file>:<line>`: torvalds/linux at v6.6.
  - `hwloc <file>:<line>`: open-mpi/hwloc at hwloc-2.7.0.
  - Repository files (sandbox.py, gpu.py, ttsim.py) are cited at HEAD c5a9a1f, whose lassi/executors is unmodified
    in the working tree.
- Conventions: upstream code is cited by file and line and paraphrased, never quoted, as
  plans/spikes/p4-ttsim-runtime.md does. Only host output is quoted.

## Question

What exactly must a sandboxed `tt_silicon` executor give a program so that model-generated TT host programs (the Tier
A cpp->tt candidates) can run on one Wormhole n300 card? Agent Rule 6 requires the sandbox for model-generated code,
and Agent Rule 9 allows one placement at a time.

1. Which device nodes UMD opens, how it enumerates them, and whether binding only node N is enough.
2. Which sysfs and procfs paths it reads, and whether the existing sys_dirs exposure can provide them.
3. Hugepages: which paths and file names, whether the files persist and are reused, which counters UMD reads, and
   whether the rootless sandbox can mount a private hugetlbfs.
4. Other host state: /dev/shm, locks, /tmp, environment variables, writes into the runtime root, and the JIT cache.
5. What the sandbox must change, and the isolation risk of each change.
6. A proposed design, its tests, and the bible rows it changes.
7. The open questions that need the owner.

Why it matters:

- The smoke runs of 2026-10-23 ran tracked references outside the sandbox. A model's TT candidates can reach silicon
  only through a sandboxed executor, and no such executor exists (bible, Execution Backends: "Not built").
- OQ-070's remaining work, the tt_silicon executor task, still to be planned, depends on this spike.
- Any later phase that evaluates TT output on silicon depends on it too.

## Classification

- Questions 1 to 4 are factual. The pinned sources, the driver's source, and the host's state answer them.
- Questions 5 and 6 are factual where they say what the sandbox must change. The choices among options are the
  owner's: which hugepage exposure, the lock's scope, the card key, the policy after a hang, accepting the device-level
  risk, and how a human starts a run.
- Question 7 lists those choices. They are drafted below as OQ-072.

## Commands and outputs

Each probe ran as `uv run tools/rx.py exec -- '<command>'`. Outputs are trimmed where `[... trimmed ...]` says so, and
upstream source lines that a probe printed are paraphrased in Findings rather than quoted.

1. rx 20261022-213241-exec-45d6. In the pinned tree, this probe grepped tt_metal/*.cpp and *.hpp for the generated
   fabric files and for get_visible_devices, start_monitoring, shm_open, boost::interprocess, /dev/shm, and flock(.
   Output, trimmed:

   ```
   tt_metal/fabric/control_plane.cpp:519:    // Automatically export physical chip mesh coordinate mapping to generated/fabric directory [... trimmed ...]
   tt_metal/fabric/control_plane.cpp:525: [... trimmed ...]
   tt_metal/fabric/control_plane.cpp:537: [... trimmed ...]
   ===
   tt_metal/llrt/rtoptions.hpp:553:    std::string get_visible_devices() const { return visible_devices; }
   ```

   So tt_metal outside third_party has no shm_open, boost::interprocess, /dev/shm, flock, or start_monitoring, and
   nothing reads get_visible_devices.

2. rx 20261022-213249-exec-cf07. This probe printed `sed -n 515,545p tt_metal/fabric/control_plane.cpp` (paraphrased in
   Finding 4) and grepped tt_metal for literal /tmp, /proc/, /sys/, and /dev/ paths, then listed the tree's generated/
   directory.

   ```
   tt_metal/jit_build/build.cpp:92:    return "/tmp/tt-metal-cache/";
   tt_metal/impl/program/program.cpp:177:    static std::ofstream f("/tmp/hashlog.txt");
   tt_metal/impl/debug/watcher_server.cpp:275:        fname = "/dev/null";
   tt_metal/fabric/physical_system_descriptor.cpp:38:    std::ifstream file("/sys/class/dmi/id/board_name");
   ===
   joseph_ufl 664 2026-10-22 21:20:39.333886231 -0700 generated/fabric/asic_to_fabric_node_mapping_rank_1_of_1.yaml
   joseph_ufl 664 2026-10-22 21:20:39.333886231 -0700 generated/fabric/physical_chip_mesh_coordinate_mapping_1_of_1.yaml
   ```

3. rx 20261022-213300-exec-7aeb. This probe read the hugetlbfs lines of /proc/mounts, listed both hugepage directories,
   read the 1 GiB pool counters, listed UMD files in /dev/shm, checked three /tmp paths, and listed /dev/tenstorrent.
   Output, trimmed:

   ```
   hugetlbfs /dev/hugepages hugetlbfs rw,relatime,pagesize=2M 0 0
   hugetlbfs /dev/hugepages-1G hugetlbfs rw,nosuid,nodev,relatime,mode=777,pagesize=1024M 0 0
   ===
   /dev/hugepages-1G:
   drwxrwxrwx  3 root       root                0 Oct 22 21:18 .
   -rw-rw-rw-  1 joseph_ufl joseph_ufl 1073741824 Oct 22 21:18 device_0_channel_1_tenstorrent
   drwxr-xr-x  3 root       root                0 Oct 22 10:46 libvirt
   -rw-rw-rw-  1 joseph_ufl joseph_ufl 1073741824 Oct 22 21:18 tenstorrent
   ===
   nr_hugepages 32
   free_hugepages 30
   resv_hugepages 0
   surplus_hugepages 0
   ===
   -rw-rw-rw-  1 joseph_ufl joseph_ufl   56 Oct 22 21:18 TT_UMD_LOCK.ARC_MSG
   -rw-rw-rw-  1 joseph_ufl joseph_ufl   56 Oct 22 21:18 TT_UMD_LOCK.ARC_MSG_0_PCIe
   -rw-rw-rw-  1 joseph_ufl joseph_ufl   56 Oct 22 21:18 TT_UMD_LOCK.CHIP_IN_USE_0_PCIe
   -rw-rw-rw-  1 joseph_ufl joseph_ufl   56 Oct 22 21:18 TT_UMD_LOCK.MEM_BARRIER_0_PCIe
   -rw-rw-rw-  1 joseph_ufl joseph_ufl   56 Oct 22 21:18 TT_UMD_LOCK.NON_MMIO_0_PCIe
   -rw-rw-rw-  1 joseph_ufl joseph_ufl   56 Oct 22 21:18 TT_UMD_LOCK.PCIE_DMA_0_PCIe
   -rw-rw-rw-  1 joseph_ufl joseph_ufl   56 Oct 22 21:18 TT_UMD_LOCK.REMOTE_ARC_MSG_0_PCIe
   ===
   ls: cannot access '/tmp/tt_umd_listeners': No such file or directory
   ls: cannot access '/tmp/tt-metal-cache': No such file or directory
   ls: cannot access '/tmp/hashlog.txt': No such file or directory
   ===
   crw-rw-rw- 1 root root 506, 0 Oct 22 10:44 0
   [... the same for 1 to 7 ...]
   drwxr-xr-x 2 root root    200 Oct 22 10:44 by-id
   lrwxrwxrwx 1 root root 4 Oct 22 10:44 wormhole-3DFB5316BCE44A16 -> ../0
   [... seven more links, one per card ...]
   ```

   /dev/hugepages lists only a root-owned libvirt directory.

4. rx 20261022-213441-exec-32e2. This probe printed physical_system_descriptor.cpp:30-52 (paraphrased in Finding 4),
   the NEEDED entries of the tree's libdevice.so, the hwloc packages, and card 0's sysfs links. Output, trimmed:

   ```
    0x0000000000000001 (NEEDED)             Shared library: [libhwloc.so.15]
   [... trimmed ...]
   libhwloc-dev 2.7.0-2ubuntu1
   libhwloc-plugins 2.7.0-2ubuntu1
   libhwloc15 2.7.0-2ubuntu1
   ===
   /sys/devices/pci0000:e2/0000:e2:01.1/0000:e3:00.0/0000:e4:00.0/0000:e5:00.0/0000:e6:01.0/0000:e8:00.0
   ../../../devices/pci0000:e2/0000:e2:01.1/0000:e3:00.0/0000:e4:00.0/0000:e5:00.0/0000:e6:01.0/0000:e8:00.0
   ../../../../../../../../kernel/iommu_groups/135
   0
   0-1
   0-2
   0-3
   1
   60
   ```

   The output lines are, in order:
   - readlink -f of /sys/class/tenstorrent/tenstorrent!0/device;
   - the /sys/bus/pci/devices/0000:e8:00.0 link;
   - its iommu_group link;
   - the first five entries of /sys/bus/pci/slots, then the count of its entries.

5. rx 20261022-213741-exec-73fc. This probe ran `git rev-parse HEAD` in tt_metal/third_party/umd, then grepped device/
   (tests excluded) for literal /sys, /proc, /dev, /tmp, and getenv(. Its first line was
   0450f1be3a21cbf17adb97517038586a6a3af4c0. The grep matches are exactly the ones the GitHub copy at that commit
   shows, cited in Findings 1 to 4: pci_device.cpp, hugepage.cpp, silicon_sysmem_manager.cpp, cpuset_lib.cpp,
   warm_reset.cpp and warm_reset.hpp, common/utils.hpp, jtag_device.cpp, and the simulation files.

6. rx 20261022-213930-exec-4ad5. This probe printed program.cpp:168-182, which shows that the /tmp/hashlog.txt stream
   sits inside `#ifdef GENERATE_HASH_LOG`. It also printed the DMI board name and physical_system_descriptor.cpp:52-75.

   ```
   H13DSH
   ```

7. rx 20261022-213954-exec-c1f7. This probe grepped tt_metal (third_party and tests excluded) for
   reset_device_ioctl, WarmReset, warm_reset(, and TT_VISIBLE_DEVICES. Output: none for either grep.

8. rx 20261022-214312-exec-59e3. This probe read the cgroup2 mount and its controllers, listed card 0's class
   directory, and read card 0's IOMMU domain type and the overcommit setting. Output, trimmed:

   ```
   cgroup2 /sys/fs/cgroup cgroup2 rw,nosuid,nodev,noexec,relatime 0 0
   cpuset cpu io memory hugetlb pids rdma misc
   memory pids
   ===
   lrwxrwxrwx 1 root root    0 Oct 22 10:44 device -> ../../../0000:e8:00.0
   -r--r--r-- 1 root root 4096 Oct 22 21:43 tt_aiclk
   [... every tt_* attribute is -r--r--r-- root ...]
   -rw-r--r-- 1 root root 4096 Oct 22 10:44 uevent
   ===
   identity
   0
   ```

   The third line is the controllers delegated to the user's systemd instance (user@<uid>.service). The last two
   lines are /sys/bus/pci/devices/0000:e8:00.0/iommu_group/type and /proc/sys/vm/nr_overcommit_hugepages.

## Sources

All were read on 2026-10-23 from GitHub, by raw file or the contents API.

- https://github.com/tenstorrent/tt-umd/tree/0450f1be3a21cbf17adb97517038586a6a3af4c0: device/pcie/pci_device.cpp,
  device/hugepage.cpp and .hpp, device/chip_helpers/silicon_sysmem_manager.cpp, device/chip/local_chip.cpp,
  device/utils/lock_manager.cpp, device/utils/robust_mutex.cpp, device/cpuset_lib.cpp,
  device/topology/topology_discovery.cpp, device/cluster.cpp, device/warm_reset.cpp,
  device/api/umd/device/warm_reset.hpp, device/tt_kmd_lib/tt_kmd_lib.c, device/pcie/ioctl.h,
  device/api/umd/device/utils/kmd_versions.hpp, device/common/utils.hpp. Probe 73fc confirmed that this is the
  pinned tree's UMD.
- https://github.com/tenstorrent/tt-metal/tree/5280a9cfb00998fd49667a29523d03aee905c129: tt_metal/llrt/tt_cluster.cpp,
  tt_metal/llrt/rtoptions.cpp and rtoptions.hpp. The probes above read the other tt-metal files in the build host's
  pinned tree.
- https://github.com/tenstorrent/tt-kmd/tree/ttkmd-2.8.0: chardev.c, memory.c, enumerate.c, module.c, wormhole.c,
  pcie.c. The host's driver reports 2.8.0, but that the module was built from this tag is assumed, not checked.
- https://github.com/torvalds/linux/tree/v6.6: fs/hugetlbfs/inode.c, fs/super.c, fs/namespace.c, mm/hugetlb.c,
  mm/shmem.c. alpha01 runs 6.6.29 with vendor patches (the smoke record's kernel string), so v6.6 stands in for the
  running source.
- https://github.com/open-mpi/hwloc/tree/hwloc-2.7.0: hwloc/topology-linux.c. Probe 32e2 shows the host package is
  2.7.0-2ubuntu1, and Ubuntu's patches were not read.

## Findings

### 1. Device nodes

- How UMD enumerates (reading):
  - UMD lists /dev/tenstorrent/ and keeps only all-digit names (umd pcie/pci_device.cpp:913-932, iterating at
    :922-927), so the by-id links are skipped and never used.
  - PCIDevice::enumerate_devices (:248-332) returns that whole list when TT_VISIBLE_DEVICES is unset (:256-258).
  - When TT_VISIBLE_DEVICES is set, enumerate_devices builds a map from PCI address to device number (:268-269)
    whatever the tokens are. The map opens every listed node, O_RDWR, reads its PCI address with
    TENSTORRENT_IOCTL_GET_DEVICE_INFO, and closes it (:934-954, open at :938). An integer token not in the list
    throws (:306-316).
  - enumerate_devices_info opens each enumerated node the same way (:334-350).
- How often (reading):
  - tt-metal on silicon runs UMD's topology discovery twice per start. The first time is the temporary descriptor
    (metal tt_cluster.cpp:379, via umd cluster.cpp:1023-1029). The second is inside the UMD Cluster it then builds
    (umd cluster.cpp:347-350, from metal tt_cluster.cpp:385).
  - Each discovery enumerates twice (umd topology/topology_discovery.cpp:39 and :97).
  - So every listed node is opened at least twice per start (once per discovery, by enumerate_devices_info). With
    TT_VISIBLE_DEVICES set, it is opened at least four times, since each enumerate_devices also builds the map.
- Consequence for the smoke runs (reading, not counted): outside the sandbox, with TT_VISIBLE_DEVICES=0, UMD opened
  all eight cards' nodes at least four times per program start, each time for a device-info ioctl and a close. Only
  card 0 was then used.
  - These enumeration opens pass O_APPEND (:213, :337, :938), which the 2.8.0 driver treats as power-aware on close
    (kmd chardev.c:681-685).
  - Whether such an open and close sends a card an ARC message was not traced.
- The chosen card: PCIDevice's constructor opens /dev/tenstorrent/<n> O_RDWR (umd pcie/pci_device.cpp:353-355), and
  tt_device_open opens the same path a second time (umd tt_kmd_lib/tt_kmd_lib.c:66-73). It then maps BAR0, and on
  Wormhole BAR4, from that descriptor (:462-509) and allocates a DMA buffer through the driver (:511, :815-853).
- The n300's second chip is reached over ethernet. Remote chips on boards outside the local set are left out of the
  cluster (umd topology/topology_discovery.cpp:200-210). The smoke log showed only "remote chip ids {1}".
- Is node N enough? Binding only /dev/tenstorrent/<n>, at its own path, makes UMD see and open exactly that card,
  TT_VISIBLE_DEVICES set to <n> or unset. That is stronger than TT_VISIBLE_DEVICES alone, which does not stop the
  enumeration opens.
  - The node is necessary, not sufficient: the run also needs Findings 2 and 3.
  - Keeping the host number <n>, rather than binding the node at /dev/tenstorrent/0, keeps UMD's hugepage file names
    and lock names (Findings 3 and 4) and the card's sysfs name tenstorrent!<n> consistent with each other.
- The host's nodes are `crw-rw-rw- root` with major 506 (probe 7aeb).
- The by-id names (wormhole-<hex> -> ../<n>) are udev links that UMD does not read.

### 2. sysfs and procfs

What UMD and tt-metal read at the pins, for the chosen card (reading):

- `/sys/bus/pci/slots/` (umd pcie/pci_device.cpp:134-175): read_device_info calls it for every device (:201),
  PCIDevice's constructor included (:356). The directory is iterated with std::filesystem::directory_iterator (:137),
  which throws when the directory is missing, so a missing slots directory makes every PCIDevice construction fail.
  It must exist; its entries may be any.
- `/sys/bus/pci/devices/<bdf>/` attributes:
  - revision is required (:358, read_sysfs throws when it is missing, :80-94);
  - numa_node is optional, defaulting to -1 (:357);
  - iommu_group/type is optional (:102-108): only a value starting with DMA means an IOMMU;
  - config is read only by the warm-reset path (:772-790, umd warm_reset.cpp:205).
- `/sys/module/tenstorrent/version` (:734-747): the driver version, which gates features (umd kmd_versions.hpp).
- `/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages` (umd hugepage.cpp:30-45): it throws when it cannot
  open the file.
- `/proc/mounts` (umd hugepage.cpp:109-149): see Finding 3.
- hwloc, through UMD's cpuset allocator (umd cpuset_lib.cpp:84-95, :113-117): it counts the Tenstorrent PCI devices
  hwloc finds, and that count sets the hugepage channels (Finding 3).
  - hwloc 2.7.0 lists `/sys/bus/pci/devices/` and finds no PCI device when the list is missing (hwloc
    topology-linux.c:6679-6697).
  - For each entry it reads config, class, vendor, and device (:6727, :6734, :6798, :6803), and local_cpus for
    locality (:5582).
  - It reads `/sys/bus/pci/slots/` too, optionally (:6850-6868).
  - The allocator also lists `/sys/bus/pci/devices/<bdf>/tenstorrent/` to find tenstorrent!<n> (umd
    cpuset_lib.cpp:121).
- Printed only after a failure: /proc/cmdline, /proc/meminfo, and /proc/buddyinfo (umd
  silicon_sysmem_manager.cpp:171-173, :239-241).
- tt-metal reads `/sys/class/dmi/id/board_name` (metal physical_system_descriptor.cpp:38). It tolerates a missing
  file as an empty name, and a name outside its two-entry table gives tray 0 (probes 32e2 and 4ad5). alpha01's board is
  H13DSH, which is outside the table, so the result is the same inside and outside the sandbox.
- No reader of /sys/class/tenstorrent: neither UMD's device/ (probe 73fc) nor tt-metal reads it. Only the executor
  needs it, outside the sandbox, for the record, as smoke.py's card_facts does.

What the sandbox shows today (sandbox.py:756-765):

- It covers /sys/class, /sys/bus, and each /sys/devices entry but system. So none of the paths above under /sys/bus
  exist inside, and UMD would see no card.
- /sys/module, /sys/kernel, /proc (its own procfs, sandbox.py:20), and /sys/devices/system stay in view.

Can sys_dirs provide them? Yes, with two binds, both strictly below the /sys/bus cover that sys_cover accepts
(sandbox.py:1016-1031):

- the host's /sys/bus/pci/slots at /sys/bus/pci/slots (60 entries on alpha01);
- the card's real PCI directory at /sys/bus/pci/devices/<bdf>. For card 0 that is
  /sys/devices/pci0000:e2/0000:e2:01.1/0000:e3:00.0/0000:e4:00.0/0000:e5:00.0/0000:e6:01.0/0000:e8:00.0 (probe 32e2).
  The /sys/bus cover's 8 inodes hold the five it needs: the cover's root, pci, slots, devices, and <bdf>.
- Then hwloc lists one PCI device, card <n>. Its local_cpus is in that directory.

Known limit (reading): the iommu_group link inside the card directory is relative and climbs eight levels
(`../../../../../../../../kernel/iommu_groups/135`, probe 32e2). Inside the bind it climbs past the root, so
iommu_group/type cannot be read there, and UMD reads "no IOMMU".

- On alpha01 that is also the host's answer: the domain type is "identity" (probe 59e3), and only DMA counts
  (umd pcie/pci_device.cpp:102-108).
- On a host with a DMA-mode IOMMU the two would differ, so the executor should refuse a card whose host
  iommu_group/type starts with DMA.

### 3. Hugepages

- Path and names (reading):
  - The directory is fixed: /dev/hugepages-1G (umd hugepage.cpp:28).
  - UMD takes it only from a /proc/mounts line whose source is nodev or hugetlbfs, whose mount point is exactly
    /dev/hugepages-1G, whose type is hugetlbfs, and whose pagesize option equals 1 GiB (:109-141). Otherwise it warns
    and gets no directory (:142-148).
  - The host's line matches: `hugetlbfs /dev/hugepages-1G hugetlbfs rw,nosuid,nodev,relatime,mode=777,pagesize=1024M
    0 0` (probe 7aeb).
  - File names (:151-172): device 0 channel 0 is `tenstorrent`. Any other device or channel adds the prefixes
    `device_<n>_` and `channel_<c>_`, as in `device_0_channel_1_tenstorrent`, `device_3_tenstorrent`, and
    `device_3_channel_1_tenstorrent`.
  - UMD sets umask 0 and opens O_RDWR|O_CREAT|O_CLOEXEC with mode 0666 (:186-189). It unlinks and retries only on
    EACCES (:190-198).
  - Each channel is mapped as one 1 GiB MAP_SHARED|MAP_POPULATE mapping and its descriptor closed (umd
    silicon_sysmem_manager.cpp:133-155). It is then pinned and mapped to the NOC through the driver (:197-258), since
    the 2.8.0 driver is newer than 2.0 (umd pcie/pci_device.cpp:618-668).
- How many (reading):
  - tt-metal asks for min(4, chips per MMIO chip) channels (metal tt_cluster.cpp:44, :375-388), which is 2 for an
    n300.
  - UMD then grants min(asked, max(1, nr_hugepages / Tenstorrent devices hwloc found)) (umd hugepage.cpp:47-68).
    With hwloc finding no device it grants 0 channels with a warning (:56-63).
  - So each n300 run uses two 1 GiB pages. With IOMMU off, the identity domain here, hugepages are the only sysmem
    path (umd silicon_sysmem_manager.cpp:55-61, :279-281).
- Persistence:
  - [MEASURED 2026-10-23, probe 7aeb, alpha01] After the five smoke runs, which were tracked references on card 0,
    the files `tenstorrent` and `device_0_channel_1_tenstorrent` (1073741824 bytes each, joseph_ufl, 0666, dated
    21:18, the first silicon run) remain in /dev/hugepages-1G, and the 1 GiB pool reads nr_hugepages 32 and
    free_hugepages 30. So the two files outlived every run and still hold two pages.
  - (Reading) UMD never unlinks them on its unmap path (umd silicon_sysmem_manager.cpp:63-91). It reopens them by
    name without truncation, so later runs on the same card reuse the same files, which fits two files after five
    runs.
  - (Reading) A hugetlbfs page fault first looks the page up in the file's page cache (linux mm/hugetlb.c:5879) and
    allocates and zeroes a new page only when there is none (:5913, :5933, :5938). So a reused file keeps the last
    run's bytes, and only a new file starts zeroed.
- Pool: nr_overcommit_hugepages is 0 (probe 59e3), so the 1 GiB pool is fixed at 32 pages, and libvirt keeps a
  directory in the same mount (probe 7aeb).
- Rootless private mount: not possible.
  - The sandbox is rootless. It runs `unshare -rinmpfu --mount-proc --kill-child` under `systemd-run --user --scope`
    (sandbox.py:16-22, :1299-1302), and -r maps the caller's uid to root in a new user namespace. No helper runs as
    root (bible, Sandbox; OQ-011).
  - hugetlbfs declares no FS_USERNS_MOUNT (linux fs/hugetlbfs/inode.c:1587-1592), unlike tmpfs (linux
    mm/shmem.c:4589, :4591). So mounting it needs CAP_SYS_ADMIN in the initial user namespace (linux
    fs/super.c:735-741).
  - (Reading; not tried on alpha01, whose kernel carries vendor patches.)
- The sandbox's own limits do not bound hugepages (reading):
  - Mapping a hugetlbfs file reserves its pages and grows the file without an RLIMIT_FSIZE check (linux
    fs/hugetlbfs/inode.c:167-175; only fallocate checks it, :844). So the sandbox's `prlimit --fsize` does not cap
    them.
  - The cgroup2 mount has no memory_hugetlb_accounting option, and the user's delegated controllers are only memory
    and pids (probe 59e3). So hugetlb pages count toward neither MemoryMax nor any limit the sandbox can set.

### 4. Other host state

- /dev/shm locks:
  - UMD's cross-process locks are robust pthread mutexes in POSIX shared memory named TT_UMD_LOCK.<name> (umd
    utils/robust_mutex.cpp:51). They are created 0666 under umask 0 (:50, :249-264) and never unlinked: the only
    "unlink" text in the file is a comment on an include (:10).
  - The names are <TYPE>, or <TYPE>_<n>_PCIe for a device (umd utils/lock_manager.cpp:15-23, :34-37).
  - LocalChip makes REMOTE_ARC_MSG, MEM_BARRIER, and CHIP_IN_USE for its PCI device (umd chip/local_chip.cpp:134-152)
    and holds CHIP_IN_USE while the chip is started (:187).
  - [MEASURED 2026-10-23, probe 7aeb] Seven such files remain in the host's /dev/shm from the smoke runs.
  - The sandbox gives each run a private /dev/shm (sandbox.py:719-721). So a sandboxed run and any other process on
    the same card cannot see each other's UMD locks: UMD's own per-card exclusion stops at the sandbox boundary.
  - The KMD has a per-device lock ioctl (kmd chardev.c:270-329) that UMD at the pin does not use.
  - tt-metal itself uses no shm_open, boost::interprocess, /dev/shm, or flock (probe 45d6).
- /tmp:
  - UMD's warm-reset listener directory /tmp/tt_umd_listeners (umd warm_reset.hpp:63) is created only by
    Monitor::start_monitoring (umd warm_reset.cpp:455-470), and nothing in tt_metal calls it (probe 45d6).
  - tt-metal's default JIT cache is /tmp/tt-metal-cache, used only when TT_METAL_CACHE and HOME are both unset (metal
    jit_build/build.cpp:92; plans/spikes/p4-ttsim-runtime.md). The executor sets TT_METAL_CACHE.
  - /tmp/hashlog.txt is compiled only under GENERATE_HASH_LOG (metal impl/program/program.cpp:176-177, probe 4ad5).
  - None of the three existed on the host after the smoke runs (probe 7aeb).
- Environment variables:
  - UMD reads TT_VISIBLE_DEVICES (umd pcie/pci_device.cpp:256; common/utils.hpp:76) and
    TT_BACKEND_CPUSET_ALLOCATOR_DEBUG, _ENABLE, and _SUPPORT_ANY_CPU (umd cpuset_lib.cpp:40, :43, :235). Its
    simulator variables do not apply to silicon.
  - tt-metal's list is in plans/spikes/p4-ttsim-runtime.md. TT_METAL_VISIBLE_DEVICES is parsed (metal
    rtoptions.cpp:414) but read nowhere (probe 45d6), and tt_metal never names TT_VISIBLE_DEVICES (probe c1f7).
- Writes into TT_METAL_RUNTIME_ROOT:
  - (Reading) The fabric control plane writes generated/fabric/physical_chip_mesh_coordinate_mapping_<r>_of_<w>.yaml
    and asic_to_fabric_node_mapping_rank_<r>_of_<w>.yaml under the runtime root. Each write is wrapped so that a
    failure only logs a warning (metal fabric/control_plane.cpp:519-540, printed by probe cf07).
  - [MEASURED 2026-10-23, probe cf07] The smoke runs left both files in the pinned tree.
  - In the sandbox the tree is the read-only toolchains bind, so the writes fail and warn (reading; not seen on
    silicon).
- JIT cache and logs: TT_METAL_CACHE and TT_METAL_LOGS_PATH work as for ttsim (metal rtoptions.cpp:367-369, :385).
  The logs default is the working directory (metal rtoptions.hpp:142).
- Resets: tt_metal never calls a device reset or warm reset (probe c1f7). UMD issues the reset ioctl only from its
  warm-reset path (umd pcie/pci_device.cpp:204-237, umd warm_reset.cpp).
- The driver's grant to an open descriptor (kmd 2.8.0, reading):
  - No ioctl checks a capability. In the files read, the only capable() is in the debugfs mappings listing (kmd
    enumerate.c:91).
  - Any holder of the node may reset the card, its ASIC, or its PCIe link (kmd chardev.c:191-262).
  - It may also pin pages (kmd memory.c:532-596, with no RLIMIT_MEMLOCK accounting in memory.c) and map the BARs.
  - TENSTORRENT_IOCTL_MAP_PEER_BAR needs another card's descriptor.
  - Closing the descriptor frees its pinned memory, TLBs, and locks, and re-aggregates the card's power state (kmd
    chardev.c:676-720). Nothing in the driver resets the chip's cores when a process dies.
  - The card's sysfs attributes are read-only (kmd wormhole.c:390-402; probe 59e3).

### 5. What the sandbox must change, and the risk of each change

1. Bind the chosen card's node, /dev/tenstorrent/<n>, at its own path in the private /dev.
   - Mechanism: none new. DEVICE_NODES_STEP already makes the parent directory and binds a node (sandbox.py:831-846),
     and the names pass _DEVICE_SEGMENT and RESERVED_DEV_NAMES (sandbox.py:428-432).
   - Risk: the program gets the driver's whole interface to that card (Finding 4): any register or NOC address
     through the BARs (the ARC included), card and PCIe-link resets, and DMA. The sandbox isolates the host, not the
     card. A read-only mount does not stop ioctl or mmap on a device node (sandbox.py:326-332).
   - A buggy or hostile program can leave the card hung or in need of a reset. Through the ARC it might change more;
     that is not read here.
   - The seccomp filter could refuse the reset ioctl (_IO(0xFA, 6), umd pcie/ioctl.h:24) for these runs, but not
     MMIO. Other cards' nodes stay absent, so the peer-BAR ioctl has no peer to name.
2. Bind /sys/bus/pci/slots and the card's PCI directory, both read-only, below the /sys/bus cover.
   - Mechanism: none new. These are DEVICE_SYS_STEP binds (sandbox.py:866-880).
   - Risk: low. The program reads the slot list (the PCI address list is already readable in /proc/bus/pci,
     sandbox.py:286-290), the card's first 64 bytes of config space, and its read-only telemetry. A read-only bind
     refuses writes even to writable attributes.
3. Add a writable hugetlbfs mount at exactly /dev/hugepages-1G inside the private /dev.
   - New code:
     - a directory bind (DEVICE_NODES_STEP touches a file and binds onto it, sandbox.py:842-843, so a directory needs
       mkdir);
     - making that one bind writable after the setup's recursive read-only step;
     - an exception in MOUNT_CHECK (sandbox.py:646-666), which today refuses every writable mount not named lassi-*.
   - The remount: the host mount is nosuid,nodev. A user namespace's copy of a mount tree locks those flags (linux
     fs/namespace.c:3780, :2150-2170), and a bind remount that drops them fails (:2744-2762, :2818-2830).
     - So the remount must keep nosuid and nodev, or must clear only the read-only flag with mount_setattr, as
       READONLY_PROGRAM sets it.
     - Whether libmount 2.37.2's `remount,bind,rw` keeps them by itself is unmeasured. The copy-back's remount
       (sandbox.py:802-803) remounts the workdir's bind, which may not carry those flags.
   - The exception should trust exactly one mount: type hugetlbfs, mount point /dev/hugepages-1G, and the expected
     root in /proc/self/mountinfo's fourth field. The program holds no capability, so it cannot add another mount.
   - What the bind exposes depends on its source (OQ-072, part 1):
     - (a) The host's whole /dev/hugepages-1G. The directory is mode 777 without the sticky bit (probe 7aeb), so the
       program can read, overwrite, or delete any file there, including other cards' and other users' 0666 sysmem
       files. It can see libvirt's directory and take every free 1 GiB page. Whatever it leaves in its own files,
       the next run on that card maps (Finding 3). That run could be a reference's or another candidate's, so data
       could pass between trials.
     - (b) A per-card directory that the executor makes inside the host mount (for example lassi-card-<n>, mode 0700)
       and empties before and after every run, bound at /dev/hugepages-1G.
       - Reading, unmeasured: /proc/mounts prints a bind's source, point, type, and the superblock's options, not
         its root, so the line still matches UMD's pattern.
       - The program sees and writes only its own files.
       - Each run starts from new zeroed pages, and nothing passes to the next run.
       - It can still create and map files to take the free 1 GiB pages (30 now) for as long as it runs, since
         nothing in the sandbox caps them (Finding 3). Any other user of 1 GiB pages, such as a libvirt guest, could
         fail to get one meanwhile.
     - (c) As (b), on a hugetlbfs mount for LASSI that the owner adds once as root, with size=2G and the user's
       uid, gid, and mode 0700. This also caps a run at its card's two pages, but the host keeps a new mount.
     - (d) No hugepages. Then UMD maps no host-memory channel, and the remote link to the n300's second chip loses
       its sysmem path (umd chip/local_chip.cpp:58-61). Whether tt-metal runs at all then is unknown. Not
       recommended.
4. Program environment.
   - The ttsim row's variables without TT_METAL_SIMULATOR (ttsim.py:258-279), as in the smoke run (smoke.py:85-93).
   - Optionally TT_VISIBLE_DEVICES=<n>. It would join ENVIRONMENT_NAMES as a visibility name that a spec may hold
     only with an exposure, as GPU_VISIBILITY_NAMES does (sandbox.py:404, :1152-1157).
   - Risk: none beyond one more allowlisted name. With only node <n> bound it costs one extra open of that node per
     enumeration (Finding 1).
5. /dev/shm stays private: no change.
   - Risk: UMD's locks do not span the boundary (Finding 4).
   - So the executor needs its own cross-process lock for LASSI runs.
   - Nothing excludes a non-LASSI process on the same card (OQ-072, part 5).
6. The workdir and the runtime root: no change. The generated/fabric writes fail into a warning (Finding 4).
7. Hangs and kills: no sandbox change. The innermost timeout kills the program, and the driver frees what the
   descriptor held, but nothing resets the chip's cores (Finding 4).
   - Whether the next run's start, which takes the cores through reset and loads firmware, recovers a card left
     mid-kernel is unmeasured.
   - The same goes for one whose ethernet link to the second chip was cut mid-transfer.
   - The ttsim executor's automatic Watcher rerun of a hang (ttsim.py:673-697) would be a second placement on a card
     in an unknown state.

### 6. Proposed design

A registered `Executor` "tt_silicon" in lassi/executors/tt_silicon.py, built like ttsim (lassi/executors/ttsim.py)
and gpu (lassi/executors/gpu.py). It opens no device itself: like the gpu executor, it reads only file metadata and
procfs or sysfs text (OQ-002).

- Capabilities: runs_code and sandboxed. Not simulator, so the stages never label its times as simulator time.
- Config keys:
  - card: the host's /dev/tenstorrent number, an int of at least 0, required, with no default. A bool, a str, a
    negative number, or a missing key is a ValueError naming the key.
  - arch: only wormhole_b0.
  - dispatch: only slow, the smoke's setting, until a task measures fast dispatch.
  - harness, as ttsim has it.
  - The keyword settings hidden_roots, toolchains, and sandbox, as ttsim has them.
- Construction refuses with DeviceUnavailable, before any directory exists, when:
  - /dev/tenstorrent/<card> is missing, not a character device, or not readable and writable (access(2), which
    opens nothing);
  - /sys/class/tenstorrent/tenstorrent!<card> is missing, or its tt_card_type is not n300;
  - the card's iommu_group/type starts with DMA (Finding 2);
  - /proc/mounts has no hugetlbfs line at /dev/hugepages-1G with pagesize 1024M;
  - fewer than two 1 GiB pages are free;
  - the card's quarantine marker exists (below).
  It records the card's serial, card type, firmware versions, PCI address and real sysfs path, and the driver
  version, as smoke.py's card_facts does.
- run():
  1. It checks the pinned tt-metal tree as ttsim's check_install does, minus the ttsim library and descriptor.
  2. It takes the lock: fcntl.flock LOCK_EX|LOCK_NB on one file, $LASSI_SCRATCH/locks/tt-silicon.lock, held for
     the whole call (cleanup, sandboxed run, cleanup). It writes its pid into the file and refuses at once, naming
     that pid, when another process holds the lock. Under the scratch root, never /tmp (Agent Rule 7). The kernel
     drops the lock if the process dies.
  3. It empties the card's hugepage directory, then makes the SandboxSpec:
     - the ttsim executor's mounts;
     - an exposure of the node /dev/tenstorrent/<card> at its own path;
     - the sys binds of Finding 2;
     - the writable hugepage directory of OQ-072 part 1, bound at /dev/hugepages-1G;
     - the environment of section 5, item 4, with TT_METAL_CACHE and TT_METAL_LOGS_PATH under the workdir's
       reserved directory (for example @tt; no model path may start with "@").
  4. It runs the program once and empties the hugepage directory again, also after a failure or a hang.
  5. It writes the quarantine marker $LASSI_SCRATCH/locks/tt-silicon-card-<card>.quarantine after a hang or a death
     by signal.
  6. It releases the lock.
  It never retries and never reruns with Watcher (Agent Rule 9: no looped device opens).
- RunResult: as ttsim's without the simulator fields (sim_ub and sim_gap stay unset). Kernel JIT diagnostics are read
  as ttsim's read_jit_diagnostics reads them.
- device(): "Tenstorrent Wormhole n300 card <n> (serial ..., firmware bundle ..., tenstorrent driver ...) with
  tt-metal 5280a9cf: silicon", in printable ASCII.
- pins: tt-metal's.
- Runner: it refuses tt_silicon in any training method and any reward stage (Agent Rule 9; bible Safeguards). It
  also refuses tt_silicon unless the run carries the human-start confirmation that OQ-072 part 6 settles. Both
  refusals come before any directory exists.
- Sandbox changes (section 5):
  - a directory-bind field on DeviceExposure, with its own writable flag;
  - the bind of that directory in DEVICE_NODES_STEP;
  - clearing the read-only flag on that one mount;
  - MOUNT_CHECK's single trusted hugetlbfs mount, passed by point and root as positional arguments;
  - for OQ-072 part 4, option (a): a seccomp rule refusing ioctl number 0xFA06, present only in tt_silicon runs.
  A spec without an exposure keeps SETUP_SCRIPT byte for byte, so the pinned sha256 test still holds.

Tests (test-writer list; host needs in brackets):

1. [Windows] Registration and capabilities: the executor is "tt_silicon", with exactly runs_code and sandboxed, and
   is not simulator.
2. [Windows] Config: card required with no default; bool, str, and negative card refused; arch and dispatch accept
   only their one value; unknown keys refused, each message naming the key.
3. [Windows] Construction under a fake host root, as the gpu tests build one. Each refusal of the construction list
   above, raised before any directory exists. Nothing under /dev is opened; a recording opener and access stub
   check it.
4. [Windows] Exposure for card <n>:
   - exactly one node, /dev/tenstorrent/<n> at its own path;
   - exactly the two sys binds (/sys/bus/pci/slots, and the card's real directory at /sys/bus/pci/devices/<bdf>);
   - the hugepage directory at /dev/hugepages-1G;
   - no other card's node or directory, and no by-id link.
5. [Windows] Environment: exactly the ttsim row's names minus TT_METAL_SIMULATOR, plus TT_VISIBLE_DEVICES=<n> if
   OQ-072 keeps it; no HOME. SandboxSpec refuses TT_VISIBLE_DEVICES without an exposure.
6. [Windows] Lock:
   - a second process's run refuses at once and names the holder;
   - the lock is released after a run that raised;
   - the lock file lies under $LASSI_SCRATCH, never /tmp.
7. [Windows] Hugepage directory:
   - every entry is removed before and after a run, the failed and hung cases included;
   - a symbolic link is removed without being followed;
   - nothing outside the directory is touched.
8. [Windows] Quarantine: a hang or a status of 128 or more writes the marker; a later construction for that card
   refuses until the marker is gone; there is no Watcher rerun.
9. [Windows] Runner: a training method or reward stage with tt_silicon is refused, and so is a run without the
   human-start confirmation, all before any directory exists.
10. [Windows] device() names the card, serial, firmware, driver, and tt-metal pin and says silicon, and the pins
    reach provenance.json.
11. [the local distro] Sandbox script text:
    - SETUP_SCRIPT's sha256 is unchanged;
    - MOUNT_CHECK passes exactly one writable hugetlbfs mount at /dev/hugepages-1G with the expected root, and fails
      a wrong point, a wrong root, or a second one (synthetic mountinfo lines through awk);
    - the seccomp rule refuses ioctl 0xFA06 and allows others (a stand-in descriptor, no device).
12. [alpha01, remote, needs the owner] Stand-in exposure, with no device and no hugepage mapped:
    - a tmpfs directory as the directory bind, and the host's /dev/zero as a stand-in node at
      /dev/lassi-standin/tt;
    - the program sees exactly those;
    - MOUNT_CHECK passes;
    - the host is unchanged afterwards.
13. [alpha01, remote, needs the owner; makes one directory in /dev/hugepages-1G and maps no page] Inside the sandbox:
    - /proc/mounts has a line at /dev/hugepages-1G that UMD's pattern accepts;
    - the mount is writable and keeps nosuid,nodev;
    - a file created there appears in the host directory, and no other host file is visible.
    The same test records that `unshare -rm` cannot mount hugetlbfs (EPERM), the rootless limit of Finding 3.
14. [alpha01 silicon, started by a human in an owner window, one card] The loopback reference through the executor:
    - it passes binary_io as in the smoke;
    - the hugepage directory is empty afterwards and free_hugepages is back to its value before the run;
    - an strace of the run, if the owner allows one, shows opens of /dev/tenstorrent/<n> only.
    Then the other four Tier A references, one at a time, compared with results/tt-silicon-smoke-20261023.
15. [alpha01 silicon, owner's explicit approval only] A seeded hang:
    - the timeout ends it and the quarantine marker appears;
    - a human checks whether the card needs a reset before the marker is cleared.

Bible rows to change:

- Execution Backends: the tt_silicon row, now (proposal below).
- Execution Backends: its opening paragraph still says the cards were removed on 2026-10-12. OQ-070's answer lists
  this paragraph as still to apply, together with Environment State's Tenstorrent row.
- Sandbox: a "tt_silicon program runs" bullet [DESIGN], written from the code when the executor is built, as the gpu
  bullet was.

### 7. Open questions for the owner

Drafted as OQ-072 below:

- the hugepage exposure;
- the lock's scope;
- the policy after a hang;
- the device-level risk of model code;
- other users of the cards;
- how a human starts a silicon run;
- the card key;
- the smoke's leftover hugepage files;
- the silicon environment.

## Proposed bible edit and Decision Log entry

Facts only: the choices stay [OPEN] and point to OQ-072. In docs/BIBLE.md, Execution Backends, replace the tt_silicon
row:

```
| tt_silicon | Wormhole n300 | One placement at a time; human-started evaluation only | Not built; the units were back on alpha01 on 2026-10-10 and were removed again on 2026-10-12 (Environment State) |
```

with:

```
| tt_silicon | TT-Metalium on one Wormhole n300 card | One placement at a time; human-started evaluation only; one card per run, named by the recipe with no default. What a sandboxed run needs, read at the pins (UMD 0450f1be, tt-metal 5280a9cf, tt-kmd 2.8.0) and checked read-only on alpha01 (plans/spikes/tt-silicon-executor.md): only that card's `/dev/tenstorrent/<n>`, bound at its own path, since UMD opens every numbered node it lists there whether `TT_VISIBLE_DEVICES` is set or not; `/sys/bus/pci/slots` and the card's PCI sysfs directory, read-only; a writable hugetlbfs mount at `/dev/hugepages-1G` that `/proc/mounts` shows, where UMD creates or reuses two 1 GiB files per n300 card that outlive the process and keep their bytes; and a lock of the executor's own, since UMD's per-card locks live in `/dev/shm`, which the sandbox makes private. The rootless sandbox cannot mount a private hugetlbfs | Not built. Eight n300 cards are on alpha01 and the gate's tt_silicon class is enabled (OQ-070). The five Tier A references ran once each on card 0, outside the sandbox, and passed binary_io [MEASURED 2026-10-23: results/tt-silicon-smoke-20261023, commit 982e71b, tt-metal 5280a9cf, driver 2.8.0, firmware bundle 19.5.0.0]; they are tracked upstream programs, never model output (Agent Rule 6). After them, two 1 GiB hugepage files still held 2 of the 32 pages [MEASURED 2026-10-23: rx 20261022-213300-exec-7aeb, alpha01]. [OPEN] the hugepage exposure, the lock's scope, the card key, the policy after a hang, the device-level risk of model code, and how a human starts a run (OQ-072) |
```

Decision Log entry: insert at the top, newest first, and raise the count sentence by one with a 2026-10-23 entry,
after any other 2026-10-23 entries OQ-070's application adds.

```
| 2026-10-23 | What a sandboxed tt_silicon executor needs (Execution Backends, the tt_silicon row; plans/spikes/tt-silicon-executor.md; OQ-070; OQ-072). At the pins (UMD 0450f1be, tt-metal 5280a9cf; tt-kmd 2.8.0) UMD lists /dev/tenstorrent and opens every numbered node there to read its PCI address, TT_VISIBLE_DEVICES set or not: at least twice per program start, and at least four times with TT_VISIBLE_DEVICES set. So binding only the chosen card's node, at its own path, is what confines a run to one card. It needs /sys/bus/pci/slots, the card's PCI attributes, the 1 GiB page count, and a hugetlbfs mount at /dev/hugepages-1G in /proc/mounts, where it creates or reuses two 1 GiB files per n300 card that outlive the process and keep their bytes; its per-card locks are files in /dev/shm, which the sandbox makes private. A per-run hugetlbfs mount is impossible in the rootless sandbox, since hugetlbfs cannot be mounted in a user namespace. The choices this leaves are OQ-072's | The 2026-10-23 smoke ran tracked references outside the sandbox; model output may run only through it (Agent Rule 6), and a run must neither open another card (Agent Rule 9) nor pass data to the next run. Read-only probes after the smoke found its two hugepage files still holding 2 of the 32 1 GiB pages and seven UMD lock files in /dev/shm [MEASURED 2026-10-23: rx 20261022-213300-exec-7aeb, alpha01]; the rest is read from the pinned sources |
```

## Owner-queue draft

```
## OQ-072 The Sandboxed tt_silicon Executor: Hugepages, The Lock, Hangs, Device Risk, And Who Starts A Run
State: OPEN
Kind: decision
Blocks: the tt_silicon executor task (to be planned; OQ-070)
Evidence: plans/spikes/tt-silicon-executor.md; results/tt-silicon-smoke-20261023/summary.md; rx 20261022-213300-exec-7aeb (the smoke's two 1 GiB hugepage files and seven UMD lock files remain; 30 of 32 pages free); rx 20261022-214312-exec-59e3 (no hugetlb controller for the user's cgroup)
Question: Model-generated TT programs can reach a Wormhole only through a sandboxed executor (Agent Rule 6). Reading the pinned sources settled what such a run needs: the card's node, two read-only sysfs paths, and a writable hugetlbfs at /dev/hugepages-1G. Outside the sandbox, with TT_VISIBLE_DEVICES=0, UMD briefly opened all eight cards' nodes several times per program start (reading, not counted); inside it, with only one node bound, it can open no other card. Six choices remain, and three small ones. (1) Hugepages: where the writable /dev/hugepages-1G comes from. (2) The lock's scope. (3) After a hang or kill. (4) Accepting that model code controls the card: through its BARs and the 2.8.0 driver, which checks no capability, it can touch any register or reset the card or its PCIe link; the sandbox isolates the host, not the card. (5) Other users of the cards: the sandbox hides UMD's own /dev/shm locks, so a sandboxed run and any other process on the same card cannot see each other. (6) How a human starts a silicon run (Agent Rule 9). Small ones: (7) the card key, (8) the smoke's leftover hugepage files, (9) the silicon environment.
Options:
- (1a) Bind the host's whole /dev/hugepages-1G writable: simplest; the program can read, change, or delete any file there (mode 777, no sticky bit; other cards' and users' sysmem files are 0666; libvirt keeps a directory there), take every free 1 GiB page until cleanup, and leave bytes the next run on that card maps.
- (1b) The executor makes a private directory per card inside /dev/hugepages-1G and empties it before and after every run, bound at /dev/hugepages-1G: the program sees only its own files, and each run starts from zeroed pages; it can still take all free 1 GiB pages (30 now) while it runs, since the user's cgroup has no hugetlb controller; one directory is added to a shared host mount.
- (1c) As (1b), on a hugetlbfs mount you add once as root (for example one per card, size=2G, owned by joseph_ufl, mode 0700): also caps a run at its card's two pages; the host keeps a new mount.
- (1d) No hugepages: UMD maps no host memory, and whether tt-metal runs at all then is unknown.
- (2a) One lock for all cards: one placement at a time, as Agent Rule 9 says. (2b) A lock per card, so different cards can run at once: an Agent Rule 9 change.
- (3a) After a hang, a kill, or a crash, quarantine the card (refuse it until a human checks it and clears a marker) and never rerun with Watcher on silicon. (3b) Continue, relying on the next run's start to reset the cores; whether that recovers the card is unmeasured.
- (4a) Accept, with the seccomp filter refusing the driver's reset ioctl in tt_silicon runs (tt-metal never calls it) and (3a). (4b) Accept as is. (4c) Run only tracked references on silicon, never model output (no executor).
- (5) Confirm that nothing else uses the cards during a LASSI silicon window, or name the cards LASSI may use.
- (6a) The runner refuses tt_silicon unless its command line names the card in a flag a recipe cannot set, and unattended sessions never pass it; training methods and reward stages refuse tt_silicon outright. (6b) Another mechanism you name.
- (7) The card key: the /dev/tenstorrent number, which the smoke used and a reboot may change, with the serial recorded in every result; or the serial itself.
- (8) Remove /dev/hugepages-1G/tenstorrent and /dev/hugepages-1G/device_0_channel_1_tenstorrent, which return 2 pages to the pool (no run is using them), or keep them.
- (9) Keep the smoke's environment (the ttsim row's settings without TT_METAL_SIMULATOR, so slow dispatch and TT_METAL_DISABLE_SFPLOADMACRO=1) for the first executor, so its runs compare with the smoke, and measure fast dispatch and SFPLOADMACRO in a later task; or drop the simulator-only settings now.
Recommendation:
- (1c) if you will add the mount, else (1b): no data passes between runs, and no other card's or user's file is in view.
- (2a): the rule as written.
- (3a): repeated opens of a card in a bad state are what Agent Rule 9 guards against.
- (4a).
- (6a).
- (7) the number with the serial recorded.
- (8) remove them.
- (9) keep the smoke's environment.
Answer:
```

## Confidence

- High for Findings 1 to 4 as statements about the pinned sources. Every line was read at the pinned commits, and
  probe 73fc confirmed that the build host's UMD is that commit. High for the host facts the probes print.
- Medium for how the proposed exposure behaves inside the sandbox. Not measured:
  - that a bind of a hugetlbfs subdirectory satisfies UMD's /proc/mounts pattern;
  - that the bind can be made writable again while keeping its locked flags;
  - that hwloc counts the card from one bound PCI directory;
  - that tt-metal starts with that view.
  Tests 12 to 14 measure them.
- Medium for the driver: the source is tag ttkmd-2.8.0, and the installed module's build was not compared with it.
- Not known:
  - whether a killed silicon run leaves the card needing a reset;
  - whether the enumeration opens of other cards send them any message.

## Consequences for the plan

- The executor is a task of its own, with a sandbox sub-task for the directory bind, the writable hugetlbfs exception,
  and the optional seccomp rule.
  - The sandbox sub-task is testable locally and with stand-ins on alpha01.
  - The silicon qualification is a separate, human-started step.
  - Everything that touches the host's hugepages or a card waits on OQ-072 and the owner's approval of the probe or
    the run.
- The bible edit above is factual and can land with OQ-070's application.
- Until the executor passes its qualification, TT programs from a model run on ttsim only.
- A silicon run of a tracked reference should also go through the executor once it exists. Outside the sandbox, UMD
  opens every card's node (Finding 1), which the sandbox's single bound node prevents.
