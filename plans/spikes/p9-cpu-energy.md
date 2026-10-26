# Spike: CPU energy telemetry on alpha01 (task P9.2, the energy half)

- Task: P9.2 (plans/p9-ee.md). Bible: Component Interfaces (the energy profiler bullet: rapl), Host Facts, Agent
  Rules 1 and 17. Feeds P9.6 and P9.13 (the CPU port's energy source).
- Date: 2026-10-25, in the owner's window; the probe was approved in the working session (OQ-092's answer, option
  (a)). One read-only `rx exec` from the scratch root; nothing was written, built, or run. rx ids use alpha01's
  clock (UTC-7).
- Not covered here: energy_uj's owner, perf_event_paranoid, /sys/bus/event_source/devices/power, and the
  package-to-CPU map (not in this probe; next owner window); the Linux source lines at alpha01's release that
  register AMD RAPL in powercap and set energy_uj's mode (source reading on Windows); the Host Facts verdict (after
  OQ-097); and P9.2's excess-files inventory, deferred with alpha01's inventories (OQ-055, decision 11 (a)).

## Command, rx 20261025-182150-exec-2692 (its powercap and governor part; the same exec then listed run records under lassi-runs for P17.27, the records read OQ-092's answer approved, left out here; its printed output went through a local tail -40 | cut -c1-240)

```
echo "== powercap"; ls -l /sys/class/powercap/ 2>&1 | head -12; for z in /sys/class/powercap/intel-rapl*/; do [ -d "$z" ] || continue; echo "$z name=$(cat $z/name 2>&1) energy_uj_mode=$(stat -c %A $z/energy_uj 2>&1) read=$(cat $z/energy_uj 2>&1 | head -c 40) max=$(cat $z/max_energy_range_uj 2>&1)"; done; echo "== governor"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>&1; uname -r
```

## Output (each read= field is cat's error message cut at 40 bytes by the command's head -c 40, before its reason; lines cut at 240 characters by the local cut; the records section left out)

```
== powercap
total 0
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl -> ../../devices/virtual/powercap/intel-rapl
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl:0 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:0
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl:0:0 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl:1 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:1
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl:1:0 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0
/sys/class/powercap/intel-rapl/ name=cat: /sys/class/powercap/intel-rapl//name: No such file or directory energy_uj_mode=stat: cannot statx '/sys/class/powercap/intel-rapl//energy_uj': No such file or directory read=cat: /sys/class/powercap
/sys/class/powercap/intel-rapl:0/ name=package-0 energy_uj_mode=-r-------- read=cat: '/sys/class/powercap/intel-rapl:0// max=65532610987
/sys/class/powercap/intel-rapl:0:0/ name=core energy_uj_mode=-r-------- read=cat: '/sys/class/powercap/intel-rapl:0:0 max=65532610987
/sys/class/powercap/intel-rapl:1/ name=package-1 energy_uj_mode=-r-------- read=cat: '/sys/class/powercap/intel-rapl:1// max=65532610987
/sys/class/powercap/intel-rapl:1:0/ name=core energy_uj_mode=-r-------- read=cat: '/sys/class/powercap/intel-rapl:1:0 max=65532610987
== governor
ondemand
acpi-cpufreq
6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+
```

## Findings [MEASURED, alpha01, rx 20261025-182150-exec-2692, 2026-10-25]

- alpha01's two packages (AMD EPYC per the bible; the vendor is not in this output) appear under the intel-rapl
  control type as zones whose name files read package-0 and package-1, each with one subzone named core: the layout
  P9.1's rapl backend reads (lassi/profilers/rapl.py). The control type's own directory holds no name or energy_uj
  file.
- energy_uj is mode 0400 (-r--------, owner read only; the owner was not printed) in every zone, and this account's
  read failed (cat printed an error whose reason head -c 40 cut). By lassi/profilers/rapl.py, not run here, the rapl
  backend would refuse when built as this account, naming the file. max_energy_range_uj reads 65532610987
  (microjoules) in every zone.
- cpu0's frequency governor is ondemand and its driver acpi-cpufreq (other CPUs not read); kernel release
  6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+.
- So the CPU port's energy is available only if the owner grants this account read access to the package zones'
  energy_uj (a root-side change on alpha01, which agents never make); otherwise the port reports wall time labeled
  not energy (backend wall). OQ-097 asks.
- Not measured: the counter's update interval and its wrap time at alpha01's load.
