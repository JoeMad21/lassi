# Spike: CPU energy telemetry on alpha01 (task P9.2, the energy half)

- Task: P9.2 (plans/p9-ee.md). Bible: Component Interfaces (the energy profiler bullet: rapl), Host Facts, Agent
  Rules 1 and 17. Feeds P9.6 and P9.13 (the CPU port's energy source).
- Date: 2026-10-25, in the owner's window; the probe was approved in the working session (OQ-092's answer, option
  (a)). One read-only `rx exec` from the scratch root; nothing was written, built, or run. rx ids use alpha01's
  clock (UTC-7).
- Not covered here: P9.2's excess-files inventory, deferred with alpha01's inventories (OQ-055, decision 11 (a)).

## Command, rx 20261025-182150-exec-2692

```
echo "== powercap"; ls -l /sys/class/powercap/ 2>&1 | head -12; for z in /sys/class/powercap/intel-rapl*/; do [ -d "$z" ] || continue; echo "$z name=$(cat $z/name 2>&1) energy_uj_mode=$(stat -c %A $z/energy_uj 2>&1) read=$(cat $z/energy_uj 2>&1 | head -c 40) max=$(cat $z/max_energy_range_uj 2>&1)"; done; echo "== governor"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>&1; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>&1; uname -r
```

## Output (the read= fields cut where the shell printed cat's permission error)

```
== powercap
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl -> ../../devices/virtual/powercap/intel-rapl
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl:0 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:0
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl:0:0 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl:1 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:1
lrwxrwxrwx 1 root root 0 Oct 25 08:26 intel-rapl:1:0 -> ../../devices/virtual/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0
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

- alpha01's two AMD packages appear under the intel-rapl control type as zones whose name files read package-0 and
  package-1, each with one subzone named core: the layout P9.1's rapl backend reads (lassi/profilers/rapl.py).
- energy_uj is mode 0400 (-r--------, root only) in every zone, and this account's read failed: the rapl backend
  refuses when built here, naming the file, as P9.1 designed. max_energy_range_uj reads 65532610987 (microjoules)
  in every zone.
- CPU frequency: governor ondemand, driver acpi-cpufreq, kernel 6.6.29.
- So the CPU port's energy is available only if the owner grants this account read access to the package zones'
  energy_uj (a root-side change on alpha01, which agents never make); otherwise the port reports wall time labeled
  not energy (backend wall). OQ-097 asks.
- Not measured: the counter's update interval and its wrap time at alpha01's load.
