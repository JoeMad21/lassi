# P0.10 verification: the sandbox on alpha01

Date 2026-09-23 (host clock UTC-07:00). rx id 20260923-073958-desktop-8r113ei-p0-core-6981, `uv run tools/rx.py run` from the clean commit f57c90a (the run printed the commit and 0 changed files). Device: none; no GPU or accelerator is involved. The sandboxed programs are tiny shell and python3 commands written by the tests, not generated code. No toolchain pin is involved. This run did not print tool versions; the same-day spike (plans/spikes/p0-sandbox.md) recorded kernel 6.6.29, util-linux 2.37.2, and systemd 249 on alpha01, and the coreutils version of `timeout` is not recorded. [MEASURED]

Command:

```
uv run tools/rx.py run --timeout 600 -- 'date -Is; git rev-parse HEAD; git status --short | wc -l; cat /proc/sys/kernel/core_pattern; cat /proc/sys/user/max_user_namespaces; uv sync --quiet && LASSI_REQUIRE_SANDBOX=1 uv run pytest -v -m remote tests/executors -p no:cacheprovider 2>&1 | grep -E "PASSED|FAILED|ERROR|passed|failed" | tail -30'
```

Output (pytest's progress percentages are trimmed from each test line):

```
2026-09-23T07:39:58-07:00
f57c90a07fd7785a69eb4aa70936df2591a5553b
0
|/lib/systemd/systemd-coredump %P %u %g %s %t 9223372036854775808 %h %d
12347780
tests/executors/test_sandbox_remote.py::test_network_connect_fails_inside PASSED
tests/executors/test_sandbox_remote.py::test_memory_hog_is_killed PASSED
tests/executors/test_sandbox_remote.py::test_cpu_time_hog_is_killed_before_the_wall_limit PASSED
tests/executors/test_sandbox_remote.py::test_sleep_past_wall_time_sets_hang PASSED
tests/executors/test_sandbox_remote.py::test_a_program_ignoring_sigterm_is_killed_after_the_grace PASSED
tests/executors/test_sandbox_remote.py::test_a_program_within_the_wall_limit_runs_to_its_end PASSED
tests/executors/test_sandbox_remote.py::test_harness_and_readonly_roots_refuse_writes_and_the_workdir_takes_them PASSED
tests/executors/test_sandbox_remote.py::test_the_harness_mount_alone_is_read_only PASSED
tests/executors/test_sandbox_remote.py::test_every_readonly_root_is_read_only_not_only_the_first PASSED
tests/executors/test_sandbox_remote.py::test_the_program_holds_no_capability_and_cannot_undo_a_mount PASSED
tests/executors/test_sandbox_remote.py::test_the_program_cannot_create_a_nested_user_namespace PASSED
tests/executors/test_sandbox_remote.py::test_the_program_runs_at_the_lowest_cpu_priority PASSED
tests/executors/test_sandbox_remote.py::test_the_root_filesystem_and_cgroups_are_read_only_and_run_is_private PASSED
tests/executors/test_sandbox_remote.py::test_the_run_cannot_rewrite_its_own_cgroup_limits PASSED
tests/executors/test_sandbox_remote.py::test_the_bus_sockets_are_hidden_inside PASSED
tests/executors/test_sandbox_remote.py::test_the_program_sees_only_the_allowed_environment PASSED
tests/executors/test_sandbox_remote.py::test_tmp_writes_stay_off_the_host[/tmp] PASSED
tests/executors/test_sandbox_remote.py::test_tmp_writes_stay_off_the_host[/var/tmp] PASSED
tests/executors/test_sandbox_remote.py::test_tmp_writes_stay_off_the_host[/dev/shm] PASSED
tests/executors/test_sandbox_remote.py::test_a_failed_setup_raises_and_never_runs_the_program PASSED
tests/executors/test_sandbox_remote.py::test_the_pid_namespace_hides_host_processes PASSED
tests/executors/test_sandbox_remote.py::test_exit_status_passes_through PASSED
tests/executors/test_sandbox_remote.py::test_native_executor_runs_a_shell_script_artifact PASSED
===================== 23 passed, 178 deselected in 20.20s ======================
```

The P0.10 acceptance clauses map to these tests: a network connect fails (test_network_connect_fails_inside), a memory hog is killed (test_memory_hog_is_killed), a sleep past wall time sets hang (test_sleep_past_wall_time_sets_hang), and a write to the harness mount fails (test_the_harness_mount_alone_is_read_only and the harness and roots test). The nested user-namespace test asserts that the host's own limit (12347780 above) is unchanged after the run.

Finding for task P0.16: kernel.core_pattern pipes crashes to /lib/systemd/systemd-coredump with a fixed core limit argument, so the sandbox's `prlimit --core=0` does not stop a crashing program from reaching the host core-dump handler. [MEASURED for the pattern; that the handler may then store a core on the root filesystem is an inference.]
