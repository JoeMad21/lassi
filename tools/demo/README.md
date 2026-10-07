# LASSI demo scripts

The demo scripts that run on alpha01 from `/mnt/nvme10/joseph_ufl/bin`. Their source lives here; the copies in
`bin` are installed from the demo slot `/mnt/nvme10/joseph_ufl/lassi-wt/demo-live`, which runs the commit that
holds them. Every demo number is one sample of the stock demo model, never a performance figure, and every
Tenstorrent number comes from ttsim, a simulator: label it simulator, never silicon.

## The demo, in order

1. `lassi-showcase` (no model, no card; about 1 minute). The LASSI-DF banner and graphics setting, then
   `lassi score` on the September 24 run demo-rngd-cpu-1: the run metrics table with Wilson 95% intervals and
   the paper's recount, one score per trial, and the start of the review packet. `--plain` turns graphics off.
2. `lassi-live --both` (one RNGD card, about 10 to 20 minutes). It checks the cards (`furiosa-smi ps` and
   `furiosa-smi status`; only a card from npu1 to npu7 at 0.00 GiB, never npu0), serves
   furiosa-ai/Llama-3.1-8B-Instruct at v2026.2 on that card, and runs two segments against it with the
   LASSI-DF banner and the live inference table:
   - the CPU segment: the LASSI replication, CUDA -> OpenMP on matrix-rotate and entropy, every generated
     program compiled with nvc++ -mp=multicore and run in the sandbox on the CPU;
   - the Tenstorrent segment: tt-pairs-v0 Tier A item eltwise_binary, C++ -> TT and TT -> C++ with the
     tt-host-v0 prompts (the item's kernels shown read-only), every TT program built against tt-metal 5280a9cf
     and run on ttsim, one trial each with at most 3 corrections.
   It prints each segment's outcome table, then stops the server and checks the card back at 0.00 GiB.
   Ctrl-C stops the run and the server at any point.

## Options

- `lassi-live --mock --both` is the fallback when no card is free (on 2026-10-07 every card from npu1 to
  npu7 was held by another tenant). No card, no server, no inference: the mock model replies with each
  item's reference program, so the builds, the sandbox, ttsim, the oracle, and the live table run for
  real and every trial passes by construction. Say so: it shows the pipeline, never a model.
- `lassi-live` alone runs only the CPU segment, as before; `lassi-live --tt` only the Tenstorrent segment.
- `--apps a,b` chooses the CPU segment's apps (any of lassi-hecbench-10); a trial that hits the correction cap
  takes 5 to 8 minutes. `--tt-apps` chooses the Tier A items: loopback, eltwise_binary, and eltwise_sfpu are
  judged by max_abs; matmul_single_core and matmul_multi_core by pcc, in a run of their own.
- `--card npu3` picks a card; `--plain` turns graphics off; `--compile-only` makes the CPU segment compile only,
  both directions.
- `lassi-live --show <run id>` reprints a finished run's outcome table (no card, no server). With `--both` the
  run ids are `<id>-cpu` and `<id>-tt`.

## If something stops it

- No free card: another tenant holds them; `furiosa-smi status` shows which. Wait, or name a free one with
  `--card`.
- Port 8123 in use, or a server of this account still running: an earlier server has not exited; wait for it.
- The server is not ready after 300 s: read the serve log the script names under
  `/mnt/nvme10/joseph_ufl/tmp/`.
- A Tenstorrent trial that ends at the cap or with a simulator finding is a result, not a failure of the demo:
  the outcome table names its end reason.

## Installing a new version

From a clean commit, through the gate: `uv run tools/rx.py run --slot demo-live -- '<copy tools/demo/* into
bin>'` syncs the slot to that commit and installs the scripts; keep a dated copy of each script it replaces.
The scripts need the slot's generated upstream prompts (`tools/extract_lassi_assets.py`) and the fetched
tt-pairs-v0 sources (`tools/fetch_bench.py assets/bench/tt-pairs-v0.yaml`), both already in place on alpha01.
