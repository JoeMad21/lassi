# Spike P3 demo: serving an LLM on the RNGD cards for LASSI trials

- Status: exploratory. Every command below is a probe, not an acceptance run; no number here is a [MEASURED] arm result.
- Date: 2026-09-24 (workstation, UTC); host times are alpha01 `date -Is` output, 2026-09-23 22:28 to 22:44 at UTC-07:00. The demo is at 15:00 EDT on 2026-09-24, which is 12:00 host time.
- Local checkout: `c9771d3ca5e2e52189115cac82460bc613fb5696` on branch `p1-faithful`, clean when the three serve jobs started (rx printed no dirty-snapshot notice). The serve jobs ran no repository code; `rx exec` probes run from the scratch root without a checkout.
- Host: alpha01 (I/ONX), reached only through `uv run tools/rx.py` (doctor, exec, job start/tail/kill/status). On the Windows workstation every rx call ran with `PYTHONIOENCODING=utf-8`, because rx crashed with a cp1252 UnicodeEncodeError on a progress bar in a remote log (rx 20260923-222934-exec-6baa).

## Question

Can alpha01 serve an LLM on a Furiosa RNGD card through furiosa-llm's OpenAI-compatible server today, so LASSI translation trials (OpenMP <-> CUDA with a correction loop) can use it through the `openai_compat` backend?

Why it matters: the demo at 15:00 EDT needs a live model arm; P3 (RNGD Serving) and every later RNGD phase (P6, P7) depend on a working serve procedure; the bible's Host Facts list the SDK venv as "not re-checked".

## Classification

Factual: answered by read-only inspection and by starting, querying, and stopping a server through the gate. The gate's rngd device class was enabled by J (`rx doctor` at 2026-09-24T05:28:21Z: `"devices_enabled": {"rngd": true, "tt_silicon": false, "rocm_gpu": false, "nvidia_gpu": false}`, `"running_jobs": []`, scratch_free_gb 427.5).

## Commands and Outputs

### 1. Cards, driver, firmware (rx 20260923-222834-exec-7ff6, 22:28:34)

`furiosa-smi info; furiosa-smi ps; furiosa-smi version; dpkg -l | grep -i furiosa; du -sh /mnt/nvme10/joseph_ufl`

```
npu0..npu7: rngd, firmware 2026.3.0, 2d3f72a, 28-31 C, 37-39 W (8 rows)
furiosa-smi ps: header only, no rows
- furiosa-smi 2026.1.1
- device driver 2026.3.1, e28c5c0
furiosa-compiler 2025.3.0-3 | furiosa-driver-rngd 2026.3.1 | furiosa-firmware-image-rngd 2026.3.0
furiosa-firmware-tools-rngd 2026.3.1-3 | furiosa-libsmi 2026.1.1-3 | furiosa-smi 2026.1.1-3 | furiosa-toolkit-rngd 2026.2.1-3
94G /mnt/nvme10/joseph_ufl
```

`/var/log/dpkg.log` (rx 20260923-223005-exec-fc66): `2026-08-02 03:14:01 upgrade furiosa-driver-rngd:all 2026.3.0 2026.3.1` (with firmware-tools 2026.3.1, libsmi 2026.1.1, toolkit 2026.2.1 in the same minute).

### 2. Occupancy: `furiosa-smi ps` does not show other users' processes

rx 20260923-223122-exec-8943 (22:31:22): `furiosa-smi ps` had no rows, yet `pgrep -af furiosa-llm` showed another user's server:

```
1818323 /mnt/nvme10/jun_ufl/f-inf/furiosa-venv/bin/python .../furiosa-llm serve furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8
  --fxb .../qwen3-coder-30b-a3b-fp8-6317c7f93b-6f1dded303-2606290751.fxb --revision main --host 0.0.0.0 --port 8300
  --devices npu:4,npu:5,npu:6,npu:7 -pp 1 -dp 1 --no-enable-prefix-caching ...
```

rx 20260923-223138-exec-6cb1 and 20260923-223152-exec-c21c (22:31:52), `furiosa-smi status`, memory column:

```
npu0 0.00/47.50 GiB | npu1 0.00/47.50 GiB | npu2 0.00/47.50 GiB | npu3 0.00/47.50 GiB
npu4 45.93/47.50 GiB | npu5 45.93/47.50 GiB | npu6 45.93/47.50 GiB | npu7 45.93/47.50 GiB
ps: jun_ufl 1818323 started Wed Sep 23 18:29:37 2026; listeners on 0.0.0.0:8300 and 0.0.0.0/[::]:8000
```

While our server ran, `furiosa-smi ps` listed our own process (`3191712 | npu1:[0, 7] | .../furiosa-llm serve ...`, rx 20260923-223305-exec-7e36) but still not jun_ufl's. So `furiosa-smi ps` lists only the caller's processes; `furiosa-smi status` memory is the check that shows every tenant. Free cards at 22:31: npu1, npu2, npu3 (npu0 reserved by Agent Rule 8; npu4-npu7 held by jun_ufl). npu1 was chosen.

### 3. The existing venv (rx 20260923-222845-exec-4c34)

```
/mnt/nvme10/joseph_ufl/furiosa-venv: pyvenv home /usr/bin, Python 3.10.12, 8.0G, created 2026-06-01
furiosa-llm 2026.2.1 | furiosa-native-runtime 2026.2.1 | furiosa-native-llm-common 2026.2.1 | furiosa-torch-ext 2026.2.1
furiosa-models 2026.2.0 | furiosa-torch 2026.2.0 | furiosa-smi-py 2026.1.2 | torch 2.10.0 | transformers 5.1.0 | openai 2.38.0
```

Compatibility with driver 2026.3.1 and firmware 2026.3.0: the prior log `/mnt/nvme10/joseph_ufl/logs/serve-8300.log` (2026-08-26, after the 2026-08-02 driver upgrade) shows this venv loading the artifact (`Parallelism Config: tp=8, pp=1, dp=2`, Uvicorn up, two `POST /v1/chat/completions` 200); section 5 confirms it again.

### 4. Models on disk (rx 20260923-222853-exec-417d, -222902-exec-fdb5, -222911-exec-832b, -222934-exec-6baa)

The gate sets `HF_HOME=/mnt/nvme10/joseph_ufl/hf`, which does not exist. The populated cache is `/mnt/nvme10/joseph_ufl/.cache/huggingface/hub`:

| Repo | Size | Ref -> commit | Kind |
| --- | --- | --- | --- |
| furiosa-ai/Llama-3.1-8B-Instruct | 16G | v2026.2 -> 231d94fbc03cdd66aaeb2411697064a45f008ec7 | Pre-compiled artifact (artifact.json, binary_bundle.zip, BF16 params) |
| Qwen/Qwen2.5-0.5B | 954M | main -> 060db649 | Raw HF weights |
| Qwen/Qwen2.5-1.5B | 2.9G | main -> 8faed761 | Raw HF weights |
| Qwen/Qwen2.5-7B | 28K | main -> d1497293 | Metadata only |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 2.9G | main -> 2e1fd397 | Raw HF weights |

- The v2026.2 commit 231d94f matches the tag on the Hub (https://huggingface.co/api/models/furiosa-ai/Llama-3.1-8B-Instruct/refs, fetched 2026-09-24).
- Its artifact.json: furiosa_llm_version b62dbc1, compiler d19a92a2f2, schema 3.0, bf16 weights, activations, and KV cache, tensor_parallel_size 8, pipeline_parallel_size 1, largest attention bucket 131072, batch buckets 1 to 256.
- `/mnt/nvme10/joseph_ufl/built-models`: `qwen2.5-0.5b` and `qwen2.5-0.5b-noop`, 972M each, artifacts built from the base (not instruct) Qwen2.5-0.5B. `build-qwen-0.5b.log` (2026-06-01) records that build with the 2026.2.1 venv (`Filtered bucket preset by max_model_len=32768`, `The computed bucket limits are 4096`, ends `Artifact Build Completed`).
- `/mnt/nvme10/joseph_ufl/.cache/furiosa/llm/compiled_graphs`: 4.0G of .edf graphs from earlier builds.

The only code-capable instruct model with a pre-compiled artifact for furiosa-llm 2026.2.1 on disk is furiosa-ai/Llama-3.1-8B-Instruct@v2026.2.

### 5. Serve, query, stop (three jobs)

`furiosa-llm serve --help` (rx 20260923-223038-exec-1eb9): `--devices` takes `npu:X` or `npu:X:Y-Z`; without it "all available unoccupied devices will be used", so it must always be given. `--max-model-len` exists; the served default comes from the artifact.

Job 1, rx 20260923-223207-rngd-serve-3afd (slot desktop-8r113ei-p1-faithful, started 22:32:07):

```
date -Is; furiosa-smi ps; furiosa-smi status | grep -a -E "\| rngd"
if ! furiosa-smi status | grep -a -E "\| npu1 " | grep -q " 0.00/"; then echo "ABORT: npu1 not free"; exit 2; fi
source /mnt/nvme10/joseph_ufl/furiosa-venv/bin/activate
export HF_HOME=/mnt/nvme10/joseph_ufl/.cache/huggingface HF_HUB_OFFLINE=1
exec furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct --revision v2026.2 --devices npu:1 --host 127.0.0.1 --port 8123 --no-enable-prefix-caching
```

Serve log (ANSI codes stripped):

```
22:32:14 Loading artifact from path: /mnt/nvme10/joseph_ufl/.cache/huggingface/hub/models--furiosa-ai--Llama-3.1-8B-Instruct/snapshots/231d94f...
22:32:37 Parallelism Config: tp=8, pp=1, dp=1
22:32:38 PP device#0 allocation plan: Binary=283.8 MiB, Model weights=15.0 GiB, Reserved IO memory=5.0 GiB
22:32:41 PP device#0 KV cache=27.2 GiB
22:32:41 Computed bucket limits: max_executable_len=131072
22:32:43 DP entry DpId(0) device [npu1pe0-3, npu1pe4-7]
22:32:43 max_kv_len=222844 (from KV cache blocks across 1 DP device(s))
INFO: Uvicorn running on http://127.0.0.1:8123
```

Ready about 36 s after the job started. No download took place (HF_HUB_OFFLINE=1).

`curl localhost:8123/v1/models` (rx 20260923-223305-exec-7e36, 22:33:05), before any request:

```
{"object":"list","data":[{"id":"furiosa-ai/Llama-3.1-8B-Instruct","created":1790227963,"object":"model","owned_by":"furiosa-ai",
 "artifact_id":"d6ae6a43-6ce0-4864-aaca-eeac4340234c","max_prompt_len":131072,"max_context_len":131072}]}  http=200
npu1 42.50/47.50 GiB while serving
```

First chat completion (rx 20260923-223315-exec-c962, 22:33:15), non-streaming, temperature 0, max_tokens 256. Prompt: "Translate this OpenMP C loop to CUDA: a __global__ kernel plus the host launch code. Reply with code only." followed by `#pragma omp parallel for` over `y[i] = a * x[i] + y[i];`.

```
http=200 ttfb=3.943251s total=3.943437s
usage: prompt_tokens 95, completion_tokens 256, total 351; finish_reason length
__global__ void kernel(float *x, float *y, float a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * x[idx] + y[idx];
  }
}
int main() { ... cudaMalloc ... cudaMemcpy ...   [cut at 256 tokens]
```

Second request, same prompt with max_tokens 1024 (rx 20260923-223330-exec-168c): `http=200 total=6.149965s`, 95 prompt and 398 completion tokens, finish_reason stop, a complete program ending in `cudaFree` and `free`.

Stop with `rx job kill 20260923-223207-rngd-serve-3afd`: the gate reported `"state": "killed"`, but the server kept running and answering (rx 20260923-223352-exec-8f0b, 22:33:52 to 22:35:54: pid 3191712 alive, STAT SNsl, port 8123 http=200, npu1 42.56 GiB). Cause, from `tools/server/gate.py`: `jobrun` starts the command with `start_new_session=True` (`stream_process` via `new_group()`), and `verb_job_kill` calls `kill_group(runner_pid)`, which signals only the runner's own process group. The command's session survives and is reparented to init, and the job's timeout dies with the runner. The server was then stopped by its owner process group: `kill -TERM -- -3191712` after checking the user is joseph_ufl (rx 20260923-223613-exec-88d7), which ended the group within 5 s; rx 20260923-223626-exec-3e5d showed no own furiosa-venv processes, `furiosa-smi ps` empty, npu1 0.00/47.50 GiB.

Job 2, rx 20260923-223724-rngd-serve-7413 (slot rngd-serve): the same command with the server in the background and a watchdog (`while kill -0 $PPID && kill -0 $S; do sleep 5; done; echo ...; kill -TERM $S`). Ready at about 22:37:57 (`tp=8, pp=1, dp=1`, `max_executable_len=131072`). `/v1/models` returned the same id; a CUDA -> OpenMP request (rx 20260923-223818-exec-e36e; `__global__ void scale(...)` to "C with OpenMP target offload", max_tokens 256) returned http=200 in 2.247767 s, 123 prompt and 146 completion tokens, finish_reason stop:

```
void scale(float *v, float s, int n) {
  #pragma omp target teams distribute parallel for map(to:v[0:n])
  for (int i = 0; i < n; i++) { v[i] *= s; }
}
int main() { int n = (n + 255) / 256 * 256; ... }
```

(The reply has real bugs: `map(to:)` drops the result, and `n` is read uninitialized. A correction loop has work to do.)

After `rx job kill`, the watchdog did not fire (rx 20260923-223845-exec-e085: server pid 3489924 alive with parent pid 1). The likely cause is that the `echo` after the loop wrote to the pipe of the dead runner, so SIGPIPE killed bash before its `kill`. The server was stopped with `kill -TERM 3489924` (rx 20260923-224207-exec-3277, exited within 5 s); rx 20260923-224223-exec-a67d showed 0 own servers, port 8123 closed, npu1 0.00 GiB.

Job 3, rx 20260923-224249-rngd-serve-de2d (slot rngd-serve), the working stop procedure:

```
trap "" PIPE
date -Is; furiosa-smi ps; furiosa-smi status | grep -a -E "\| rngd"
if ! furiosa-smi status | grep -a -E "\| npu1 " | grep -q " 0.00/"; then echo "ABORT: npu1 not free"; exit 2; fi
source /mnt/nvme10/joseph_ufl/furiosa-venv/bin/activate
export HF_HOME=/mnt/nvme10/joseph_ufl/.cache/huggingface HF_HUB_OFFLINE=1
furiosa-llm serve furiosa-ai/Llama-3.1-8B-Instruct --revision v2026.2 --devices npu:1 --host 127.0.0.1 --port 8123 --no-enable-prefix-caching &
S=$!; R=$PPID; echo "watchdog: server pid $S, runner pid $R"
while kill -0 $R 2>/dev/null && kill -0 $S 2>/dev/null; do sleep 5; done
kill -TERM $S 2>/dev/null
wait $S
```

Uvicorn was up at about 22:43:19; `/v1/models` returned the expected id (rx 20260923-224339-exec-94c1); `rx job kill` at about 22:43:41; rx 20260923-224349-exec-949a at 22:43:49: `server 3789762 exited`, own furiosa-llm serve count 0, port 8123 closed, `furiosa-smi ps` empty, npu1 0.00/47.50 GiB. `rx doctor` afterwards: `"running_jobs": []`.

### 6. Disk

`du -sh /mnt/nvme10/joseph_ufl` was 94G before and 94G after (rx 20260923-224223-exec-a67d). New files: a worktree slot `rngd-serve` and six request/response JSON files of a few KB under `$TMPDIR`. Nothing was downloaded and no venv was created.

### 7. Paths not taken, with sizes (Hub API, fetched 2026-09-24)

- furiosa-ai/Qwen2.5-Coder-7B-Instruct: only tag v2025.3.0 (https://huggingface.co/api/models/furiosa-ai/Qwen2.5-Coder-7B-Instruct/refs), no v2026.2, so furiosa-llm 2026.2.1 would not load it (bible Host Facts: revision pinned to the SDK tag).
- furiosa-ai/Qwen3-8B-FP8: tags v2026.3 and v2026.4 only; v2026.3 totals 10,001,870,870 bytes, FXB included (https://huggingface.co/api/models/furiosa-ai/Qwen3-8B-FP8/tree/v2026.3?recursive=true). It needs a furiosa-llm 2026.3.x venv; the 2026.2.1 venv is 8.0G, so a like-sized one plus the model would bring the scratch root to about 112G, PROJECTED, not measured.
- furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8 (arm A2): v2026.3 totals 31,825,437,206 bytes (https://huggingface.co/api/models/furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8/tree/v2026.3?recursive=true); with a new venv that is about 134G, PROJECTED, over the 120G cap. jun_ufl already serves this model on npu4-npu7 with a 2026.3-era FXB, which shows the A2 path works on this host; that server belongs to another tenant and was not queried.
- Model list: https://huggingface.co/api/models?author=furiosa-ai&limit=200 (fetched 2026-09-24).

## Finding

Yes. The stock venv (furiosa-llm 2026.2.1) serves the pre-compiled furiosa-ai/Llama-3.1-8B-Instruct@v2026.2 on one RNGD card (npu1) under driver 2026.3.1 and firmware 2026.3.0: tp 8, pp 1, dp 1, max_context_len 131072, ready in about 35 s, first 256-token completion in 3.94 s, with no download and no disk growth. Two operational traps were found and worked around:

- `furiosa-smi ps` hides other users' processes, so it reported npu4-npu7 free while jun_ufl's server held them. The occupancy check must read the `furiosa-smi status` memory column.
- `rx job kill` does not stop a job's command, because the command runs in its own session. The watchdog wrapper in job 3 makes `rx job kill` effective; a gate fix is needed.

Confidence: high for the serve procedure and versions (three launches, the same result each time); low for any latency figure beyond "a few seconds per short request" (three unrepeated, exploratory requests).

## Consequences for the plan

- Demo setup:
  - Start the job-3 command as `rx job start --slot rngd-serve --name rngd-serve --timeout <s>` on a card from npu1 to npu3 that shows 0.00 GiB in `furiosa-smi status`.
  - Confirm `/v1/models`.
  - Run LASSI on alpha01 through `rx run`, with the `openai_compat` backend at base_url `http://127.0.0.1:8123/v1` and model_id `furiosa-ai/Llama-3.1-8B-Instruct`. The server binds 127.0.0.1 with no API key, so the workstation cannot reach it and no tunnel is allowed.
  - Stop the job with `rx job kill <id>`, then verify with `rx exec` that no own furiosa-llm serve process remains and the card shows 0.00 GiB.
  - The serve job holds its slot busy, so it must not use the branch slot that `rx run` needs.
- Llama-3.1-8B-Instruct is the stock demo model, not one of arms A1 to A4. Trials against it are exploratory demo runs, not reproduction results.
- The provenance for such runs: furiosa-llm 2026.2.1, artifact furiosa-ai/Llama-3.1-8B-Instruct@v2026.2 (231d94f, artifact_id d6ae6a43-6ce0-4864-aaca-eeac4340234c), driver 2026.3.1, firmware 2026.3.0, npu1, tp 8, pp 1, dp 1, max_context_len 131072.
- `/v1/models` reports `max_context_len` and `max_prompt_len`, not `max_model_len`. The runner should record whichever field the server sends, and the serve log gives `max_executable_len` and `max_kv_len`.
- Gate: `verb_job_kill` must also signal the command's process group (fix task in `tools/server/gate.py`; J reinstalls the gate). The gate's `HF_HOME` (`$SCRATCH/hf`) differs from the populated cache (`$SCRATCH/.cache/huggingface`), so serve commands must set `HF_HOME` and `HF_HUB_OFFLINE=1`, or a 16G download starts.
- Agent Rule 8 names `furiosa-smi ps` as the pre-claim check, and that check misses other tenants. Changing an Agent Rule needs an owner decision (owner-queue item).
- P3 remains BLOCKED in plans/STATUS.md until J records otherwise; this spike does not change phase state.
