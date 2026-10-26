# The host skeleton's clean-commit build (task P17.33)

Provenance: provenance.json (rx 20261025-202808-desktop-8r113ei-detached-36f263b3-2007; commit
36f263b34c51c3e5213a393137950cffaa6a2873, dirty false; alpha01, Ubuntu 22.04.5 LTS, kernel
6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+; 2026-10-25 20:28:08 to 20:28:16 on alpha01's clock, 23:28 EDT; pins in
provenance.json, tt-metal at 5280a9cf). Approved in the working session (the owner's yes of 2026-10-25, about
23:25 EDT).

Command, from the clean commit's checkout: `bash plans/runs/night/skeleton_build.sh`, which compiles
assets/skeletons/tt-host-skeleton-v0/host.cpp with the ttmetal-host toolchain's command line (toolchains/tt-metal.pin,
the HOST_* keys) against the pinned tree $LASSI_TOOLCHAINS/tt-metal@5280a9cf, in a fresh directory under
$LASSI_SCRATCH, and runs nothing.

## Result [MEASURED, alpha01, 2026-10-25]

From the job's output (the rx run's printed log):

```
core limit (soft hard): 1 1
commit: 36f263b34c51c3e5213a393137950cffaa6a2873 (0 changed paths)
pin: tt-metal 5280a9cfb00998fd49667a29523d03aee905c129; tree: /mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf
host.cpp sha256: 5756f7dab5783acf29886a846fe644aa50bd33962fe1bd2135d02e88e0917515
compiler: /usr/bin/clang++-20 (Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132))
compile exit status: 0
artifact: main, 64912 bytes
error lines: 0; warning lines: 0
```

host.cpp as committed (sha256 5756f7da..., the value in its MANIFEST.yaml) compiles and links at the pin with no
error and no warning. Nothing ran: the skeleton's placeholder kernel paths hold no kernel, so no ttsim run is
implied.
