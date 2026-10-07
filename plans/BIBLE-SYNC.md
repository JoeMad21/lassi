# Bible edits awaiting the master copy

AGENTS.md, Authority: when the master copy is unreachable, each mirror edit is appended here, exact, for the owner. Apply them to the master in order, then clear an entry once it is mirrored.

## 2026-10-07: the owner's directions on server access and the MI300Xs (mirror at master revision 266)

1. Agent Rules, rule 17

Old:

````
17. No model inference on alpha01 without the owner's approval in the working session. File uploads and compiler work need none; serving a model (RNGD included), loading one in process (hf_local), and any training run wait for the owner's yes, asked with what runs, for how long, and what it touches.
````

New:

````
17. No access to alpha01 without the owner's approval in the working session: no rx command of any kind (run, job, exec, doctor, pull, or sync), read-only probes, file uploads, and compiler work included, and no model inference (serving, RNGD included, or loading one in process) or training there. Ask once per batch of server steps, with what runs, for how long, and what it touches, and wait for the owner's yes.
````

2. Environment State, the AMD MI300X row, last cell

Old:

````
its GPU half has no host: the owner said on 2026-10-06 that the MI300Xs will not be granted and to focus on Tenstorrent and CPU (OQ-040) |
````

New:

````
its GPU half has no host for now: the owner said on 2026-10-06 that there is no MI300X access and to focus on Tenstorrent and CPU, and on 2026-10-07 that access may come in the future (OQ-040) |
````

3. Build Roadmap, the work-order sentence

Old:

````
the GPU half runs only if a GPU host is granted, and none is planned (OQ-038; OQ-040, 2026-10-06).
````

New:

````
the GPU half runs only once a GPU host is granted; none is planned now, and MI300X access may come later (OQ-038; OQ-040, 2026-10-06 and 2026-10-07).
````

4. Risks And Questions, the no-GPU-host row, last cell

Old:

````
the owner said on 2026-10-06 that the MI300Xs will not be granted, so no GPU host is planned and these stay unmeasured |
````

New:

````
the owner said on 2026-10-06 that there is no MI300X access for now and on 2026-10-07 that it may come in the future, so these stay unmeasured until a GPU host is granted |
````

5. Decision Log, the count sentence

Old:

````
One hundred and thirty-one decisions have been made: twenty-six on 2026-09-22, twenty on 2026-09-23, thirty-six on 2026-09-24, seven on 2026-09-25, sixteen on 2026-09-26, five on 2026-09-28, three on 2026-09-30, two on 2026-10-04, three on 2026-10-05, twelve on 2026-10-06, and one on 2026-10-07;
````

New:

````
One hundred and thirty-three decisions have been made: twenty-six on 2026-09-22, twenty on 2026-09-23, thirty-six on 2026-09-24, seven on 2026-09-25, sixteen on 2026-09-26, five on 2026-09-28, three on 2026-09-30, two on 2026-10-04, three on 2026-10-05, twelve on 2026-10-06, and three on 2026-10-07;
````

6. Decision Log, two new rows at the top of the table

Old:

````
| Date | Decision | Rationale |
| --- | --- | --- |
````

New:

````
| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-10-07 | MI300X access may come later (OQ-040; Environment State, Build Roadmap, Risks And Questions; plans/STATUS.md, plans/p17-portable.md). The owner said in the working session: "to be clear, we may in the future have access to the MI300Xs." The 2026-10-06 answer means no MI300X access now, not never: P17.12, P17.13, P17.G, P8, and P9 stay BLOCKED until the owner grants access, and OQ-040 reopens then | The 2026-10-06 entry recorded the answer as the MI300Xs not being granted; the owner's clarification narrows it to the present |
| 2026-10-07 | Agent Rule 17 widened: no access to alpha01 at all without the owner's approval in the working session (the owner, opening the session: "Keep going, you do not have access to the server without my approval."; Agent Rules; AGENTS.md, Remote Execution). Every rx command, read-only probes, uploads, and compiles included, waits for the owner's yes, asked once per batch of server steps with what runs, for how long, and what it touches; a task whose server steps lack it goes OWNER. The read-only probes task P5.1 ran on 2026-10-06 came before this widening | The rule of 2026-10-06 held only inference and training; the owner now holds every server action |
````
