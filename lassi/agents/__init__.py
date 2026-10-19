"""Agents: registered components that recipes bind to roles (bible Agents And Judges; Repository Layout).

An adversary is an Agent that declares the capability proposes_inputs and
provides propose(view) (lassi.core.capabilities PROPOSES_INPUTS and
Adversary; lassi.core.interfaces AdversaryView, Proposal, and
ProposalOutcome; task P16.2). A recipe's adversary section binds one at
adversary.kind, and the loader refuses an Agent there that does not declare
the capability. It sees the item's input contract, the source program, the
candidate, its budget, and its earlier outcomes, never a held-out input, the
item's seed, or a reference output, and proposes data, never code.

Agents here are generic components: no module names a project, a suite, a
model, or a host, and none imports the projects package (Agent Rule 3). No
agent is registered here yet; the fuzzer and the llm adversary come with
tasks P16.3 and P16.4.
"""
