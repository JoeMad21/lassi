"""Harness: the control plane's side of the harness I/O contract (bible Harness Contract, Frontend Rules).

- lassi.harness.lassi_io: the lassi_io file format (one named array per
  binary file), its reader and writer, and the seeded generator of held-out
  input files, and write_inputs, the writer of a proposed input set (task
  P16.5). Programs read and write the same files through the C and C++
  header assets/harness/c/lassi_io.h.
- lassi.harness.contract: an item's input contract (input_contract, from its
  held-out input spec and the optional manifest key `contract`) and the
  check that refuses a proposed input set outside it before any run
  (check_inputs, InputContractError with one of CONTRACT_REASONS).
"""

from lassi.harness.contract import (
    CONTRACT_REASONS,
    ContractInput,
    InputContract,
    InputContractError,
    check_inputs,
    input_contract,
)
from lassi.harness.lassi_io import (
    DTYPES,
    MAGIC,
    VERSION,
    DType,
    LassiArray,
    LassiIOError,
    Pcg32,
    check_input_spec,
    generate_inputs,
    read_array,
    stream_of,
    write_array,
    write_inputs,
)

__all__ = [
    "CONTRACT_REASONS",
    "ContractInput",
    "DTYPES",
    "MAGIC",
    "VERSION",
    "DType",
    "InputContract",
    "InputContractError",
    "LassiArray",
    "LassiIOError",
    "Pcg32",
    "check_input_spec",
    "check_inputs",
    "generate_inputs",
    "input_contract",
    "read_array",
    "stream_of",
    "write_array",
    "write_inputs",
]
