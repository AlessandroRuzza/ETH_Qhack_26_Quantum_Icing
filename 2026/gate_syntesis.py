from typing import Any

import numpy as np

from bloqade import squin
from bloqade.cirq_utils import load_circuit
from bloqade.types import Qubit
from kirin.dialects.ilist import IList
from qiskit.synthesis import gridsynth_rz

from gate_syntesis_helpers import *
from gate_syntesis_helpers import _qiskit_to_cirq, _RzMeta, _rz_metrics
Register = IList[Qubit, Any]


@squin.kernel
def X_gate(qubit) -> Register:
    squin.h(qubit)
    squin.s(qubit)
    squin.s(qubit)
    squin.h(qubit)
    return qubit


@squin.kernel
def Y_gate(qubit) -> Register:
    squin.h(qubit)
    squin.s(qubit)
    squin.s(qubit)
    squin.h(qubit)
    squin.s(qubit)
    squin.s(qubit)
    return qubit


@squin.kernel
def Z_gate(qubit) -> Register:
    squin.s(qubit)
    squin.s(qubit)
    return qubit


@squin.kernel
def CZ_gate(qubit1, qubit2) -> Register:
    squin.h(qubit2)
    squin.cx(qubit1, qubit2)
    squin.h(qubit2)
    return [qubit1, qubit2]


@squin.kernel
def SWAP_gate(qubit1, qubit2) -> Register:
    squin.cx(qubit1, qubit2)
    squin.cx(qubit2, qubit1)
    squin.cx(qubit1, qubit2)
    return [qubit1, qubit2]


@squin.kernel
def Toffoli_gate(qubit1, qubit2, qubit3) -> Register:
    squin.h(qubit3)

    squin.cx(qubit2, qubit3)
    for _ in range(7):
        squin.t(qubit3)

    squin.cx(qubit1, qubit3)
    squin.t(qubit3)

    squin.cx(qubit2, qubit3)
    for _ in range(7):
        squin.t(qubit3)

    squin.cx(qubit1, qubit3)
    squin.t(qubit2)
    squin.t(qubit3)

    squin.h(qubit3)

    squin.cx(qubit1, qubit2)
    squin.t(qubit1)
    for _ in range(7):
        squin.t(qubit2)

    squin.cx(qubit1, qubit2)

    return [qubit1, qubit2, qubit3]


@squin.kernel
def CCZ_gate(qubit1, qubit2, qubit3) -> Register:
    squin.h(qubit3)
    Toffoli_gate(qubit1, qubit2, qubit3)
    squin.h(qubit3)
    return [qubit1, qubit2, qubit3]





def Rz_gate(n: int, epsilon: float = 1e-10):
    """Factory: restituisce un squin kernel che applica Rz(π/2^n) al qubit."""
    if n == 0:
        _rz_metrics[id(X_gate)] = _RzMeta(n=0, sequence=("h", "s", "s", "h"), theta=np.pi, circuit=None, ancillas=0)
        return X_gate

    if n == 1:
        @squin.kernel
        def _rz_s(qubit) -> Register:
            squin.s(qubit)
            return qubit
        _rz_metrics[id(_rz_s)] = _RzMeta(n=1, sequence=("s",), theta=np.pi / 2, circuit=None, ancillas=0)
        return _rz_s

    if n == 2:
        @squin.kernel
        def _rz_t(qubit) -> Register:
            squin.t(qubit)
            return qubit
        _rz_metrics[id(_rz_t)] = _RzMeta(n=2, sequence=("t",), theta=np.pi / 4, circuit=None, ancillas=0)
        return _rz_t

    theta     = np.pi / 2**n
    qk_circ   = gridsynth_rz(theta, epsilon=epsilon)
    sequence  = gate_sequence_from_circuit(qk_circ)
    cirq_circ = _qiskit_to_cirq(qk_circ)
    _inner = load_circuit(
        cirq_circ,
        kernel_name=f"Rz_{n}",
        register_as_argument=True,
        return_register=True,
    )

    @squin.kernel
    def _rz_approx(qubit) -> Register:
        _inner([qubit])
        return qubit

    _rz_metrics[id(_rz_approx)] = _RzMeta(n=n, sequence=sequence, theta=theta, circuit=qk_circ, ancillas=0)
    return _rz_approx


def Rx_gate(n: int, epsilon: float = 1e-10):
    """Rx(π/2^n) = H · Rz(π/2^n) · H"""
    rz = Rz_gate(n, epsilon)

    @squin.kernel
    def _rx(qubit) -> Register:
        squin.h(qubit)
        rz(qubit)
        squin.h(qubit)
        return qubit

    return _rx


def Ry_gate(n: int, epsilon: float = 1e-10):
    """Ry(π/2^n) = S · H · Rz(π/2^n) · H · S†"""
    rz = Rz_gate(n, epsilon)

    @squin.kernel
    def _ry(qubit) -> Register:
        squin.s(qubit)  # S† = S·S·S
        squin.s(qubit)
        squin.s(qubit)
        squin.h(qubit)
        rz(qubit)
        squin.h(qubit)
        squin.s(qubit)
        return qubit

    return _ry



def Rz_gate_injected(n: int, epsilon: float = 1e-10):
    """Factory: come Rz_gate ma i gate T/Tdg sono applicati via magic-state injection.
    Restituisce un kernel (qubit, ancillas) con t_count_from_sequence(seq) ancillas.
    Per n=0,1 non servono ancillas e il kernel restituito prende solo (qubit,).
    """
    if n == 0:
        _rz_metrics[id(X_gate)] = _RzMeta(n=0, sequence=("h", "s", "s", "h"), theta=np.pi, circuit=None, ancillas=0)
        return X_gate

    if n == 1:
        @squin.kernel
        def _rz_s(qubit) -> Register:
            squin.s(qubit)
            return qubit
        _rz_metrics[id(_rz_s)] = _RzMeta(n=1, sequence=("s",), theta=np.pi / 2, circuit=None, ancillas=0)
        return _rz_s

    sequence = ("t",) if n == 2 else gate_sequence_from_circuit(gridsynth_rz(np.pi / 2**n, epsilon=epsilon))
    anc = t_count_from_sequence(sequence)

    @squin.kernel
    def _rz_injected(qubit, ancillas) -> Qubit:
        return apply_injected_gate_sequence(qubit, ancillas, sequence)

    _rz_metrics[id(_rz_injected)] = _RzMeta(n=n, sequence=sequence, theta=np.pi / 2**n, circuit=None, ancillas=anc)
    return _rz_injected


def Rx_gate_injected(n: int, epsilon: float = 1e-10):
    """Rx(π/2^n) = H · Rz_injected(π/2^n) · H"""
    rz_inj = Rz_gate_injected(n, epsilon)

    if n <= 1:
        @squin.kernel
        def _rx(qubit) -> Register:
            squin.h(qubit)
            rz_inj(qubit)
            squin.h(qubit)
            return qubit
        return _rx

    @squin.kernel
    def _rx(qubit, ancillas) -> Qubit:
        squin.h(qubit)
        qubit = rz_inj(qubit, ancillas)
        squin.h(qubit)
        return qubit

    return _rx


def Ry_gate_injected(n: int, epsilon: float = 1e-10):
    """Ry(π/2^n) = S† · H · Rz_injected(π/2^n) · H · S"""
    rz_inj = Rz_gate_injected(n, epsilon)

    if n <= 1:
        @squin.kernel
        def _ry(qubit) -> Register:
            squin.s(qubit)
            squin.s(qubit)
            squin.s(qubit)
            squin.h(qubit)
            rz_inj(qubit)
            squin.h(qubit)
            squin.s(qubit)
            return qubit
        return _ry

    @squin.kernel
    def _ry(qubit, ancillas) -> Qubit:
        squin.s(qubit)
        squin.s(qubit)
        squin.s(qubit)
        squin.h(qubit)
        qubit = rz_inj(qubit, ancillas)
        squin.h(qubit)
        squin.s(qubit)
        return qubit

    return _ry


@squin.kernel
def Steane_zero_logical_graph() -> Register:
    q = squin.qalloc(7)

    # The diagram starts from |+> on every physical qubit.
    # qalloc starts from |0>, so apply H to all qubits first.
    for i in range(7):
        squin.h(q[i])

    # CZ edges from the graph-state preparation circuit.
    CZ_gate(q[0], q[6])

    CZ_gate(q[1], q[3])
    CZ_gate(q[5], q[3])

    CZ_gate(q[0], q[4])
    CZ_gate(q[5], q[6])

    CZ_gate(q[1], q[2])
    CZ_gate(q[0], q[2])

    CZ_gate(q[5], q[4])
    CZ_gate(q[1], q[4])

    # Final H gates shown in the figure.
    squin.h(q[2])
    squin.h(q[3])
    squin.h(q[4])
    squin.h(q[6])

    return q


@squin.kernel
def Steane_H_logical(q) -> Register:
    for i in range(7):
        squin.h(q[i])
    return q


@squin.kernel
def Steane_S_logical(q) -> Register:
    for i in range(7):
        squin.s(q[i])
        squin.s(q[i])
        squin.s(q[i])
    return q


@squin.kernel
def Steane_CNOT_logical(control_block, target_block) -> Register:
    for i in range(7):
        squin.cx(control_block[i], target_block[i])
    return control_block + target_block

@squin.kernel
def Steane_Sdg_logical(q) -> Register:
    for i in range(7):
        squin.s(q[i])
    return q



#-----------------------


@squin.kernel
def test_steane_logical():
    qL = Steane_zero_logical_graph()


@squin.kernel
def Steane_prepare_magic_state_logical(q) -> Register:
    squin.h(q[0])
    squin.h(q[1])
    squin.h(q[3])

    squin.h(q[6])
    squin.t(q[6])

    squin.cx(q[0], q[2])
    squin.cx(q[1], q[2])

    squin.cx(q[0], q[4])
    squin.cx(q[3], q[4])

    squin.cx(q[1], q[5])
    squin.cx(q[3], q[5])

    for i in range(6):
        squin.cx(q[6], q[i])

    squin.cx(q[2], q[6])
    squin.cx(q[3], q[6])

    return q


@squin.kernel
def Steane_magic_state_logical() -> Register:
    q = squin.qalloc(7)
    Steane_prepare_magic_state_logical(q)
    return q


@squin.kernel
def Steane_T_logical_reset(logical_block) -> Register:
    for i in range(7):
        squin.reset(logical_block[i])
    return logical_block


@squin.kernel
def Steane_T_injection_logical(data_block, magic_block) -> Register:
    Steane_CNOT_logical(data_block, magic_block)

    m = Steane_measure_logical_Z_weight3(magic_block)

    if m:
        Steane_S_logical(data_block)

    Steane_T_logical_reset(magic_block)

    return data_block

@squin.kernel
def Steane_Tdg_injection_logical(data_block, magic_block) -> Register:
    Steane_CNOT_logical(data_block, magic_block)

    m = Steane_measure_logical_Z_weight3(magic_block)

    if m:
        Steane_S_logical(data_block)

    Steane_Sdg_logical(data_block)
    Steane_T_logical_reset(magic_block)

    return data_block

@squin.kernel
def Steane_apply_logical_gate_sequence(data_block, magic_blocks, sequence) -> Register:
    magic_index = 0

    for gate_name in sequence:
        if gate_name == "h":
            Steane_H_logical(data_block)

        elif gate_name == "s":
            Steane_S_logical(data_block)

        elif gate_name == "sdg":
            Steane_Sdg_logical(data_block)

        elif gate_name == "t":
            Steane_T_injection_logical(data_block, magic_blocks[magic_index])
            magic_index += 1

        elif gate_name == "tdg":
            Steane_Tdg_injection_logical(data_block, magic_blocks[magic_index])
            magic_index += 1

    return data_block


@squin.kernel
def Steane_apply_logical_gate_sequence_reuse_magic(data_block, magic_block, sequence) -> Register:
    for gate_name in sequence:
        if gate_name == "h":
            Steane_H_logical(data_block)

        elif gate_name == "s":
            Steane_S_logical(data_block)

        elif gate_name == "sdg":
            Steane_Sdg_logical(data_block)

        elif gate_name == "t":
            Steane_prepare_magic_state_logical(magic_block)
            Steane_T_injection_logical(data_block, magic_block)

        elif gate_name == "tdg":
            Steane_prepare_magic_state_logical(magic_block)
            Steane_Tdg_injection_logical(data_block, magic_block)

    return data_block


def make_part4_logical_fidelity_kernel_reuse_magic(sequence):
    """Build a Part 4 fidelity kernel that simulates arbitrary T-count.

    A single 7-qubit logical magic-state block is reused: before each T/Tdg
    injection it is prepared as |A_L>, and the injection gadget resets it at the
    end. This keeps the simulation at 14 physical qubits instead of allocating
    one 7-qubit block per T gate.
    """
    sequence = tuple(sequence)

    @squin.kernel
    def _part4_logical_fidelity_kernel():
        data_block = Steane_zero_logical_graph()
        magic_block = squin.qalloc(7)

        Steane_H_logical(data_block)
        Steane_apply_logical_gate_sequence_reuse_magic(data_block, magic_block, sequence)

        return data_block

    return _part4_logical_fidelity_kernel


def make_part4_logical_fidelity_kernel(sequence):
    """Build a small Steane logical-state fidelity kernel.

    The kernel prepares |0_L>, applies H_L so the following Rz_L has a visible
    effect, then applies the Part 2 Clifford+T sequence using the Part 4 logical
    gate implementation.

    This helper intentionally supports only 0 or 1 injected T/Tdg gate, which is
    enough for the small statevector checks in the notebook. Larger sequences
    require many 7-qubit magic-state blocks and are better treated as resource
    estimates.
    """
    sequence = tuple(sequence)
    logical_t_count = t_count_from_sequence(sequence)

    if logical_t_count == 0:
        @squin.kernel
        def _part4_logical_fidelity_kernel():
            data_block = Steane_zero_logical_graph()
            Steane_H_logical(data_block)
            Steane_apply_logical_gate_sequence(data_block, [], sequence)
            return data_block

        return _part4_logical_fidelity_kernel

    if logical_t_count == 1:
        @squin.kernel
        def _part4_logical_fidelity_kernel():
            data_block = Steane_zero_logical_graph()
            Steane_H_logical(data_block)
            magic0 = Steane_magic_state_logical()
            Steane_apply_logical_gate_sequence(data_block, [magic0], sequence)
            return data_block

        return _part4_logical_fidelity_kernel

    raise ValueError(
        "Part 4 statevector fidelity helper supports only sequences with "
        f"0 or 1 T/Tdg gate; got {logical_t_count}."
    )
