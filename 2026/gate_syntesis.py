from typing import Any

import numpy as np

from bloqade import squin
from bloqade.cirq_utils import load_circuit
from bloqade.types import Qubit
from kirin.dialects.ilist import IList
from qiskit.synthesis import gridsynth_rz

from gate_syntesis_helpers import *
from gate_syntesis_helpers import _qiskit_to_cirq
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
        return X_gate

    if n == 1:
        @squin.kernel
        def _rz_s(qubit) -> Register:
            squin.s(qubit)
            return qubit
        return _rz_s

    if n == 2:
        @squin.kernel
        def _rz_t(qubit) -> Register:
            squin.t(qubit)
            return qubit
        return _rz_t

    # n >= 3: usa gridsynth_rz per l'approssimazione Clifford+T
    theta = np.pi / 2**n
    qk_circ = gridsynth_rz(theta, epsilon=epsilon)
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



@squin.kernel
def Injected_T_gate(qubit, ancilla) -> Qubit:
    # Prepare the magic state |A> = T|+> on the ancilla.
    squin.h(ancilla)
    squin.t(ancilla)

    # Entangle the data qubit with the magic-state ancilla.
    squin.cx(qubit, ancilla)

    # Measure the ancilla.
    measurement = squin.measure(ancilla)

    # Feed-forward correction.
    # If the measurement is 1, the data qubit has T^\dagger instead of T.
    # Applying S gives S T^\dagger = T.
    if measurement:
        squin.s(qubit)

    return qubit


@squin.kernel
def Injected_Tdg_gate(qubit, ancilla) -> Qubit:
    qubit = Injected_T_gate(qubit, ancilla)
    squin.s(qubit)
    squin.s(qubit)
    squin.s(qubit)
    return qubit



@squin.kernel
def apply_injected_gate_sequence(qubit, ancillas, sequence) -> Qubit:
    ancilla_index = 0

    for gate_name in sequence:
        if gate_name == "h":
            squin.h(qubit)
        elif gate_name == "s":
            squin.s(qubit)
        elif gate_name == "sdg":
            squin.s(qubit)
            squin.s(qubit)
            squin.s(qubit)
        elif gate_name == "t":
            qubit = Injected_T_gate(qubit, ancillas[ancilla_index])
            ancilla_index += 1
        elif gate_name == "tdg":
            qubit = Injected_Tdg_gate(qubit, ancillas[ancilla_index])
            ancilla_index += 1

    return qubit

@squin.kernel
def Rz_gate_injected(qubit, ancillas, sequence) -> Qubit:
    return apply_injected_gate_sequence(qubit, ancillas, sequence)


@squin.kernel
def Rx_gate_injected(qubit, ancillas, sequence) -> Qubit:
    squin.h(qubit)
    qubit = apply_injected_gate_sequence(qubit, ancillas, sequence)
    squin.h(qubit)
    return qubit


@squin.kernel
def Ry_gate_injected(qubit, ancillas, sequence) -> Qubit:
    squin.s(qubit)
    squin.s(qubit)
    squin.s(qubit)
    squin.h(qubit)
    qubit = apply_injected_gate_sequence(qubit, ancillas, sequence)
    squin.h(qubit)
    squin.s(qubit)
    return qubit


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



#-----------------------


@squin.kernel
def test_steane_logical():
    qL = Steane_zero_logical_graph()



