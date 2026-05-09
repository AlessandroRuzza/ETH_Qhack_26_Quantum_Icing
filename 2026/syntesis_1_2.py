from typing import Any

import cirq
import numpy as np

from bloqade import squin
from bloqade.cirq_utils import load_circuit
from bloqade.types import Qubit
from kirin.dialects.ilist import IList
from qiskit import qasm2
from qiskit.synthesis import gridsynth_rz

Register = IList[Qubit, Any]

_GATE_MAP = {
    "h":   cirq.H,
    "s":   cirq.S,
    "sdg": cirq.S**-1,
    "t":   cirq.T,
    "tdg": cirq.T**-1,
    "x":   cirq.X,
}

def _qiskit_to_cirq(qk_circ) -> cirq.Circuit:
    q = cirq.LineQubit(0)
    ops = [_GATE_MAP[instr.operation.name](q) for instr in qk_circ.data]
    return cirq.Circuit(ops)

def Rz(theta):
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


I = np.eye(2, dtype=complex)

H = 1 / np.sqrt(2) * np.array([
    [1, 1],
    [1, -1]
], dtype=complex)

S = np.array([
    [1, 0],
    [0, 1j]
], dtype=complex)

T = np.array([
    [1, 0],
    [0, np.exp(1j * np.pi / 4)]
], dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

S_DAGGER = S.conj().T
T_DAGGER = T.conj().T

def gate_distance(U, V):
    return np.sqrt(1 - abs(np.trace(U.conj().T @ V)) / 2)


def gate_sequence_from_circuit(circuit):
    """Extract the one-qubit Clifford+T gate sequence emitted by gridsynth_rz."""
    supported_gates = {"id", "h", "s", "sdg", "t", "tdg", "x"}
    sequence = []

    for instruction in circuit.data:
        name = instruction.operation.name.lower()

        if name in {"barrier", "delay"}:
            continue

        if len(instruction.qubits) != 1:
            raise ValueError(f"Expected a one-qubit circuit, found gate {name!r}.")

        if name not in supported_gates:
            raise ValueError(f"Unsupported gate from synthesized circuit: {name!r}.")

        if name != "id":
            sequence.append(name)

    return tuple(sequence)


def unitary_from_gate_sequence(sequence):
    """Build the logical one-qubit unitary for a Clifford+T gate sequence."""
    gate_matrices = {
        "h": H,
        "s": S,
        "sdg": S_DAGGER,
        "t": T,
        "tdg": T_DAGGER,
        "x": X,
    }

    U = I.copy()

    for gate_name in sequence:
        U = gate_matrices[gate_name] @ U

    return U


def t_count_from_sequence(sequence):
    return sum(gate_name in {"t", "tdg"} for gate_name in sequence)

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




    return qubit

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
def Postselected_T_gate(qubit, ancilla) -> int:
    # Same magic-state gadget as Injected_T_gate, but without feed-forward.
    # A returned 1 marks a shot that must be dropped in postselection.
    squin.h(ancilla)
    squin.t(ancilla)
    squin.cx(qubit, ancilla)
    return squin.measure(ancilla)


@squin.kernel
def Injected_Tdg_gate(qubit, ancilla) -> Qubit:
    qubit = Injected_T_gate(qubit, ancilla)
    squin.s(qubit)
    squin.s(qubit)
    squin.s(qubit)
    return qubit


@squin.kernel
def Postselected_Tdg_gate(qubit, ancilla) -> int:
    measurement = Postselected_T_gate(qubit, ancilla)
    squin.s(qubit)
    squin.s(qubit)
    squin.s(qubit)
    return measurement


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
def apply_postselected_gate_sequence(qubit, ancillas, sequence) -> int:
    ancilla_index = 0
    dropout = False

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
            measurement = Postselected_T_gate(qubit, ancillas[ancilla_index])
            dropout = dropout | measurement
            ancilla_index += 1
        elif gate_name == "tdg":
            measurement = Postselected_Tdg_gate(qubit, ancillas[ancilla_index])
            dropout = dropout | measurement
            ancilla_index += 1

    return dropout


def make_postselected_dropout_kernel(sequence):
    sequence = tuple(sequence)
    ancilla_count = t_count_from_sequence(sequence)

    @squin.kernel
    def postselected_dropout_kernel() -> int:
        qubits = squin.qalloc(1 + ancilla_count)
        return apply_postselected_gate_sequence(qubits[0], qubits[1:], sequence)

    return postselected_dropout_kernel
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
    """Ry(π/2^n) = S · H · Rz(π/2^n) · H · S†
    S† implementato come S·S·S (S⁴ = I)
    """
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


def print_metrics(n: int, epsilon: float = 1e-10) -> None:
    """Stampa le metriche del circuito Clifford+T che approssima Rz(π/2^n)."""
    theta = np.pi / 2**n

    if n == 0:
        print(f"Rz(π/2^{n}) = X  [gate esatto]")
        print(f"  T-count   : 0")
        print(f"  Clifford  : 4  (H S S H)")
        print(f"  Total     : 4")
        print(f"  Depth     : 4")
        return

    if n == 1:
        print(f"Rz(π/2^{n}) = S  [gate esatto]")
        print(f"  T-count   : 0")
        print(f"  Clifford  : 1")
        print(f"  Total     : 1")
        print(f"  Depth     : 1")
        return

    if n == 2:
        print(f"Rz(π/2^{n}) = T  [gate esatto]")
        print(f"  T-count   : 1")
        print(f"  Clifford  : 0")
        print(f"  Total     : 1")
        print(f"  Depth     : 1")
        return

    qk_circ = gridsynth_rz(theta, epsilon=epsilon)
    ops = qk_circ.count_ops()
    t_count = sum(ops.get(g, 0) for g in ("t", "tdg"))
    clifford = sum(v for g, v in ops.items() if g not in ("t", "tdg"))
    total = sum(ops.values())

    print(f"Rz(π/2^{n})  θ={theta:.6f}  ε={epsilon:.0e}  [approssimato]")
    print(f"  T-count   : {t_count}")
    print(f"  Tdg-count : {ops.get('tdg', 0)}")
    print(f"  Clifford  : {clifford}  {dict(ops)}")
    print(f"  Total     : {total}")
    print(f"  Depth     : {qk_circ.depth()}")
