import cirq
import numpy as np
from bloqade import squin
from qiskit.synthesis import gridsynth_rz

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


def print_metrics(n: int, epsilon: float = 1e-10) -> None:
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


# --- Postselected gate gadgets ---

@squin.kernel
def Postselected_T_gate(qubit, ancilla) -> int:
    # Same magic-state gadget as Injected_T_gate, but without feed-forward.
    # A returned 1 marks a shot that must be dropped in postselection.
    squin.h(ancilla)
    squin.t(ancilla)
    squin.cx(qubit, ancilla)
    return squin.measure(ancilla)


@squin.kernel
def Postselected_Tdg_gate(qubit, ancilla) -> int:
    measurement = Postselected_T_gate(qubit, ancilla)
    squin.s(qubit)
    squin.s(qubit)
    squin.s(qubit)
    return measurement


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


@squin.kernel
def Steane_measure_logical_Z_weight3(q) -> int:
    m0 = squin.measure(q[0])
    m1 = squin.measure(q[1])
    m2 = squin.measure(q[2])

    return m0 ^ m1 ^ m2

    
