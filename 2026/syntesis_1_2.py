from typing import Any

from bloqade import squin
from bloqade.types import Qubit
from kirin.dialects.ilist import IList

Register = IList[Qubit, Any]

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

def gate_distance(U, V):
    return np.sqrt(1 - abs(np.trace(U.conj().T @ V)) / 2)

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


@squin.kernel
def Rz_gate(qubit, n) -> Register:
    if (n == 0):
        qubit = Z_gate(qubit)
    elif (n == 1):
        squin.s(qubit)
    elif (n == 2):
        squin.t(qubit)
    elif (n in (3, 4, 5)):
        # TODO: replace with an actual approximation for Rz(pi/2^n).
        qubit = qubit


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
def Rz_gate_injected(qubit, ancillas, n) -> Register:
    if (n == 0):
        qubit = Z_gate(qubit)

    elif (n == 1):
        squin.s(qubit)

    elif (n == 2):
        # Rz(pi/4) = T, but T cannot be applied directly to the main qubit.
        qubit = Injected_T_gate(qubit, ancillas[0])

    elif (n == 3):
        # Approximation used in Part 2:
        # H T H T H T H T H S
        # Here every T is replaced by Injected_T_gate.

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[0])

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[1])

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[2])

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[3])

        squin.h(qubit)
        squin.s(qubit)

    elif (n == 4):
        # Optimized approximation from Part 2:
        # H T H T H T H S S S T H T H S S T H S H T H S S H S S H S H S S S T H S S T
        # It contains 9 T gates, hence 9 injected T gadgets.

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[0])

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[1])

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[2])

        squin.h(qubit)
        squin.s(qubit)
        squin.s(qubit)
        squin.s(qubit)
        qubit = Injected_T_gate(qubit, ancillas[3])

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[4])

        squin.h(qubit)
        squin.s(qubit)
        squin.s(qubit)
        qubit = Injected_T_gate(qubit, ancillas[5])

        squin.h(qubit)
        squin.s(qubit)

        squin.h(qubit)
        qubit = Injected_T_gate(qubit, ancillas[6])

        squin.h(qubit)
        squin.s(qubit)
        squin.s(qubit)

        squin.h(qubit)
        squin.s(qubit)
        squin.s(qubit)

        squin.h(qubit)
        squin.s(qubit)

        squin.h(qubit)
        squin.s(qubit)
        squin.s(qubit)
        squin.s(qubit)
        qubit = Injected_T_gate(qubit, ancillas[7])

        squin.h(qubit)
        squin.s(qubit)
        squin.s(qubit)
        qubit = Injected_T_gate(qubit, ancillas[8])

    elif (n == 5):
        # Baseline approximation for Rz(pi/32): identity.
        # This uses no T gates and therefore no injection.
        qubit = qubit

    return qubit

