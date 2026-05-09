from typing import Any

import numpy as np
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
        qubit = X_gate(qubit)
    elif (n == 1):
        squin.s(qubit)
    elif (n == 2):
        squin.t(qubit)
    return qubit


@squin.kernel
def Rx_gate(qubit, n) -> Register:
    squin.h(qubit)
    Rz_gate(qubit, n)
    squin.h(qubit)
    return qubit

