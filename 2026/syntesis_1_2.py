from bloqade import squin

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
    squin.H(qubit3)
    Toffoli_gate(qubit1, qubit2)
    squin.H(qubit3)
    return [qubit1, qubit2, qubit3]


@squin.kernel
def Rz_gate(qubit, n) -> Register:
    if (n == 0):
        qubit = X_gate(qubit)
    elif (n == 1):
        squin.S(qubit)
    elif (n == 2):
        squin.T(qubit)
    elif (n == 3):
        pass
    elif (n == 4):
        pass
    elif (n == 5):
        pass


    return qubit