"""Bloques de circuito compartidos por las QCNN.

Los ángulos de codificación llegan con forma ``(B, n)``; PennyLane los difunde
(*parameter broadcasting*) y simula el lote completo en una sola ejecución.
"""

from __future__ import annotations

import pennylane as qml


def codificar(angulos, wires):
    """``RY(θ_i)`` en cada qubit; ``angulos[..., i]`` corresponde a ``wires[i]``."""
    for i, w in enumerate(wires):
        qml.RY(angulos[..., i], wires=w)


def anillo_cnot(wires):
    """CNOT de cada qubit al siguiente, cerrando el anillo."""
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def conv_ry(thetas, wires, entrelazar: bool = True):
    """Convolución de v1: ``RY`` entrenable por qubit y anillo de CNOT."""
    for i, w in enumerate(wires):
        qml.RY(thetas[i], wires=w)
    if entrelazar:
        anillo_cnot(wires)


def conv_rot(thetas, wires, entrelazar: bool = True):
    """Convolución de v2/v3: ``Rot`` SU(2) por qubit (3 ángulos) y anillo de CNOT."""
    for i, w in enumerate(wires):
        qml.Rot(thetas[i, 0], thetas[i, 1], thetas[i, 2], wires=w)
    if entrelazar:
        anillo_cnot(wires)


def pooling_parametrico(params, wires, entrelazar: bool = True):
    """Pooling de v3: en cada par ``(a, b)``, CNOT de ``a`` a ``b`` y ``RY`` entrenable en ``b``.

    Devuelve los supervivientes (el segundo de cada par).
    """
    supervivientes = []
    for k in range(0, len(wires), 2):
        a, b = wires[k], wires[k + 1]
        if entrelazar:
            qml.CNOT(wires=[a, b])
        qml.RY(params[k // 2], wires=b)
        supervivientes.append(b)
    return supervivientes
