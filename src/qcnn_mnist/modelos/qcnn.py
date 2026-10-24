"""QCNN híbridas: las tres etapas del estudio y las ablaciones de v3."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pennylane as qml

from qcnn_mnist.modelos.base import (
    BIAS_ENTRADA,
    N_CLASES,
    LecturaProbabilidades,
    Modelo,
    angular,
    bias_entrada,
    he,
)
from qcnn_mnist.modelos.circuitos import (
    codificar,
    conv_ry,
    conv_rot,
    pooling_parametrico,
)

# Pooling fijo de v1/v2 (10 → 8 → 4): CNOT del qubit descartado al superviviente.
POOL1_V12 = [(0, 1), (2, 3)]
SUPERVIVIENTES1_V12 = [1, 3, 4, 5, 6, 7, 8, 9]
POOL2_V12 = [(1, 3), (4, 5), (6, 7), (8, 9)]
SUPERVIVIENTES2_V12 = [3, 5, 7, 9]


def _qnode(n_qubits: int, dispositivo: str, diff_method: str):
    dev = qml.device(dispositivo, wires=n_qubits)
    return lambda f: qml.qnode(dev, interface="jax", diff_method=diff_method)(f)


class QCNNv1(LecturaProbabilidades):
    """v1: RY + anillo de CNOT, pirámide 10 → 8 → 4, probabilidades de 16 estados.

    Bias de la capa de entrada fijo en ``(1e-3, 0)``; λ = 0,3.
    """

    nombre = "qcnn_v1"
    tam_imagen, tam_parche, n_parches, n_qubits = 10, 2, 10, 10
    lam = 0.3

    def __init__(self, dispositivo: str = "default.qubit", diff_method: str = "best"):
        @_qnode(self.n_qubits, dispositivo, diff_method)
        def circuito(angulos, theta1, theta2):
            wires = list(range(self.n_qubits))
            codificar(angulos, wires)
            conv_ry(theta1, wires)
            for a, b in POOL1_V12:
                qml.CNOT(wires=[a, b])
            conv_ry(theta2, SUPERVIVIENTES1_V12)
            for a, b in POOL2_V12:
                qml.CNOT(wires=[a, b])
            return qml.probs(wires=SUPERVIVIENTES2_V12)

        self.circuito = circuito

    def inicializar(self, clave):
        k1, k2, k3 = jax.random.split(clave, 3)
        d = self.tam_parche**2
        entrenables = {
            "W": he(k1, (2, d)),
            "theta1": angular(k2, (self.n_qubits,)),
            "theta2": angular(k3, (len(SUPERVIVIENTES1_V12),)),
        }
        return entrenables, {"b": BIAS_ENTRADA}

    def salida(self, entrenables, fijos, x):
        ang = self.angulos(entrenables["W"], fijos["b"], x)
        return self.circuito(ang, entrenables["theta1"], entrenables["theta2"])


class QCNNv2(LecturaProbabilidades):
    """v2: como v1 con ``Rot`` SU(2), *re-uploading* antes de la segunda convolución
    y bias de entrada entrenable; λ = 0,1."""

    nombre = "qcnn_v2"
    tam_imagen, tam_parche, n_parches, n_qubits = 10, 2, 10, 10
    lam = 0.1

    def __init__(self, dispositivo: str = "default.qubit", diff_method: str = "best"):
        @_qnode(self.n_qubits, dispositivo, diff_method)
        def circuito(angulos, theta1, theta2):
            wires = list(range(self.n_qubits))
            codificar(angulos, wires)
            conv_rot(theta1, wires)
            for a, b in POOL1_V12:
                qml.CNOT(wires=[a, b])
            for q in SUPERVIVIENTES1_V12:  # re-uploading del dato original de cada qubit
                qml.RY(angulos[..., q], wires=q)
            conv_rot(theta2, SUPERVIVIENTES1_V12)
            for a, b in POOL2_V12:
                qml.CNOT(wires=[a, b])
            return qml.probs(wires=SUPERVIVIENTES2_V12)

        self.circuito = circuito

    def inicializar(self, clave):
        k1, k2, k3, k4 = jax.random.split(clave, 4)
        d = self.tam_parche**2
        entrenables = {
            "W": he(k1, (2, d)),
            "b": bias_entrada(k2),
            "theta1": angular(k3, (self.n_qubits, 3)),
            "theta2": angular(k4, (len(SUPERVIVIENTES1_V12), 3)),
        }
        return entrenables, {}

    def salida(self, entrenables, fijos, x):
        ang = self.angulos(entrenables["W"], entrenables["b"], x)
        return self.circuito(ang, entrenables["theta1"], entrenables["theta2"])


class QCNNv3(Modelo):
    """v3: 16 qubits, pirámide 16 → 8 → 4 con pooling paramétrico, ⟨Z⟩ de 4 qubits
    y capa lineal 4 → 10.

    Circuito: RY(θ) → conv Rot 16 → pool 16→8 → re-uploading en los 8 → conv Rot 8
    → pool 8→4 → conv Rot 4 → ⟨Z⟩.

    Ablaciones:
        entrelazar=False: sin ninguna compuerta de dos qubits (anillos ni pooling).
        congelar=True: parámetros cuánticos aleatorios y fijos; solo se entrenan
            la capa de entrada y la de salida.
    """

    tam_imagen, tam_parche, n_parches, n_qubits = 16, 4, 16, 16
    CUANTICOS = ("theta1", "pool1", "theta2", "pool2", "theta3")

    def __init__(
        self,
        entrelazar: bool = True,
        congelar: bool = False,
        dispositivo: str = "default.qubit",
        diff_method: str = "best",
    ):
        self.entrelazar, self.congelar = entrelazar, congelar
        self.nombre = "qcnn_v3"
        if not entrelazar:
            self.nombre = "v3_sin_entrelazamiento"
        if congelar:
            self.nombre = "v3_congelado"

        @_qnode(self.n_qubits, dispositivo, diff_method)
        def circuito(angulos, theta1, pool1, theta2, pool2, theta3):
            wires = list(range(self.n_qubits))
            codificar(angulos, wires)
            conv_rot(theta1, wires, entrelazar)
            s1 = pooling_parametrico(pool1, wires, entrelazar)
            for q in s1:  # re-uploading
                qml.RY(angulos[..., q], wires=q)
            conv_rot(theta2, s1, entrelazar)
            s2 = pooling_parametrico(pool2, s1, entrelazar)
            conv_rot(theta3, s2, entrelazar)
            return [qml.expval(qml.PauliZ(w)) for w in s2]

        self.circuito = circuito

    def inicializar(self, clave):
        k = jax.random.split(clave, 8)
        d = self.tam_parche**2
        clasicos = {
            "W": he(k[0], (2, d)),
            "b": bias_entrada(k[1]),
            "W_out": he(k[2], (N_CLASES, 4)),
            "b_out": jnp.zeros(N_CLASES),
        }
        cuanticos = {
            "theta1": angular(k[3], (16, 3)),
            "pool1": angular(k[4], (8,)),
            "theta2": angular(k[5], (8, 3)),
            "pool2": angular(k[6], (4,)),
            "theta3": angular(k[7], (4, 3)),
        }
        if self.congelar:
            return clasicos, cuanticos
        return {**clasicos, **cuanticos}, {}

    def salida(self, entrenables, fijos, x):
        p = {**fijos, **entrenables}
        ang = self.angulos(p["W"], p["b"], x)
        z = jnp.stack(self.circuito(ang, *(p[n] for n in self.CUANTICOS)), axis=-1)
        return z @ p["W_out"].T + p["b_out"]
