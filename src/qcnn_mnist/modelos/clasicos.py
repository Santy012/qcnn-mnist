"""Controles clásicos y la ablación "solo clásico"."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qcnn_mnist.codificacion import EPS_NORMA
from qcnn_mnist.modelos.base import N_CLASES, Modelo, angular, bias_entrada, he


def glorot_uniforme(clave, forma, fan_in: int, fan_out: int):
    """Inicialización por defecto de Keras, la que usaba la CNN original."""
    limite = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(clave, forma, minval=-limite, maxval=limite)


def _media_pares(x, indices):
    """Promedia los pares ``(a, b)`` de ``indices`` y conserva el resto en orden."""
    return jnp.stack([(x[..., a] + x[..., b]) / 2 if b is not None else x[..., a] for a, b in indices], -1)


class CNN(Modelo):
    """CNN restringida del primer notebook: un filtro ``parche × parche`` con paso igual
    al parche (equivale a procesar los mismos parches que la QCNN), ReLU, una densa
    del tamaño del número de qubits con ReLU y la salida de 10 clases."""

    def __init__(self, tam_imagen: int, tam_parche: int, n_parches: int):
        self.tam_imagen, self.tam_parche, self.n_parches = tam_imagen, tam_parche, n_parches
        self.n_oculta = n_parches  # = número de qubits de la QCNN que controla
        self.nombre = f"cnn_{tam_imagen}"

    def inicializar(self, clave):
        k1, k2, k3 = jax.random.split(clave, 3)
        d, n, h = self.tam_parche**2, self.n_parches, self.n_oculta
        return {
            "kernel": glorot_uniforme(k1, (d,), d, d),
            "c": jnp.zeros(()),
            "W1": glorot_uniforme(k2, (h, n), n, h),
            "b1": jnp.zeros(h),
            "W2": glorot_uniforme(k3, (N_CLASES, h), h, N_CLASES),
            "b2": jnp.zeros(N_CLASES),
        }, {}

    def salida(self, entrenables, fijos, x):
        p = entrenables
        mapa = jax.nn.relu(self.parches(x) @ p["kernel"] + p["c"])  # (B, n_parches)
        oculta = jax.nn.relu(mapa @ p["W1"].T + p["b1"])
        return oculta @ p["W2"].T + p["b2"]


class CPN(Modelo):
    """*Classical Pyramid Network*: misma topología que la QCNN que controla, sin
    estados cuánticos. Por parche: ``β`` normalizado de la capa de entrada; cada
    "convolución" es ``tanh(x · θ)`` y cada pooling promedia pares; termina en 4
    valores y una capa lineal 4 → 10.

    10×10 (control de v1/v2): 10 → 8 (promedia (0,1) y (2,3)) → 4, dos capas tanh.
    16×16 (control de v3): 16 → 8 → 4 y tres capas tanh, como las tres
    convoluciones de v3.
    """

    def __init__(self, tam_imagen: int):
        self.tam_imagen = tam_imagen
        self.nombre = f"cpn_{tam_imagen}"
        if tam_imagen == 10:
            self.tam_parche, self.n_parches = 2, 10
            self.pools = [
                [(0, 1), (2, 3), *((i, None) for i in range(4, 10))],
                [(0, 1), (2, 3), (4, 5), (6, 7)],
            ]
            self.capas = [10, 8]
        elif tam_imagen == 16:
            self.tam_parche, self.n_parches = 4, 16
            self.pools = [[(i, i + 1) for i in range(0, 16, 2)], [(i, i + 1) for i in range(0, 8, 2)]]
            self.capas = [16, 8, 4]
        else:
            raise ValueError("La CPN solo está definida para 10×10 y 16×16.")

    def inicializar(self, clave):
        k = jax.random.split(clave, 3 + len(self.capas))
        entrenables = {
            "W": he(k[0], (2, self.tam_parche**2)),
            "b": bias_entrada(k[1]),
            "W_out": he(k[2], (N_CLASES, 4)),
            "b_out": jnp.zeros(N_CLASES),
        }
        for i, n in enumerate(self.capas):
            entrenables[f"theta{i + 1}"] = angular(k[3 + i], (n,))
        return entrenables, {}

    def salida(self, entrenables, fijos, x):
        p = entrenables
        v = self.parches(x) @ p["W"].T + p["b"]
        h = v[..., 1] / jnp.sqrt(jnp.sum(v**2, axis=-1) + EPS_NORMA)  # (B, n_parches)
        for i in range(len(self.capas)):
            h = jnp.tanh(h * p[f"theta{i + 1}"])
            if i < len(self.pools):
                h = _media_pares(h, self.pools[i])
        return h @ p["W_out"].T + p["b_out"]


class SoloClasico(Modelo):
    """Ablación de v3 sin circuito: la capa de entrada de v3 conectada directo a la de
    salida. Cada parche aporta ``cos(θ)``, el ⟨Z⟩ que daría su qubit justo después de
    la codificación, y una capa lineal 16 → 10 clasifica."""

    nombre = "v3_solo_clasico"
    tam_imagen, tam_parche, n_parches = 16, 4, 16

    def inicializar(self, clave):
        k1, k2, k3 = jax.random.split(clave, 3)
        return {
            "W": he(k1, (2, self.tam_parche**2)),
            "b": bias_entrada(k2),
            "W_out": he(k3, (N_CLASES, self.n_parches)),
            "b_out": jnp.zeros(N_CLASES),
        }, {}

    def salida(self, entrenables, fijos, x):
        p = entrenables
        z = jnp.cos(self.angulos(p["W"], p["b"], x))
        return z @ p["W_out"].T + p["b_out"]
