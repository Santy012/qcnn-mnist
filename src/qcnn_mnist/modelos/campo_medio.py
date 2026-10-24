"""Simulación de campo medio del circuito de v3: estado producto de qubits independientes.

Cada qubit se guarda como un vector de 2 amplitudes; el estado de ``n`` qubits es
siempre el producto de los ``n`` vectores, así que no puede haber entrelazamiento.
Las compuertas de un qubit son exactamente las de PennyLane (mismas matrices). Cada
CNOT(c, t) se sustituye por ``RX(π · p₁(c))`` sobre ``t``, con ``p₁(c) = (1 − ⟨Z_c⟩)/2``
calculado en ese punto del circuito: invierte ``t`` en la proporción en que ``c`` está
en |1⟩, sin crear correlaciones.

El estado tiene forma ``(B, n, 2)``: lote, qubit, amplitudes de |0⟩ y |1⟩.
"""

from __future__ import annotations

import jax.numpy as jnp


def estado_inicial(lote: int, n_qubits: int):
    """Todos los qubits en |0⟩."""
    return jnp.zeros((lote, n_qubits, 2), dtype=jnp.complex128).at[..., 0].set(1.0)


def matriz_ry(theta):
    """``RY(θ)`` como en PennyLane; ``theta`` escalar o ``(B,)`` → ``(..., 2, 2)``."""
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2).astype(jnp.complex128)


def matriz_rx(theta):
    """``RX(θ)`` como en PennyLane; ``theta`` escalar o ``(B,)`` → ``(..., 2, 2)``."""
    c, s = jnp.cos(theta / 2).astype(jnp.complex128), -1j * jnp.sin(theta / 2)
    return jnp.stack([jnp.stack([c, s], -1), jnp.stack([s, c], -1)], -2)


def matriz_rot(phi, theta, omega):
    """``Rot(φ, θ, ω) = RZ(ω) RY(θ) RZ(φ)`` como en PennyLane."""
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    suma, resta = (phi + omega) / 2, (phi - omega) / 2
    return jnp.array(
        [
            [jnp.exp(-1j * suma) * c, -jnp.exp(1j * resta) * s],
            [jnp.exp(-1j * resta) * s, jnp.exp(1j * suma) * c],
        ]
    )


def aplicar(estado, q: int, U):
    """Aplica la matriz ``U`` (``(2, 2)`` o ``(B, 2, 2)``) al qubit ``q``."""
    return estado.at[:, q].set((U @ estado[:, q, :, None])[..., 0])


def prob_uno(estado, q: int):
    """``p₁(q) = |⟨1|ψ_q⟩|²``, forma ``(B,)``."""
    return jnp.abs(estado[:, q, 1]) ** 2


def esperanza_z(estado, q: int):
    """``⟨Z_q⟩ = |a₀|² − |a₁|²``, forma ``(B,)``."""
    a = jnp.abs(estado[:, q]) ** 2
    return a[:, 0] - a[:, 1]


def cnot_campo_medio(estado, c: int, t: int):
    """Sustituto de CNOT(c, t): ``RX(π · p₁(c))`` sobre ``t``."""
    return aplicar(estado, t, matriz_rx(jnp.pi * prob_uno(estado, c)))


# --- bloques del circuito de v3, en el mismo orden que ``circuitos.py`` -----------------


def codificar(estado, angulos, wires):
    for i, w in enumerate(wires):
        estado = aplicar(estado, w, matriz_ry(angulos[:, i]))
    return estado


def anillo_cnot(estado, wires):
    n = len(wires)
    for i in range(n):
        estado = cnot_campo_medio(estado, wires[i], wires[(i + 1) % n])
    return estado


def conv_rot(estado, thetas, wires):
    for i, w in enumerate(wires):
        estado = aplicar(estado, w, matriz_rot(thetas[i, 0], thetas[i, 1], thetas[i, 2]))
    return anillo_cnot(estado, wires)


def pooling_parametrico(estado, params, wires):
    supervivientes = []
    for k in range(0, len(wires), 2):
        a, b = wires[k], wires[k + 1]
        estado = cnot_campo_medio(estado, a, b)
        estado = aplicar(estado, b, matriz_ry(params[k // 2]))
        supervivientes.append(b)
    return estado, supervivientes


def circuito_v3(angulos, theta1, pool1, theta2, pool2, theta3):
    """Circuito de v3 en campo medio. ``angulos``: ``(B, 16)``. Devuelve la lista de
    ⟨Z⟩ de los 4 qubits finales, como el circuito de PennyLane."""
    wires = list(range(angulos.shape[-1]))
    estado = estado_inicial(angulos.shape[0], len(wires))
    estado = codificar(estado, angulos, wires)
    estado = conv_rot(estado, theta1, wires)
    estado, s1 = pooling_parametrico(estado, pool1, wires)
    estado = codificar(estado, angulos[:, s1], s1)  # re-uploading
    estado = conv_rot(estado, theta2, s1)
    estado, s2 = pooling_parametrico(estado, pool2, s1)
    estado = conv_rot(estado, theta3, s2)
    return [esperanza_z(estado, w) for w in s2]
