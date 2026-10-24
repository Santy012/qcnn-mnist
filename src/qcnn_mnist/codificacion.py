"""Codificación de parches: extracción, capa lineal con normalización L2 y ángulo de RY."""

from __future__ import annotations

import math

import jax.numpy as jnp

EPS_NORMA = 1e-8


def extraer_parches(imagenes, tam_parche: int):
    """Divide imágenes ``(..., tam, tam)`` en parches no solapados ``tam_parche × tam_parche``.

    Devuelve ``(..., n_parches, tam_parche²)`` con los parches en orden por filas
    (de izquierda a derecha y de arriba abajo) y cada parche aplanado por filas.
    """
    *lote, alto, ancho = imagenes.shape
    if alto % tam_parche or ancho % tam_parche:
        raise ValueError(f"La imagen {alto}×{ancho} no se divide en parches de {tam_parche}.")
    nf, nc = alto // tam_parche, ancho // tam_parche
    x = imagenes.reshape(*lote, nf, tam_parche, nc, tam_parche)
    x = jnp.moveaxis(x, -3, -2)  # (..., nf, nc, tam_parche, tam_parche)
    return x.reshape(*lote, nf * nc, tam_parche * tam_parche)


def banda_central(parches, n: int):
    """Selecciona ``n`` parches de filas completas centradas verticalmente en la rejilla.

    ``parches`` tiene forma ``(..., lado², d)`` en orden por filas. Con una rejilla de
    5×5 y ``n = 10`` devuelve las filas 1 y 2 (parches 5 a 14).
    """
    lado = math.isqrt(parches.shape[-2])
    if lado * lado != parches.shape[-2] or n % lado:
        raise ValueError(f"{n} parches no forman filas completas de una rejilla {lado}×{lado}.")
    inicio = (lado - n // lado) // 2 * lado
    return parches[..., inicio : inicio + n, :]


def amplitudes(W, b, parche):
    """Comprime un parche en un par ``(α, β)`` de norma 1: ``v = W·p + b``, ``v / ‖v‖``.

    ``W`` tiene forma ``(2, d)`` y ``b`` forma ``(2,)``. Admite lotes de parches
    ``(..., d)``. El ε dentro de la raíz evita dividir entre cero.
    """
    v = parche @ W.T + b
    return v / jnp.sqrt(jnp.sum(v**2, axis=-1, keepdims=True) + EPS_NORMA)


def angulo(amp):
    """Ángulo ``θ = 2·arctan2(β, α)`` tal que ``RY(θ)|0⟩ = α|0⟩ + β|1⟩``.

    En ``α = β = 0`` el gradiente de ``arctan2`` es NaN. Se usa el patrón de doble
    ``where``: en ese punto se evalúa ``arctan2`` en ``(0, 1)`` y se devuelve 0, de
    modo que tanto el valor como el gradiente quedan en 0.
    """
    alfa, beta = amp[..., 0], amp[..., 1]
    definido = (alfa != 0) | (beta != 0)
    alfa_seguro = jnp.where(definido, alfa, 1.0)
    beta_seguro = jnp.where(definido, beta, 0.0)
    return jnp.where(definido, 2.0 * jnp.arctan2(beta_seguro, alfa_seguro), 0.0)
