"""Interfaz común de los modelos e inicializaciones compartidas."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import optax

from qcnn_mnist.codificacion import amplitudes, angulo, banda_central, extraer_parches

N_CLASES = 10
EPS_LOG = 1e-9
BIAS_ENTRADA = jnp.array([1e-3, 0.0])


def he(clave, forma):
    """Normal de He: ``σ = √(2 / fan_in)``, con ``fan_in`` la última dimensión."""
    return jax.random.normal(clave, forma) * jnp.sqrt(2.0 / forma[-1])


def angular(clave, forma, sigma: float = 0.1):
    """Ángulos de compuertas: normal con media 0 y desviación ``sigma``."""
    return jax.random.normal(clave, forma) * sigma


def bias_entrada(clave):
    """Bias entrenable de la capa de entrada, cerca del bias fijo de v1."""
    return BIAS_ENTRADA + jax.random.normal(clave, (2,)) * 0.01


class Modelo(ABC):
    """Modelo de clasificación de 10 clases sobre imágenes ``tam_imagen × tam_imagen``.

    Los parámetros se dividen en dos diccionarios: ``entrenables`` (reciben gradiente
    y se cuentan) y ``fijos`` (constantes, por ejemplo un circuito congelado).
    """

    nombre: str
    tam_imagen: int
    tam_parche: int
    n_parches: int  # parches que usa el modelo (banda central si son menos que la rejilla)

    @abstractmethod
    def inicializar(self, clave) -> tuple[dict, dict]:
        """Devuelve ``(entrenables, fijos)``."""

    @abstractmethod
    def salida(self, entrenables: dict, fijos: dict, x):
        """Salida del modelo para un lote ``x`` de forma ``(B, tam, tam)``: ``(B, k)``."""

    def perdida(self, entrenables, fijos, x, y):
        """Entropía cruzada media sobre logits."""
        logits = self.salida(entrenables, fijos, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    def predecir(self, entrenables, fijos, x):
        return jnp.argmax(self.salida(entrenables, fijos, x)[:, :N_CLASES], axis=-1)

    @staticmethod
    def contar_parametros(entrenables: dict) -> int:
        return int(sum(p.size for p in jax.tree_util.tree_leaves(entrenables)))

    # --- utilidades compartidas -------------------------------------------------

    def parches(self, x):
        """Parches que usa el modelo: ``(B, n_parches, tam_parche²)``."""
        p = extraer_parches(x, self.tam_parche)
        return p if p.shape[-2] == self.n_parches else banda_central(p, self.n_parches)

    def angulos(self, W, b, x):
        """Ángulo de codificación de cada parche: ``(B, n_parches)``."""
        return angulo(amplitudes(W, b, self.parches(x)))


class LecturaProbabilidades(Modelo):
    """Lectura por probabilidades: 16 estados de 4 qubits, los 10 primeros son las clases.

    La pérdida penaliza la masa en los 6 estados sobrantes (*ghost states*) con peso λ.
    """

    lam: float

    def perdida(self, entrenables, fijos, x, y):
        p = self.salida(entrenables, fijos, x)
        p_correcta = jnp.take_along_axis(p, y[:, None], axis=-1)[:, 0]
        fantasma = p[:, N_CLASES:].sum(axis=-1)
        return jnp.mean(-jnp.log(p_correcta + EPS_LOG) + self.lam * fantasma)
