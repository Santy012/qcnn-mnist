"""Motor de entrenamiento común a todos los modelos (Adam, minilotes, JIT)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qcnn_mnist.datos import Datos
from qcnn_mnist.evaluacion import accuracy, predecir_por_bloques
from qcnn_mnist.modelos import Modelo


@dataclass
class Historial:
    perdida_train: list[float] = field(default_factory=list)
    acc_test: list[float] = field(default_factory=list)
    segundos_epoca: list[float] = field(default_factory=list)


def entrenar(
    modelo: Modelo,
    datos: Datos,
    lr: float,
    epocas: int,
    lote: int,
    semilla: int,
    informar=print,
):
    """Entrena ``modelo`` y devuelve ``(entrenables, fijos, historial)``.

    La semilla fija la inicialización y el orden de los minilotes (una permutación
    por época). Se usan ``n_train // lote`` lotes completos por época.
    """
    entrenables, fijos = modelo.inicializar(jax.random.PRNGKey(semilla))
    optimizador = optax.adam(lr)
    estado = optimizador.init(entrenables)

    @jax.jit
    def paso(entrenables, estado, x, y):
        perdida, grads = jax.value_and_grad(modelo.perdida)(entrenables, fijos, x, y)
        cambios, estado = optimizador.update(grads, estado, entrenables)
        return optax.apply_updates(entrenables, cambios), estado, perdida

    predecir = jax.jit(modelo.predecir)
    x_train, y_train = jnp.asarray(datos.x_train), jnp.asarray(datos.y_train)
    x_test = jnp.asarray(datos.x_test)
    rng = np.random.default_rng(semilla)
    n_lotes = len(x_train) // lote
    historial = Historial()

    for epoca in range(epocas):
        t0 = time.perf_counter()
        orden = rng.permutation(len(x_train))
        perdidas = []
        for k in range(n_lotes):
            idx = orden[k * lote : (k + 1) * lote]
            entrenables, estado, perdida = paso(entrenables, estado, x_train[idx], y_train[idx])
            perdidas.append(perdida)
        historial.perdida_train.append(float(np.mean(jax.device_get(perdidas))))
        y_pred = predecir_por_bloques(predecir, entrenables, fijos, x_test)
        historial.acc_test.append(accuracy(datos.y_test, y_pred))
        historial.segundos_epoca.append(time.perf_counter() - t0)
        informar(
            f"  época {epoca + 1:2d}/{epocas}  pérdida {historial.perdida_train[-1]:.4f}  "
            f"acc test {historial.acc_test[-1]:.3f}  ({historial.segundos_epoca[-1]:.1f} s)"
        )
    return entrenables, fijos, historial
