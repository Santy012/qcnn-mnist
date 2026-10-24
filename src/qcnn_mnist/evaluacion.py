"""Métricas de evaluación."""

from __future__ import annotations

import numpy as np

from qcnn_mnist.modelos.base import N_CLASES

BLOQUE_EVALUACION = 100


def predecir_por_bloques(prediccion, entrenables, fijos, x, bloque: int = BLOQUE_EVALUACION):
    """Aplica ``prediccion(entrenables, fijos, x_bloque)`` por bloques para acotar memoria."""
    return np.concatenate(
        [np.asarray(prediccion(entrenables, fijos, x[i : i + bloque])) for i in range(0, len(x), bloque)]
    )


def accuracy(y_real, y_pred) -> float:
    return float(np.mean(np.asarray(y_real) == np.asarray(y_pred)))


def matriz_confusion(y_real, y_pred, n_clases: int = N_CLASES) -> np.ndarray:
    """Filas: clase real; columnas: clase predicha."""
    m = np.zeros((n_clases, n_clases), dtype=int)
    np.add.at(m, (np.asarray(y_real), np.asarray(y_pred)), 1)
    return m
