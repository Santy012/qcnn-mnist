"""Carga y preprocesamiento de MNIST sin TensorFlow.

Replica el preprocesamiento de los notebooks originales: intensidades en [0, 1],
reducción bilineal a ``tam × tam`` y submuestra estratificada fija.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Mismo archivo que descarga ``keras.datasets.mnist.load_data()``.
URL_MNIST = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256_MNIST = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DIRECTORIO_DATOS = Path(__file__).resolve().parents[2] / "data"

SEMILLA_DATOS = 42
N_TRAIN = 2000
N_TEST = 500


@dataclass(frozen=True)
class Datos:
    """Partición de entrenamiento y prueba ya preprocesada."""

    x_train: np.ndarray  # (n_train, tam, tam), float64 en [0, 1]
    y_train: np.ndarray  # (n_train,), int32
    x_test: np.ndarray
    y_test: np.ndarray


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar_mnist(directorio: Path = DIRECTORIO_DATOS) -> Path:
    """Descarga ``mnist.npz`` si no existe y verifica su SHA-256."""
    ruta = Path(directorio) / "mnist.npz"
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = ruta.with_suffix(".part")
        urllib.request.urlretrieve(URL_MNIST, temporal)
        temporal.replace(ruta)
    if _sha256(ruta) != SHA256_MNIST:
        raise ValueError(f"SHA-256 inesperado en {ruta}; borre el archivo y vuelva a descargarlo.")
    return ruta


def cargar_mnist(directorio: Path = DIRECTORIO_DATOS):
    """Devuelve ``(x_train, y_train, x_test, y_test)`` originales (28×28, uint8)."""
    with np.load(descargar_mnist(directorio)) as d:
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def _pesos_bilineales(n_entrada: int, n_salida: int) -> np.ndarray:
    """Matriz ``(n_salida, n_entrada)`` de interpolación lineal en una dimensión.

    Usa centros de píxel a la mitad, ``src = (dst + 0.5) · n_entrada / n_salida − 0.5``,
    con los bordes recortados y sin antialias: el comportamiento por defecto de
    ``tf.image.resize(..., method="bilinear")``.
    """
    escala = n_entrada / n_salida
    src = np.clip((np.arange(n_salida) + 0.5) * escala - 0.5, 0.0, n_entrada - 1)
    izq = np.floor(src).astype(int)
    der = np.minimum(izq + 1, n_entrada - 1)
    frac = src - izq
    pesos = np.zeros((n_salida, n_entrada))
    filas = np.arange(n_salida)
    np.add.at(pesos, (filas, izq), 1.0 - frac)
    np.add.at(pesos, (filas, der), frac)
    return pesos


def redimensionar_bilineal(imagenes: np.ndarray, tam: int) -> np.ndarray:
    """Redimensiona imágenes ``(..., alto, ancho)`` a ``(..., tam, tam)`` con interpolación bilineal."""
    imagenes = np.asarray(imagenes, dtype=np.float64)
    filas = _pesos_bilineales(imagenes.shape[-2], tam)
    columnas = _pesos_bilineales(imagenes.shape[-1], tam)
    return filas @ imagenes @ columnas.T


def particion_estratificada(etiquetas: np.ndarray, n: int, semilla: int) -> np.ndarray:
    """Índices de una submuestra de tamaño ``n`` que conserva la proporción de cada clase.

    El redondeo se reparte por el método del resto mayor. Los índices se devuelven
    en orden aleatorio, no agrupados por clase.
    """
    etiquetas = np.asarray(etiquetas)
    rng = np.random.default_rng(semilla)
    clases, conteos = np.unique(etiquetas, return_counts=True)
    cuota = conteos / conteos.sum() * n
    por_clase = np.floor(cuota).astype(int)
    faltan = n - por_clase.sum()
    por_clase[np.argsort(-(cuota - por_clase), kind="stable")[:faltan]] += 1
    indices = [
        rng.choice(np.flatnonzero(etiquetas == c), size=k, replace=False)
        for c, k in zip(clases, por_clase)
    ]
    return rng.permutation(np.concatenate(indices))


def preparar_datos(
    tam_imagen: int,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    semilla: int = SEMILLA_DATOS,
    directorio: Path = DIRECTORIO_DATOS,
) -> Datos:
    """Partición fija y preprocesada: normaliza a [0, 1] y luego reduce a ``tam_imagen``.

    La elección de imágenes depende solo de ``semilla``, así que es la misma para
    cualquier ``tam_imagen``.
    """
    x_train, y_train, x_test, y_test = cargar_mnist(directorio)
    i_train = particion_estratificada(y_train, n_train, semilla)
    i_test = particion_estratificada(y_test, n_test, semilla)

    def procesar(x):
        return redimensionar_bilineal(x.astype(np.float64) / 255.0, tam_imagen)

    return Datos(
        x_train=procesar(x_train[i_train]),
        y_train=y_train[i_train].astype(np.int32),
        x_test=procesar(x_test[i_test]),
        y_test=y_test[i_test].astype(np.int32),
    )
