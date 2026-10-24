"""Un experimento = una configuración × una semilla: entrena, evalúa y guarda en ``results/``."""

from __future__ import annotations

import json
import os
import platform
import time
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np

from qcnn_mnist.datos import N_TEST, N_TRAIN, preparar_datos
from qcnn_mnist.entrenamiento import entrenar
from qcnn_mnist.evaluacion import accuracy, matriz_confusion, predecir_por_bloques
from qcnn_mnist.modelos import construir_modelo

RAIZ = Path(__file__).resolve().parents[2]
DIRECTORIO_RESULTADOS = RAIZ / "results"


@dataclass(frozen=True)
class Configuracion:
    modelo: str
    lr: float = 0.01
    epocas: int = 30
    lote: int = 16
    semillas: tuple[int, ...] = (42, 43, 44)
    n_train: int = N_TRAIN
    n_test: int = N_TEST
    dispositivo: str = "default.qubit"
    diff_method: str = "best"
    extra: dict = field(default_factory=dict)

    @classmethod
    def desde_toml(cls, ruta: Path) -> "Configuracion":
        with open(ruta, "rb") as f:
            c = tomllib.load(f)
        e, s = c.get("entrenamiento", {}), c.get("simulador", {})
        return cls(
            modelo=c["modelo"],
            lr=e.get("lr", 0.01),
            epocas=e.get("epocas", 30),
            lote=e.get("lote", 16),
            semillas=tuple(e.get("semillas", (42, 43, 44))),
            n_train=e.get("n_train", N_TRAIN),
            n_test=e.get("n_test", N_TEST),
            dispositivo=s.get("dispositivo", "default.qubit"),
            diff_method=s.get("diff_method", "best"),
        )

    def con(self, **cambios) -> "Configuracion":
        return replace(self, **{k: v for k, v in cambios.items() if v is not None})


def ruta_resultado(directorio: Path, modelo: str, semilla: int) -> Path:
    return Path(directorio) / modelo / f"semilla_{semilla}.json"


def _escribir_atomico(ruta: Path, escribir) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    escribir(temporal)
    os.replace(temporal, ruta)


def _versiones() -> dict:
    paquetes = ["jax", "pennylane", "pennylane-lightning", "optax", "numpy"]
    return {"python": platform.python_version(), **{p: version(p) for p in paquetes}}


def correr(config: Configuracion, semilla: int, directorio: Path = DIRECTORIO_RESULTADOS, informar=print) -> dict:
    """Entrena una semilla de una configuración y guarda JSON y parámetros finales."""
    modelo = construir_modelo(config.modelo, dispositivo=config.dispositivo, diff_method=config.diff_method)
    datos = preparar_datos(modelo.tam_imagen, n_train=config.n_train, n_test=config.n_test)
    informar(f"{config.modelo} · semilla {semilla} · {datos.x_train.shape[0]}/{datos.x_test.shape[0]} imágenes")

    t0 = time.perf_counter()
    entrenables, fijos, historial = entrenar(
        modelo, datos, config.lr, config.epocas, config.lote, semilla, informar=informar
    )
    segundos = time.perf_counter() - t0

    predecir = jax.jit(modelo.predecir)
    pred_train = predecir_por_bloques(predecir, entrenables, fijos, datos.x_train)
    pred_test = predecir_por_bloques(predecir, entrenables, fijos, datos.x_test)

    resultado = {
        "modelo": config.modelo,
        "semilla": semilla,
        "n_parametros": modelo.contar_parametros(entrenables),
        "configuracion": {
            "lr": config.lr,
            "epocas": config.epocas,
            "lote": config.lote,
            "n_train": config.n_train,
            "n_test": config.n_test,
            "tam_imagen": modelo.tam_imagen,
            "tam_parche": modelo.tam_parche,
            "n_parches": modelo.n_parches,
            "dispositivo": config.dispositivo,
            "diff_method": config.diff_method,
        },
        "perdida_train": historial.perdida_train,
        "acc_test_por_epoca": historial.acc_test,
        "segundos_epoca": historial.segundos_epoca,
        "acc_train": accuracy(datos.y_train, pred_train),
        "acc_test": accuracy(datos.y_test, pred_test),
        "matriz_confusion": matriz_confusion(datos.y_test, pred_test).tolist(),
        "segundos": segundos,
        "versiones": _versiones(),
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }

    ruta = ruta_resultado(directorio, config.modelo, semilla)
    parametros = {f"entrenable.{k}": np.asarray(v) for k, v in entrenables.items()}
    parametros |= {f"fijo.{k}": np.asarray(v) for k, v in fijos.items()}

    def guardar_npz(p):
        with open(p, "wb") as f:  # cerrado antes de renombrar (Windows no renombra archivos abiertos)
            np.savez(f, **parametros)

    _escribir_atomico(ruta.with_name(f"semilla_{semilla}_params.npz"), guardar_npz)
    _escribir_atomico(ruta, lambda p: p.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"))
    informar(f"  → acc train {resultado['acc_train']:.3f} · acc test {resultado['acc_test']:.3f} · {segundos:.0f} s")
    return resultado
