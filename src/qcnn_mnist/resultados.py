"""Lectura y resumen de los resultados guardados en ``results/``.

Los notebooks y el README obtienen sus cifras de aquí; ninguna se escribe a mano.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qcnn_mnist.experimento import DIRECTORIO_RESULTADOS

# Orden de presentación y nombre legible de cada modelo.
NOMBRES = {
    "qcnn_v1": "QCNN v1",
    "qcnn_v2": "QCNN v2",
    "qcnn_v3": "QCNN v3",
    "v3_sin_entrelazamiento": "v3 sin interacción entre qubits",
    "v3_congelado": "v3 con circuito congelado",
    "v3_solo_clasico": "v3 solo clásico",
    "cnn_10": "CNN 10×10",
    "cpn_10": "CPN 10×10",
    "cnn_16": "CNN 16×16",
    "cpn_16": "CPN 16×16",
}
ORDEN = list(NOMBRES)

# Familia de cada modelo: decide el color en las figuras.
FAMILIAS = {
    "qcnn_v1": "qcnn",
    "qcnn_v2": "qcnn",
    "qcnn_v3": "qcnn",
    "v3_sin_entrelazamiento": "ablacion",
    "v3_congelado": "ablacion",
    "v3_solo_clasico": "ablacion",
    "cnn_10": "cnn",
    "cnn_16": "cnn",
    "cpn_10": "cpn",
    "cpn_16": "cpn",
}


@dataclass(frozen=True)
class Resumen:
    """Agregado de las semillas de un modelo. Desviaciones muestrales (``ddof = 1``)."""

    modelo: str
    semillas: tuple[int, ...]
    n_parametros: int
    acc_test_media: float
    acc_test_desv: float
    acc_train_media: float
    acc_train_desv: float
    horas: float  # cómputo total de las semillas

    @property
    def nombre(self) -> str:
        return NOMBRES.get(self.modelo, self.modelo)


def cargar(directorio: Path = DIRECTORIO_RESULTADOS) -> dict[str, list[dict]]:
    """Modelo → lista de resultados (uno por semilla, ordenados por semilla)."""
    resultados: dict[str, list[dict]] = {}
    for ruta in sorted(Path(directorio).glob("*/semilla_*.json")):
        d = json.loads(ruta.read_text(encoding="utf-8"))
        resultados.setdefault(d["modelo"], []).append(d)
    for lista in resultados.values():
        lista.sort(key=lambda d: d["semilla"])
    return dict(sorted(resultados.items(), key=lambda kv: _posicion(kv[0])))


def _posicion(modelo: str) -> int:
    return ORDEN.index(modelo) if modelo in ORDEN else len(ORDEN)


def _desv(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def resumir(corridas: list[dict]) -> Resumen:
    """Resume las semillas de un modelo."""
    test = np.array([c["acc_test"] for c in corridas])
    train = np.array([c["acc_train"] for c in corridas])
    parametros = {c["n_parametros"] for c in corridas}
    if len(parametros) != 1:
        raise ValueError(f"Conteos de parámetros distintos entre semillas: {parametros}")
    return Resumen(
        modelo=corridas[0]["modelo"],
        semillas=tuple(c["semilla"] for c in corridas),
        n_parametros=parametros.pop(),
        acc_test_media=float(test.mean()),
        acc_test_desv=_desv(test),
        acc_train_media=float(train.mean()),
        acc_train_desv=_desv(train),
        horas=sum(c["segundos"] for c in corridas) / 3600,
    )


def resumen(resultados: dict[str, list[dict]]) -> dict[str, Resumen]:
    return {m: resumir(c) for m, c in resultados.items()}


def serie(corridas: list[dict], clave: str) -> np.ndarray:
    """Matriz ``(semillas, épocas)`` de una serie por época (``perdida_train``, ``acc_test_por_epoca``)."""
    return np.array([c[clave] for c in corridas], dtype=float)


def confusion_total(corridas: list[dict]) -> np.ndarray:
    """Suma de las matrices de confusión de las semillas (filas: clase real)."""
    return np.sum([np.array(c["matriz_confusion"]) for c in corridas], axis=0)


def error_estandar(p: float, n: int) -> float:
    """Error estándar binomial de una accuracy ``p`` medida sobre ``n`` imágenes."""
    return math.sqrt(p * (1 - p) / n)


def num(x: float, decimales: int = 1, signo: bool = False) -> str:
    """Número con coma decimal: ``2.4712 → '2,5'``; con ``signo``, ``'+2,5'``."""
    return f"{x:{'+' if signo else ''}.{decimales}f}".replace(".", ",")


def pct(x: float, decimales: int = 1) -> str:
    """Porcentaje con coma decimal: ``0.4333 → '43,3'``."""
    return num(100 * x, decimales)


def media_desv(r: Resumen) -> str:
    """``'43,3 ± 3,0'`` (en puntos porcentuales)."""
    return f"{pct(r.acc_test_media)} ± {pct(r.acc_test_desv)}"


def tabla_markdown(resumenes: dict[str, Resumen], modelos: list[str] | None = None) -> str:
    """Tabla Markdown: modelo, parámetros, accuracy de prueba y de entrenamiento, horas."""
    filas = [
        "| Modelo | Parámetros | Accuracy de prueba (%) | Accuracy de entrenamiento (%) | Cómputo (h) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for m in modelos or list(resumenes):
        r = resumenes[m]
        filas.append(
            f"| {r.nombre} | {r.n_parametros} | {media_desv(r)} "
            f"| {pct(r.acc_train_media)} ± {pct(r.acc_train_desv)} | {f'{r.horas:.2f}'.replace('.', ',')} |"
        )
    return "\n".join(filas)


def cargar_parametros(modelo: str, semilla: int, directorio: Path = DIRECTORIO_RESULTADOS) -> tuple[dict, dict]:
    """Parámetros finales guardados: ``(entrenables, fijos)``."""
    with np.load(Path(directorio) / modelo / f"semilla_{semilla}_params.npz") as f:
        entrenables = {k.split(".", 1)[1]: f[k] for k in f.files if k.startswith("entrenable.")}
        fijos = {k.split(".", 1)[1]: f[k] for k in f.files if k.startswith("fijo.")}
    return entrenables, fijos
