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
    "v2_lectura_z": "v2 con lectura ⟨Z⟩",
    "qcnn_v3": "QCNN v3",
    "v3_congelado": "v3 con circuito congelado",
    "v3_solo_clasico": "v3 solo clásico",
    "v3_campo_medio": "v3 en campo medio",
    "v3_sin_entrelazamiento": "v3 sin interacción entre qubits",
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
    "v2_lectura_z": "ablacion",
    "v3_campo_medio": "ablacion",
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


def tabla_ganancias(resumenes: dict[str, Resumen], pasos: list[tuple[str, str, str]]) -> str:
    """Tabla Markdown de la ganancia media de accuracy de prueba en cada paso ``(de, a, qué cambia)``."""
    filas = ["| Paso | Ganancia (puntos) | Qué cambia |", "| --- | ---: | --- |"]
    for a, b, cambia in pasos:
        g = 100 * (resumenes[b].acc_test_media - resumenes[a].acc_test_media)
        filas.append(f"| {resumenes[a].nombre} → {resumenes[b].nombre} | {num(g, signo=True)} | {cambia} |")
    return "\n".join(filas)


# Pasos que separan el efecto de la lectura del de la escala al pasar de v2 a v3.
PASOS_V2_V3 = [
    ("qcnn_v2", "v2_lectura_z", "solo la lectura (⟨Z⟩ + capa lineal)"),
    ("v2_lectura_z", "qcnn_v3", "resolución, parches, qubits y pooling"),
    ("qcnn_v2", "qcnn_v3", "las dos cosas"),
    ("cpn_10", "cpn_16", "resolución y parches"),
    ("cnn_10", "cnn_16", "resolución y parches"),
]
EVOLUCION = ["qcnn_v1", "qcnn_v2", "v2_lectura_z", "qcnn_v3"]
# Orden también elegido para que los colores vecinos se distingan con daltonismo.
ABLACIONES = ["v3_congelado", "v3_solo_clasico", "v3_campo_medio", "v3_sin_entrelazamiento", "cpn_16"]
# Frente a «sin interacción»: cuánto se recupera con flujo de información sin entrelazamiento
# (campo medio) y con entrelazamiento (v3).
ENTRELAZAMIENTO = ["v3_campo_medio", "qcnn_v3"]


def tabla_diferencias(resultados: dict[str, list[dict]], referencia: str, modelos: list[str]) -> str:
    """Tabla Markdown de la diferencia de accuracy de prueba de cada modelo con ``referencia``,
    en puntos porcentuales: media y por semilla (emparejadas por número de semilla)."""
    ref = {c["semilla"]: c["acc_test"] for c in resultados[referencia]}
    semillas = sorted(ref)
    filas = [
        f"| Modelo | Parámetros | Diferencia con {NOMBRES[referencia]} (puntos) | Por semilla ({' / '.join(map(str, semillas))}) |",
        "| --- | ---: | ---: | --- |",
    ]
    for m in modelos:
        r = resumir(resultados[m])
        acc = {c["semilla"]: c["acc_test"] for c in resultados[m]}
        d = np.array([100 * (acc[s] - ref[s]) for s in semillas])
        por_semilla = " / ".join(num(x, signo=True) for x in d)
        filas.append(f"| {r.nombre} | {r.n_parametros} | {num(d.mean(), signo=True)} | {por_semilla} |")
    return "\n".join(filas)


def cargar_parametros(modelo: str, semilla: int, directorio: Path = DIRECTORIO_RESULTADOS) -> tuple[dict, dict]:
    """Parámetros finales guardados: ``(entrenables, fijos)``."""
    with np.load(Path(directorio) / modelo / f"semilla_{semilla}_params.npz") as f:
        entrenables = {k.split(".", 1)[1]: f[k] for k in f.files if k.startswith("entrenable.")}
        fijos = {k.split(".", 1)[1]: f[k] for k in f.files if k.startswith("fijo.")}
    return entrenables, fijos
