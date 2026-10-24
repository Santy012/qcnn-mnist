"""Figuras comunes de los notebooks: curvas por época, barras de accuracy y matrices de confusión.

Cada modelo tiene un color fijo en todas las figuras. Las tres versiones de la QCNN
van en una escala de azules de claro a oscuro (la evolución); los controles y las
ablaciones, en tonos categóricos distintos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from qcnn_mnist.experimento import DIRECTORIO_RESULTADOS
from qcnn_mnist.resultados import NOMBRES, Resumen, confusion_total, serie

DIRECTORIO_FIGURAS = DIRECTORIO_RESULTADOS / "figuras"

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"

COLORES = {
    "qcnn_v1": "#86b6ef",
    "qcnn_v2": "#2a78d6",
    "qcnn_v3": "#184f95",
    "cpn_10": "#eb6834",
    "cpn_16": "#eb6834",
    "cnn_10": "#1baf7a",
    "cnn_16": "#1baf7a",
    "v3_sin_entrelazamiento": "#6250d6",
    "v3_congelado": "#e87ba4",
    "v3_solo_clasico": "#008300",
}

AZULES = LinearSegmentedColormap.from_list("azules", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])

AZAR = 0.1  # accuracy de elegir una clase al azar


def aplicar_estilo() -> None:
    """Estilo común: fondo claro, ejes discretos, rejilla tenue, sin bordes superior y derecho."""
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": TINTA,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.titlecolor": TINTA,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": EJE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": REJILLA,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": TINTA_TENUE,
            "ytick.color": TINTA_TENUE,
            "xtick.labelcolor": TINTA_SECUNDARIA,
            "ytick.labelcolor": TINTA_SECUNDARIA,
            "legend.frameon": False,
            "legend.labelcolor": TINTA_SECUNDARIA,
            "lines.linewidth": 2,
        }
    )


def guardar(fig, nombre: str, directorio: Path = DIRECTORIO_FIGURAS) -> Path:
    """Guarda la figura como PNG en ``results/figuras/``."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight")
    return ruta


def curvas(ax, resultados: dict[str, list[dict]], modelos: list[str], clave: str, semillas: bool = True):
    """Media por época (línea) ± desviación (banda) de ``clave``; las semillas, en trazo fino.

    ``clave``: ``"perdida_train"`` o ``"acc_test_por_epoca"``. La accuracy se muestra en %.
    """
    escala = 100 if clave.startswith("acc") else 1
    for m in modelos:
        s = serie(resultados[m], clave) * escala
        epocas = np.arange(1, s.shape[1] + 1)
        color = COLORES[m]
        if semillas:
            for fila in s:
                ax.plot(epocas, fila, color=color, linewidth=0.8, alpha=0.35)
        media, desv = s.mean(0), s.std(0, ddof=1) if len(s) > 1 else np.zeros(s.shape[1])
        ax.fill_between(epocas, media - desv, media + desv, color=color, alpha=0.15, linewidth=0)
        ax.plot(epocas, media, color=color, label=NOMBRES[m])
        ax.annotate(
            NOMBRES[m], (epocas[-1], media[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=9, color=TINTA_SECUNDARIA,
        )
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy de prueba (%)" if escala == 100 else "Pérdida de entrenamiento")
    ax.set_xlim(1, None)
    ax.margins(x=0)
    coma_decimal(ax)
    if len(modelos) > 1:
        ax.legend(loc="best", fontsize=9)
    return ax


def barras(ax, resumenes: dict[str, Resumen], modelos: list[str], azar: bool = True):
    """Barras horizontales de accuracy de prueba (media ± desviación entre semillas), con valor."""
    y = np.arange(len(modelos))[::-1]
    medias = [100 * resumenes[m].acc_test_media for m in modelos]
    desvs = [100 * resumenes[m].acc_test_desv for m in modelos]
    ax.barh(
        y, medias, height=0.62, color=[COLORES[m] for m in modelos],
        edgecolor=SUPERFICIE, linewidth=2,
    )
    ax.errorbar(medias, y, xerr=desvs, fmt="none", ecolor=TINTA_SECUNDARIA, elinewidth=1.2, capsize=3)
    for yi, mu, sd in zip(y, medias, desvs):
        ax.text(mu + sd + 1.2, yi, f"{mu:.1f} ± {sd:.1f}".replace(".", ","), va="center", fontsize=9, color=TINTA)
    ax.set_yticks(y, [f"{NOMBRES[m]}  ({resumenes[m].n_parametros} p.)" for m in modelos])
    ax.set_xlabel("Accuracy de prueba (%), media ± desviación de 3 semillas")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    if azar:
        ax.axvline(100 * AZAR, color=TINTA_TENUE, linestyle="--", linewidth=1)
        ax.text(100 * AZAR + 0.8, y[-1] - 0.45, "azar", fontsize=8, color=TINTA_TENUE, va="top")
    return ax


def coma_decimal(ax, eje: str = "y") -> None:
    """Marcas del eje con coma decimal."""
    formato = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(formato)


def circuito(modelo: str, semilla: int = 42, escala: float = 0.45):
    """Dibuja el circuito de una QCNN con los parámetros entrenados de ``semilla``.

    La codificación se dibuja con ángulos nulos: el diagrama muestra la estructura.
    """
    import pennylane as qml

    from qcnn_mnist.modelos import construir_modelo
    from qcnn_mnist.resultados import cargar_parametros

    m = construir_modelo(modelo)
    entrenables, fijos = cargar_parametros(modelo, semilla)
    p = {**fijos, **entrenables}
    nombres = getattr(m, "CUANTICOS", ("theta1", "theta2"))
    fig, _ = qml.draw_mpl(m.circuito, decimals=None, style="pennylane")(
        np.zeros(m.n_qubits), *(p[n] for n in nombres)
    )
    ancho, alto = fig.get_size_inches()
    fig.set_size_inches(ancho * escala, alto * escala)
    return fig


def rejilla_parches(ax, imagen, tam_parche: int, usados: range | None = None, titulo: str | None = None):
    """Imagen con la rejilla de parches numerada; los parches fuera de ``usados`` se atenúan."""
    tam = imagen.shape[0]
    lado = tam // tam_parche
    ax.imshow(imagen, cmap="gray_r", vmin=0, vmax=1, extent=(0, tam, tam, 0))
    for k in range(lado * lado):
        f, c = divmod(k, lado)
        x, y = c * tam_parche, f * tam_parche
        if usados is not None and k not in usados:
            ax.add_patch(plt.Rectangle((x, y), tam_parche, tam_parche, color=SUPERFICIE, alpha=0.75, linewidth=0))
        ax.add_patch(plt.Rectangle((x, y), tam_parche, tam_parche, fill=False, edgecolor=COLORES["qcnn_v2"], linewidth=0.8))
        ax.text(x + 0.15 * tam_parche, y + 0.3 * tam_parche, str(k), fontsize=7, color=COLORES["qcnn_v3"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for lado_eje in ax.spines.values():
        lado_eje.set_visible(False)
    if titulo:
        ax.set_title(titulo, fontsize=10)
    return ax


def confusion(ax, corridas: list[dict], titulo: str | None = None):
    """Matriz de confusión sumada sobre las semillas, normalizada por fila (clase real)."""
    m = confusion_total(corridas).astype(float)
    filas = m / m.sum(1, keepdims=True)
    ax.imshow(filas, cmap=AZULES, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if filas[i, j] >= 0.05:
                color = SUPERFICIE if filas[i, j] > 0.55 else TINTA
                ax.text(j, i, f"{100 * filas[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.tick_params(length=0)
    if titulo:
        ax.set_title(titulo)
    return ax
