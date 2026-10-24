"""Inserta en el README las tablas de resultados generadas a partir de ``results/``.

Cada tabla vive entre los marcadores ``<!-- resultados:NOMBRE -->`` y
``<!-- /resultados:NOMBRE -->``; el script reescribe solo lo que hay entre ellos.

Uso:
    python scripts/actualizar_readme.py              reescribe las tablas
    python scripts/actualizar_readme.py --comprobar  falla si el README no está al día
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from qcnn_mnist import resultados as res
from qcnn_mnist.experimento import RAIZ

README = RAIZ / "README.md"


def bloques() -> dict[str, str]:
    """Contenido generado de cada bloque del README."""
    r = res.cargar()
    s = res.resumen(r)
    return {
        "todos": res.tabla_markdown(s, res.ORDEN),
        "v2_v3": res.tabla_ganancias(s, res.PASOS_V2_V3),
        "ablaciones": res.tabla_diferencias(r, "qcnn_v3", res.ABLACIONES),
        "entrelazamiento": res.tabla_diferencias(r, "v3_sin_entrelazamiento", res.ENTRELAZAMIENTO),
    }


def actualizar(texto: str, contenido: dict[str, str]) -> str:
    """Sustituye el interior de cada bloque. Falla si falta algún marcador."""
    for nombre, tabla in contenido.items():
        patron = re.compile(rf"(<!-- resultados:{nombre} -->\n)(?:.*?\n)?(<!-- /resultados:{nombre} -->)", re.S)
        if not patron.search(texto):
            raise ValueError(f"Falta el bloque «{nombre}» en el README.")
        texto = patron.sub(lambda m: m.group(1) + tabla + "\n" + m.group(2), texto)
    return texto


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comprobar", action="store_true", help="no escribe; falla si el README no está al día")
    args = parser.parse_args()

    actual = README.read_text(encoding="utf-8")
    nuevo = actualizar(actual, bloques())
    if args.comprobar:
        if nuevo != actual:
            print("El README no coincide con results/. Ejecutar: python scripts/actualizar_readme.py")
            return 1
        print("README al día.")
        return 0
    README.write_text(nuevo, encoding="utf-8", newline="\n")
    print("README actualizado." if nuevo != actual else "README sin cambios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
