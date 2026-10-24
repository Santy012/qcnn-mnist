"""Corre experimentos a partir de configuraciones y guarda los resultados en ``results/``.

Ejemplos:
    python scripts/run.py configs/qcnn_v3.toml
    python scripts/run.py --todas
    python scripts/run.py --todas --n-train 64 --n-test 32 --epocas 1 --semillas 42 --salida /tmp/prueba

Lo que ya tiene resultado guardado se salta (la corrida es reanudable) salvo con ``--forzar``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

from qcnn_mnist.experimento import DIRECTORIO_RESULTADOS, RAIZ, Configuracion, correr, ruta_resultado

# De lo más barato a lo más caro: controles clásicos, v1, v2, v3 y ablaciones.
ORDEN = [
    "cnn_10",
    "cpn_10",
    "cnn_16",
    "cpn_16",
    "qcnn_v1",
    "qcnn_v2",
    "v2_lectura_z",
    "qcnn_v3",
    "v3_solo_clasico",
    "v3_campo_medio",
    "v3_congelado",
    "v3_sin_entrelazamiento",
]


def evitar_suspension() -> None:
    """En Windows, pide que el equipo no se suspenda por inactividad mientras el
    proceso corre. Se libera sola al terminar. No impide la suspensión al cerrar la tapa."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def informar(mensaje: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {mensaje}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", type=Path, help="archivos TOML de configs/")
    parser.add_argument("--todas", action="store_true", help="todas las configuraciones en el orden del estudio")
    parser.add_argument("--semillas", type=int, nargs="+")
    parser.add_argument("--epocas", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--dispositivo", help="simulador de PennyLane (sustituye al de la configuración)")
    parser.add_argument("--diff-method")
    parser.add_argument("--salida", type=Path, default=DIRECTORIO_RESULTADOS)
    parser.add_argument("--forzar", action="store_true", help="repite lo que ya tiene resultado")
    args = parser.parse_args()

    rutas = [RAIZ / "configs" / f"{m}.toml" for m in ORDEN] if args.todas else args.configs
    if not rutas:
        parser.error("indique archivos de configuración o --todas")
    configs = [
        Configuracion.desde_toml(r).con(
            epocas=args.epocas,
            n_train=args.n_train,
            n_test=args.n_test,
            dispositivo=args.dispositivo,
            diff_method=args.diff_method,
            semillas=tuple(args.semillas) if args.semillas else None,
        )
        for r in rutas
    ]

    evitar_suspension()
    # Semilla por semilla: si la corrida se corta, queda al menos una semilla de todo.
    semillas = sorted({s for c in configs for s in c.semillas})
    errores = 0
    for semilla in semillas:
        for config in configs:
            if semilla not in config.semillas:
                continue
            if ruta_resultado(args.salida, config.modelo, semilla).exists() and not args.forzar:
                informar(f"{config.modelo} · semilla {semilla}: ya existe, se salta")
                continue
            try:
                correr(config, semilla, args.salida, informar=informar)
            except Exception:
                errores += 1
                detalle = traceback.format_exc()
                informar(f"ERROR en {config.modelo} · semilla {semilla}\n{detalle}")
                args.salida.mkdir(parents=True, exist_ok=True)
                with open(args.salida / "errores.log", "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {config.modelo} · semilla {semilla}\n{detalle}\n")
    informar(f"Terminado con {errores} errores.")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
