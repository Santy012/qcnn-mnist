"""Mide el costo de entrenamiento por época de cada modelo y simulador, y estima el total.

Para cada combinación compila el paso de Adam (2 pasos de calentamiento), mide
``--pasos`` pasos de lote 16 y un bloque de evaluación de 100 imágenes. La época
estimada es ``n_train / lote`` pasos más ``n_test / 100`` bloques de evaluación.

    python scripts/medir_tiempos.py
    python scripts/medir_tiempos.py --modelos qcnn_v3 --simuladores default.qubit:best lightning.qubit:adjoint
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qcnn_mnist.datos import N_TEST, N_TRAIN, preparar_datos
from qcnn_mnist.evaluacion import BLOQUE_EVALUACION
from qcnn_mnist.modelos import CONSTRUCTORES, construir_modelo

CON_CIRCUITO = ["qcnn_v1", "qcnn_v2", "qcnn_v3", "v3_congelado", "v3_sin_entrelazamiento"]
LOTE, EPOCAS, SEMILLAS = 16, 30, 3


def _bloquear(arbol):
    return jax.tree_util.tree_map(lambda a: a.block_until_ready(), arbol)


def medir(nombre: str, dispositivo: str, diff_method: str, n_pasos: int, datos) -> dict:
    modelo = construir_modelo(nombre, dispositivo=dispositivo, diff_method=diff_method)
    entrenables, fijos = modelo.inicializar(jax.random.PRNGKey(42))
    opt = optax.adam(0.01)
    estado = opt.init(entrenables)

    @jax.jit
    def paso(entrenables, estado, x, y):
        perdida, grads = jax.value_and_grad(modelo.perdida)(entrenables, fijos, x, y)
        cambios, estado = opt.update(grads, estado, entrenables)
        return optax.apply_updates(entrenables, cambios), estado, perdida

    predecir = jax.jit(modelo.predecir)
    x, y = jnp.asarray(datos.x_train), jnp.asarray(datos.y_train)
    xb, yb = x[:LOTE], y[:LOTE]
    xe = jnp.asarray(datos.x_test[:BLOQUE_EVALUACION])

    t0 = time.perf_counter()
    for _ in range(2):
        _bloquear(paso(entrenables, estado, xb, yb))
    _bloquear(predecir(entrenables, fijos, xe))
    compilacion = time.perf_counter() - t0

    t0 = time.perf_counter()
    for k in range(n_pasos):
        i = (k * LOTE) % (len(x) - LOTE)
        entrenables, estado, perdida = _bloquear(paso(entrenables, estado, x[i : i + LOTE], y[i : i + LOTE]))
    t_paso = (time.perf_counter() - t0) / n_pasos

    t0 = time.perf_counter()
    _bloquear(predecir(entrenables, fijos, xe))
    t_eval = time.perf_counter() - t0

    epoca = t_paso * (N_TRAIN // LOTE) + t_eval * (N_TEST // BLOQUE_EVALUACION)
    grads = jax.grad(modelo.perdida)(*modelo.inicializar(jax.random.PRNGKey(0)), xb, yb)
    return {
        "modelo": nombre,
        "simulador": f"{dispositivo}:{diff_method}",
        "compilacion_s": compilacion,
        "paso_s": t_paso,
        "eval_bloque_s": t_eval,
        "epoca_s": epoca,
        "total_h": epoca * EPOCAS * SEMILLAS / 3600,
        "_perdida": float(modelo.perdida(*modelo.inicializar(jax.random.PRNGKey(0)), xb, yb)),
        "_grads": np.concatenate([np.ravel(g) for g in jax.tree_util.tree_leaves(grads)]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelos", nargs="+", default=sorted(CONSTRUCTORES))
    parser.add_argument("--simuladores", nargs="+", default=["default.qubit:best", "lightning.qubit:adjoint"])
    parser.add_argument("--pasos", type=int, default=10)
    args = parser.parse_args()

    datos = {t: preparar_datos(t) for t in (10, 16)}
    filas = []
    for nombre in args.modelos:
        sims = args.simuladores if nombre in CON_CIRCUITO else ["default.qubit:best"]
        ref = None
        for sim in sims:
            dispositivo, diff_method = sim.split(":")
            tam = construir_modelo(nombre).tam_imagen
            try:
                fila = medir(nombre, dispositivo, diff_method, args.pasos, datos[tam])
            except Exception as e:  # un simulador puede no soportar la combinación
                print(f"{nombre:24s} {sim:26s} ERROR: {type(e).__name__}: {e}", flush=True)
                continue
            if ref is None:
                ref = fila
                fila["dif_max"] = 0.0
            else:
                fila["dif_max"] = float(
                    max(abs(fila["_perdida"] - ref["_perdida"]), np.max(np.abs(fila["_grads"] - ref["_grads"])))
                )
            filas.append(fila)
            print(
                f"{nombre:24s} {sim:26s} compila {fila['compilacion_s']:7.1f} s · paso {fila['paso_s']:8.4f} s · "
                f"época {fila['epoca_s']:8.1f} s · 90 épocas {fila['total_h']:6.2f} h · dif {fila['dif_max']:.1e}",
                flush=True,
            )

    print("\nMejor simulador por modelo y total estimado (30 épocas × 3 semillas):")
    total = 0.0
    for nombre in args.modelos:
        candidatas = [f for f in filas if f["modelo"] == nombre and f["dif_max"] < 1e-8]
        if not candidatas:
            continue
        mejor = min(candidatas, key=lambda f: f["epoca_s"])
        total += mejor["total_h"]
        print(f"  {nombre:24s} {mejor['simulador']:26s} {mejor['total_h']:6.2f} h")
    print(f"  TOTAL estimado: {total:.2f} h")


if __name__ == "__main__":
    main()
