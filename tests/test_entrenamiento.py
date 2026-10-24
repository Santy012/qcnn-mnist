import json

import numpy as np
import pytest

from qcnn_mnist.datos import Datos
from qcnn_mnist.entrenamiento import entrenar
from qcnn_mnist.evaluacion import accuracy, matriz_confusion
from qcnn_mnist.experimento import Configuracion, correr, ruta_resultado
from qcnn_mnist.modelos import construir_modelo


def _datos_sinteticos(tam, n=32, semilla=0):
    rng = np.random.default_rng(semilla)
    return Datos(
        x_train=rng.random((n, tam, tam)),
        y_train=rng.integers(0, 10, n).astype(np.int32),
        x_test=rng.random((n // 2, tam, tam)),
        y_test=rng.integers(0, 10, n // 2).astype(np.int32),
    )


def test_metricas():
    assert accuracy([1, 2, 3], [1, 0, 3]) == pytest.approx(2 / 3)
    m = matriz_confusion([0, 0, 1], [0, 1, 1])
    assert m[0, 0] == 1 and m[0, 1] == 1 and m[1, 1] == 1 and m.sum() == 3


@pytest.mark.parametrize("nombre", ["cpn_10", "qcnn_v1"])
def test_entrenar_es_reproducible(nombre):
    modelo = construir_modelo(nombre)
    datos = _datos_sinteticos(modelo.tam_imagen)
    _, _, h1 = entrenar(modelo, datos, 0.01, 2, 8, semilla=42, informar=lambda _: None)
    _, _, h2 = entrenar(modelo, datos, 0.01, 2, 8, semilla=42, informar=lambda _: None)
    _, _, h3 = entrenar(modelo, datos, 0.01, 2, 8, semilla=43, informar=lambda _: None)
    assert len(h1.perdida_train) == 2 and len(h1.acc_test) == 2
    assert h1.perdida_train == h2.perdida_train
    assert h1.perdida_train != h3.perdida_train


def test_correr_guarda_resultados(tmp_path):
    config = Configuracion(modelo="cnn_10", epocas=1, n_train=64, n_test=32)
    resultado = correr(config, 42, tmp_path, informar=lambda _: None)
    ruta = ruta_resultado(tmp_path, "cnn_10", 42)
    guardado = json.loads(ruta.read_text(encoding="utf-8"))
    assert guardado["n_parametros"] == resultado["n_parametros"] == 225
    assert np.array(guardado["matriz_confusion"]).sum() == 32
    assert ruta.with_name("semilla_42_params.npz").exists()
