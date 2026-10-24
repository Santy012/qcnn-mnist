import numpy as np
import pytest

from qcnn_mnist.datos import (
    DIRECTORIO_DATOS,
    particion_estratificada,
    preparar_datos,
    redimensionar_bilineal,
)


def test_bilineal_conserva_imagen_constante():
    x = np.full((3, 28, 28), 0.7)
    y = redimensionar_bilineal(x, 10)
    assert y.shape == (3, 10, 10)
    np.testing.assert_allclose(y, 0.7)


def test_bilineal_dos_a_uno_promedia():
    # Con centros a la mitad, el único píxel de salida cae entre los dos de entrada.
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(redimensionar_bilineal(x, 1), [[1.5]])


def test_bilineal_identidad_y_rango():
    rng = np.random.default_rng(0)
    x = rng.random((28, 28))
    np.testing.assert_allclose(redimensionar_bilineal(x, 28), x)
    y = redimensionar_bilineal(x, 16)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_particion_estratificada():
    etiquetas = np.repeat(np.arange(10), [100, 120, 90, 110, 95, 105, 98, 102, 87, 93])
    idx = particion_estratificada(etiquetas, 200, semilla=42)
    assert idx.shape == (200,)
    assert len(np.unique(idx)) == 200
    esperado = np.bincount(etiquetas) / len(etiquetas) * 200
    assert np.all(np.abs(np.bincount(etiquetas[idx], minlength=10) - esperado) <= 1)
    np.testing.assert_array_equal(idx, particion_estratificada(etiquetas, 200, semilla=42))
    assert not np.array_equal(idx, particion_estratificada(etiquetas, 200, semilla=43))


@pytest.fixture(scope="module")
def datos_16():
    if not (DIRECTORIO_DATOS / "mnist.npz").exists():
        try:
            return preparar_datos(16)
        except OSError:
            pytest.skip("MNIST no está descargado y no hay red.")
    return preparar_datos(16)


def test_preparar_datos(datos_16):
    d = datos_16
    assert d.x_train.shape == (2000, 16, 16) and d.x_test.shape == (500, 16, 16)
    assert d.y_train.dtype == np.int32
    assert d.x_train.min() >= 0.0 and d.x_train.max() <= 1.0
    assert np.all(np.bincount(d.y_train, minlength=10) >= 180)
    assert np.all(np.bincount(d.y_test, minlength=10) >= 40)


def test_particion_no_depende_del_tamano(datos_16):
    d10 = preparar_datos(10)
    np.testing.assert_array_equal(d10.y_train, datos_16.y_train)
    np.testing.assert_array_equal(d10.y_test, datos_16.y_test)
