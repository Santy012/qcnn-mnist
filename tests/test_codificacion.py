import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from qcnn_mnist.codificacion import amplitudes, angulo, extraer_parches


def test_extraer_parches_formas_y_orden():
    img = jnp.arange(16.0).reshape(4, 4)
    p = extraer_parches(img, 2)
    assert p.shape == (4, 4)
    np.testing.assert_array_equal(p[0], [0, 1, 4, 5])  # arriba a la izquierda
    np.testing.assert_array_equal(p[1], [2, 3, 6, 7])  # arriba a la derecha
    np.testing.assert_array_equal(p[2], [8, 9, 12, 13])


@pytest.mark.parametrize("tam, parche, n", [(10, 2, 25), (16, 4, 16)])
def test_extraer_parches_lote(tam, parche, n):
    x = jnp.ones((3, tam, tam))
    assert extraer_parches(x, parche).shape == (3, n, parche * parche)


def test_amplitudes_norma_uno():
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.normal(size=(2, 16)))
    b = jnp.asarray(rng.normal(size=(2,)))
    p = jnp.asarray(rng.random((5, 16, 16)))
    amp = amplitudes(W, b, p)
    assert amp.shape == (5, 16, 2)
    # El ε dentro de la raíz aparta la norma de 1 en ε / (2‖v‖²) cuando ‖v‖ es pequeño.
    np.testing.assert_allclose(jnp.linalg.norm(amp, axis=-1), 1.0, atol=1e-4)


def test_angulo_reproduce_amplitudes():
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="jax")
    def estado(theta):
        qml.RY(theta, wires=0)
        return qml.state()

    for a in [(0.6, 0.8), (-0.6, 0.8), (0.6, -0.8), (-1.0, 0.0)]:
        amp = jnp.asarray(a)
        psi = estado(angulo(amp))
        np.testing.assert_allclose(jnp.real(psi), amp, atol=1e-6)


@pytest.mark.parametrize("b", [(1e-3, 0.0), (0.0, 0.0)])
def test_parche_negro_sin_nan(b):
    W = jnp.ones((2, 4))
    b = jnp.asarray(b)
    negro = jnp.zeros(4)

    def f(W, b):
        return angulo(amplitudes(W, b, negro))

    valor = f(W, b)
    grad_W, grad_b = jax.grad(f, argnums=(0, 1))(W, b)
    assert jnp.isfinite(valor)
    assert jnp.all(jnp.isfinite(grad_W)) and jnp.all(jnp.isfinite(grad_b))
