"""Prueba de humo: el paquete se importa y PennyLane funciona con JAX (grad y jit)."""

import jax
import jax.numpy as jnp
import pennylane as qml

import qcnn_mnist


def test_paquete_importable():
    assert qcnn_mnist.__version__


def test_circuito_jax_diferenciable():
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="jax")
    def circuito(theta):
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    theta = jnp.array(0.3)
    # <Z_1> = cos(theta) y su derivada es -sin(theta).
    assert jnp.isclose(jax.jit(circuito)(theta), jnp.cos(theta), atol=1e-6)
    assert jnp.isclose(jax.jit(jax.grad(circuito))(theta), -jnp.sin(theta), atol=1e-6)
