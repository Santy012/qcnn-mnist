"""Simulación de campo medio de v3 y ablación de lectura de v2."""

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from qcnn_mnist.modelos import campo_medio as cm
from qcnn_mnist.modelos import construir_modelo

B = 4


def _uno(lote, n, qubits):
    """Estado de base con ``qubits`` en |1⟩ y el resto en |0⟩."""
    e = cm.estado_inicial(lote, n)
    for q in qubits:
        e = cm.aplicar(e, q, jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128))
    return e


@pytest.mark.parametrize(
    ("propia", "de_pennylane", "args"),
    [
        (cm.matriz_ry, qml.RY, (0.7,)),
        (cm.matriz_rx, qml.RX, (-1.3,)),
        (cm.matriz_rot, qml.Rot, (0.3, -1.1, 2.4)),
    ],
)
def test_matrices_iguales_a_pennylane(propia, de_pennylane, args):
    np.testing.assert_allclose(propia(*args), qml.matrix(de_pennylane(*args, wires=0)), atol=1e-12)


def test_control_en_cero_no_cambia_el_objetivo():
    e = cm.aplicar(cm.estado_inicial(1, 2), 1, cm.matriz_ry(0.9))
    np.testing.assert_allclose(cm.cnot_campo_medio(e, 0, 1), e, atol=1e-12)


def test_control_en_uno_invierte_el_objetivo():
    e = _uno(1, 2, [0])
    for objetivo_inicial, z_esperado in [(0, -1.0), (1, 1.0)]:
        ei = e if objetivo_inicial == 0 else _uno(1, 2, [0, 1])
        assert float(cm.esperanza_z(cm.cnot_campo_medio(ei, 0, 1), 1)[0]) == pytest.approx(z_esperado)


def test_base_computacional_y_solo_cnot_igual_que_pennylane():
    n = 6
    puertas = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (2, 5)]
    dev = qml.device("default.qubit", wires=n)
    for unos in ([0], [1, 4], [0, 2, 5], [3]):

        @qml.qnode(dev)
        def cuantico():
            for q in unos:
                qml.PauliX(q)
            for c, t in puertas:
                qml.CNOT([c, t])
            return [qml.expval(qml.PauliZ(w)) for w in range(n)]

        e = _uno(1, n, unos)
        for c, t in puertas:
            e = cm.cnot_campo_medio(e, c, t)
        np.testing.assert_allclose([cm.esperanza_z(e, w)[0] for w in range(n)], cuantico(), atol=1e-12)


def _params_v3():
    modelo = construir_modelo("v3_campo_medio")
    entrenables, _ = modelo.inicializar(jax.random.PRNGKey(0))
    ang = jax.random.uniform(jax.random.PRNGKey(1), (B, 16), maxval=jnp.pi)
    return modelo, entrenables, ang


def test_estado_producto_normalizado_por_qubit():
    """El estado es un vector de 2 amplitudes por qubit (producto por construcción) y cada uno
    sigue normalizado tras todo el circuito."""
    _, p, ang = _params_v3()
    e = cm.estado_inicial(B, 16)
    e = cm.codificar(e, ang, list(range(16)))
    e = cm.conv_rot(e, p["theta1"], list(range(16)))
    e, s1 = cm.pooling_parametrico(e, p["pool1"], list(range(16)))
    assert e.shape == (B, 16, 2)
    np.testing.assert_allclose(jnp.sum(jnp.abs(e) ** 2, axis=-1), 1.0, atol=1e-12)


def test_sin_cnot_coincide_con_v3_sin_entrelazamiento(monkeypatch):
    """Las compuertas de un qubit son las mismas que en PennyLane: si se quitan los
    sustitutos de CNOT, el campo medio da lo mismo que ``v3_sin_entrelazamiento``."""
    modelo, p, ang = _params_v3()
    referencia = construir_modelo("v3_sin_entrelazamiento")
    cuanticos = [p[n] for n in modelo.CUANTICOS]
    esperado = jnp.stack(referencia.circuito(ang, *cuanticos), -1)

    monkeypatch.setattr(cm, "cnot_campo_medio", lambda e, c, t: e)
    obtenido = jnp.stack(cm.circuito_v3(ang, *cuanticos), -1)
    np.testing.assert_allclose(obtenido, esperado, atol=1e-10)


def test_campo_medio_usa_todos_los_parches():
    """A diferencia de la ablación sin interacción, la salida depende de los 16 parches."""
    modelo, p, ang = _params_v3()
    cuanticos = [p[n] for n in modelo.CUANTICOS]
    base = jnp.stack(cm.circuito_v3(ang, *cuanticos), -1)
    for k in range(16):
        otro = jnp.stack(cm.circuito_v3(ang.at[:, k].add(0.5), *cuanticos), -1)
        assert jnp.abs(otro - base).max() > 1e-9, k


def test_v2_lectura_z_usa_el_circuito_de_v2():
    """Mismas compuertas que v2: con la lectura por probabilidades de v2, el ⟨Z⟩ de cada qubit
    final se obtiene marginalizando las 16 probabilidades."""
    v2, vz = construir_modelo("qcnn_v2"), construir_modelo("v2_lectura_z")
    p, _ = vz.inicializar(jax.random.PRNGKey(0))
    ang = jax.random.uniform(jax.random.PRNGKey(1), (B, 10), maxval=jnp.pi)
    probs = v2.circuito(ang, p["theta1"], p["theta2"])  # (B, 16), qubits 3, 5, 7, 9
    bits = (np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1
    z_desde_probs = probs @ (1 - 2 * bits)
    np.testing.assert_allclose(jnp.stack(vz.circuito(ang, p["theta1"], p["theta2"]), -1), z_desde_probs, atol=1e-10)
