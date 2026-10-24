import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from qcnn_mnist.modelos import CONSTRUCTORES, construir_modelo

# Conteo de parámetros entrenables esperado (incluye todas las capas entrenables).
CONTEO = {
    "qcnn_v1": 26,
    "qcnn_v2": 64,
    "qcnn_v3": 180,
    "v3_sin_entrelazamiento": 180,
    "v3_congelado": 84,
    "v3_solo_clasico": 204,
    "cnn_10": 225,
    "cnn_16": 459,
    "cpn_10": 78,
    "cpn_16": 112,
}
B = 3


@pytest.fixture(scope="module", params=sorted(CONSTRUCTORES))
def modelo_y_params(request):
    modelo = construir_modelo(request.param)
    entrenables, fijos = modelo.inicializar(jax.random.PRNGKey(0))
    x = jax.random.uniform(jax.random.PRNGKey(1), (B, modelo.tam_imagen, modelo.tam_imagen))
    return modelo, entrenables, fijos, x


def test_registro_completo():
    assert set(CONSTRUCTORES) == set(CONTEO)


def test_conteo_de_parametros(modelo_y_params):
    modelo, entrenables, _, _ = modelo_y_params
    assert modelo.contar_parametros(entrenables) == CONTEO[modelo.nombre]


def test_forma_de_salida(modelo_y_params):
    modelo, entrenables, fijos, x = modelo_y_params
    s = modelo.salida(entrenables, fijos, x)
    if modelo.nombre in ("qcnn_v1", "qcnn_v2"):
        assert s.shape == (B, 16)
        np.testing.assert_allclose(s.sum(axis=-1), 1.0, atol=1e-10)
    else:
        assert s.shape == (B, 10)
    assert modelo.predecir(entrenables, fijos, x).shape == (B,)


def test_gradiente_finito(modelo_y_params):
    modelo, entrenables, fijos, x = modelo_y_params
    y = jnp.array([0, 5, 9])
    perdida, grads = jax.jit(jax.value_and_grad(modelo.perdida))(entrenables, fijos, x, y)
    assert jnp.isfinite(perdida)
    for g in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(g))
    assert any(jnp.any(g != 0) for g in jax.tree_util.tree_leaves(grads))


def test_salida_por_lote_igual_a_imagen_por_imagen():
    modelo = construir_modelo("qcnn_v3")
    entrenables, fijos = modelo.inicializar(jax.random.PRNGKey(0))
    x = jax.random.uniform(jax.random.PRNGKey(1), (B, 16, 16))
    lote = modelo.salida(entrenables, fijos, x)
    uno_a_uno = jnp.concatenate([modelo.salida(entrenables, fijos, x[i : i + 1]) for i in range(B)])
    np.testing.assert_allclose(lote, uno_a_uno, atol=1e-10)


def test_congelado_separa_parametros_cuanticos():
    modelo = construir_modelo("v3_congelado")
    entrenables, fijos = modelo.inicializar(jax.random.PRNGKey(0))
    assert set(fijos) == {"theta1", "pool1", "theta2", "pool2", "theta3"}
    assert set(entrenables) == {"W", "b", "W_out", "b_out"}


def _operaciones(modelo, entrenables, fijos):
    p = {**fijos, **entrenables}
    angulos = jnp.zeros((16,))
    with qml.queuing.AnnotatedQueue() as cola:
        modelo.circuito.func(angulos, *(p[n] for n in modelo.CUANTICOS))
    return qml.tape.QuantumScript.from_queue(cola).operations


def test_sin_entrelazamiento_no_tiene_compuertas_de_dos_qubits():
    for nombre, esperado in [("v3_sin_entrelazamiento", False), ("qcnn_v3", True)]:
        modelo = construir_modelo(nombre)
        entrenables, fijos = modelo.inicializar(jax.random.PRNGKey(0))
        ops = _operaciones(modelo, entrenables, fijos)
        assert any(len(op.wires) > 1 for op in ops) is esperado
