"""Registro de modelos del estudio."""

from qcnn_mnist.modelos.base import Modelo
from qcnn_mnist.modelos.clasicos import CNN, CPN, SoloClasico
from qcnn_mnist.modelos.qcnn import QCNNv1, QCNNv2, QCNNv2LecturaZ, QCNNv3, QCNNv3CampoMedio

CONSTRUCTORES = {
    "qcnn_v1": lambda **kw: QCNNv1(**kw),
    "qcnn_v2": lambda **kw: QCNNv2(**kw),
    "v2_lectura_z": lambda **kw: QCNNv2LecturaZ(**kw),
    "qcnn_v3": lambda **kw: QCNNv3(**kw),
    "v3_sin_entrelazamiento": lambda **kw: QCNNv3(entrelazar=False, **kw),
    "v3_campo_medio": lambda **kw: QCNNv3CampoMedio(),
    "v3_congelado": lambda **kw: QCNNv3(congelar=True, **kw),
    "v3_solo_clasico": lambda **kw: SoloClasico(),
    "cnn_10": lambda **kw: CNN(tam_imagen=10, tam_parche=2, n_parches=10),
    "cnn_16": lambda **kw: CNN(tam_imagen=16, tam_parche=4, n_parches=16),
    "cpn_10": lambda **kw: CPN(tam_imagen=10),
    "cpn_16": lambda **kw: CPN(tam_imagen=16),
}


def construir_modelo(nombre: str, **opciones) -> Modelo:
    """Construye un modelo por nombre. ``opciones`` (dispositivo, diff_method) solo
    afectan a los modelos con circuito."""
    if nombre not in CONSTRUCTORES:
        raise KeyError(f"Modelo desconocido: {nombre}. Opciones: {sorted(CONSTRUCTORES)}")
    return CONSTRUCTORES[nombre](**opciones)


__all__ = ["CONSTRUCTORES", "Modelo", "construir_modelo"]
