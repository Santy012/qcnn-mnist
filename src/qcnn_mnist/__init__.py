"""QCNN híbrida sobre MNIST: datos, codificación, circuitos, modelos, entrenamiento y evaluación."""

import jax

# Precisión doble en todo el estudio, como en los notebooks originales.
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"
