# QCNN MNIST

Estudio de una red convolucional cuántica híbrida (QCNN) sobre MNIST: evolución de la arquitectura en tres versiones y medición de cuánto aporta la parte cuántica frente a controles clásicos de tamaño parecido.

> Proyecto en construcción.

## Instalación

Requiere Python 3.13.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

En Windows, si el entorno se crea en una ruta muy larga, la instalación puede fallar por el límite de 260 caracteres; basta con usar una ruta más corta o habilitar las rutas largas del sistema.

## Pruebas

```bash
pytest
```
