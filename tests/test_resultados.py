"""Agregación de resultados y ejecución de los notebooks."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from qcnn_mnist import resultados as res

RAIZ = Path(__file__).resolve().parents[1]


def _escribir(directorio, modelo, semilla, acc_test, acc_train=0.5, n_parametros=10):
    ruta = directorio / modelo / f"semilla_{semilla}.json"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(
            {
                "modelo": modelo,
                "semilla": semilla,
                "n_parametros": n_parametros,
                "acc_test": acc_test,
                "acc_train": acc_train,
                "segundos": 1800.0,
                "perdida_train": [2.0, 1.0],
                "acc_test_por_epoca": [0.1, acc_test],
                "matriz_confusion": np.eye(10, dtype=int).tolist(),
            }
        ),
        encoding="utf-8",
    )


def test_cargar_ordena_modelos_y_semillas(tmp_path):
    _escribir(tmp_path, "cpn_16", 43, 0.6)
    _escribir(tmp_path, "qcnn_v3", 44, 0.4)
    _escribir(tmp_path, "qcnn_v3", 42, 0.5)
    r = res.cargar(tmp_path)
    assert list(r) == ["qcnn_v3", "cpn_16"]
    assert [c["semilla"] for c in r["qcnn_v3"]] == [42, 44]


def test_resumen_media_desviacion_muestral_y_horas(tmp_path):
    for s, a in zip((42, 43, 44), (0.40, 0.50, 0.60)):
        _escribir(tmp_path, "qcnn_v3", s, a)
    r = res.resumen(res.cargar(tmp_path))["qcnn_v3"]
    assert r.acc_test_media == pytest.approx(0.5)
    assert r.acc_test_desv == pytest.approx(0.1)  # ddof = 1
    assert r.horas == pytest.approx(1.5)
    assert r.semillas == (42, 43, 44)


def test_resumen_rechaza_conteos_distintos(tmp_path):
    _escribir(tmp_path, "qcnn_v3", 42, 0.5, n_parametros=180)
    _escribir(tmp_path, "qcnn_v3", 43, 0.5, n_parametros=181)
    with pytest.raises(ValueError):
        res.resumen(res.cargar(tmp_path))


def test_series_y_confusion(tmp_path):
    _escribir(tmp_path, "cpn_16", 42, 0.6)
    _escribir(tmp_path, "cpn_16", 43, 0.7)
    corridas = res.cargar(tmp_path)["cpn_16"]
    assert res.serie(corridas, "acc_test_por_epoca").shape == (2, 2)
    assert np.array_equal(res.confusion_total(corridas), 2 * np.eye(10))


def test_formato_y_tabla(tmp_path):
    assert res.pct(0.4333) == "43,3"
    assert res.num(-17.06, signo=True) == "-17,1"
    assert res.error_estandar(0.5, 500) == pytest.approx(0.02236, abs=1e-5)
    _escribir(tmp_path, "cnn_16", 42, 0.78, acc_train=0.8, n_parametros=459)
    tabla = res.tabla_markdown(res.resumen(res.cargar(tmp_path)))
    assert tabla.splitlines()[-1] == "| CNN 16×16 | 459 | 78,0 ± 0,0 | 80,0 ± 0,0 | 0,50 |"


def test_resultados_del_estudio_completos():
    """``results/`` tiene las 10 configuraciones con las 3 semillas."""
    r = res.cargar()
    if not r:
        pytest.skip("Sin resultados en results/")
    assert set(r) == set(res.ORDEN)
    assert all([c["semilla"] for c in corridas] == [42, 43, 44] for corridas in r.values())


def test_actualizar_bloques_del_readme():
    sys.path.insert(0, str(RAIZ / "scripts"))
    from actualizar_readme import actualizar

    texto = "a\n<!-- resultados:t -->\nviejo\n<!-- /resultados:t -->\nb"
    assert actualizar(texto, {"t": "| nueva |"}) == "a\n<!-- resultados:t -->\n| nueva |\n<!-- /resultados:t -->\nb"
    with pytest.raises(ValueError):
        actualizar(texto, {"otro": "x"})


def test_readme_al_dia_con_results():
    """Las tablas del README coinciden con las que se generan desde ``results/``."""
    if not res.cargar():
        pytest.skip("Sin resultados en results/")
    sys.path.insert(0, str(RAIZ / "scripts"))
    from actualizar_readme import README, actualizar, bloques

    texto = README.read_text(encoding="utf-8")
    assert actualizar(texto, bloques()) == texto, "Ejecutar: python scripts/actualizar_readme.py"


@pytest.mark.notebooks
@pytest.mark.parametrize("ruta", sorted((RAIZ / "notebooks").glob("*.ipynb")), ids=lambda p: p.stem)
def test_notebook_corre(ruta):
    """Cada notebook corre de principio a fin leyendo ``results/``."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(ruta, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ruta.parent)}}).execute()
