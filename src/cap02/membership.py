"""ataque de inferencia de pertenencia (MIA) sobre un modelo entrenado.

Entrena dos modelos con los MISMOS datos —uno memorizador (arbol sin
podar) y otro regularizado— y ataca ambos con el criterio de perdida:
si el modelo esta mas seguro de un registro, probablemente lo vio al
entrenar. Se evalua como manda Carlini et al. (2022): ademas del AUC,
la tasa de aciertos TPR a tasas de falsa alarma MUY bajas, que es
donde un ataque se vuelve peligroso de verdad.

Escribe data/processed/mia_roc.csv con las curvas para la figura.

Uso: python3 src/cap02/membership.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SALIDA = RAIZ / "data" / "processed" / "mia_roc.csv"

CUASI = ["edad", "sexo", "codigo_postal", "profesion"]
OBJETIVO = "diagnostico"

MODELOS = {
    "memorizador": dict(n_estimators=60, max_depth=None,
                        min_samples_leaf=1),
    "regularizado": dict(n_estimators=60, max_depth=6,
                         min_samples_leaf=50),
}


def codificar(df: pd.DataFrame) -> np.ndarray:
    """Codifica los cuasi-identificadores como enteros.

    Args:
        df: tabla con las columnas de CUASI.

    Returns:
        Matriz de caracteristicas numericas.
    """
    x = pd.DataFrame(index=df.index)
    x["edad"] = df["edad"]
    x["sexo"] = (df["sexo"] == "M").astype(int)
    # el CP entra entero: alta cardinalidad, el combustible de la
    # memorizacion (cap. 2, seccion de memorizacion)
    x["cp"] = df["codigo_postal"].astype(int)
    x["prof"] = pd.factorize(df["profesion"])[0]
    return x.to_numpy()


def perdida(modelo, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Perdida logaritmica por muestra: el estadistico del ataque.

    Args:
        modelo: clasificador ya entrenado.
        x: caracteristicas.
        y: etiquetas verdaderas.

    Returns:
        Vector de perdidas (menor = el modelo esta mas seguro).
    """
    p = modelo.predict_proba(x)
    clases = list(modelo.classes_)
    idx = np.array([clases.index(v) for v in y])
    seguro = np.clip(p[np.arange(len(y)), idx], 1e-12, 1.0)
    return -np.log(seguro)


def tpr_a_fpr(fpr: np.ndarray, tpr: np.ndarray,
              objetivo: float) -> float:
    """Interpola la TPR a una FPR objetivo.

    Args:
        fpr: tasas de falsa alarma de la curva ROC.
        tpr: tasas de acierto correspondientes.
        objetivo: FPR de interes (p. ej. 0.001).

    Returns:
        TPR alcanzada a esa FPR.
    """
    return float(np.interp(objetivo, fpr, tpr))


def main() -> None:
    """Entrena, ataca y compara los dos modelos."""
    log = crear_registro("cap02.membership")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS)
    x, y = codificar(df), df[OBJETIVO].to_numpy()

    # mitad dentro del entrenamiento (miembros), mitad fuera
    orden = rng.permutation(len(df))
    dentro, fuera = orden[: len(df) // 2], orden[len(df) // 2:]
    log.info("miembros: %d · no miembros: %d", len(dentro), len(fuera))

    curvas = []
    for nombre, params in MODELOS.items():
        modelo = RandomForestClassifier(random_state=42, n_jobs=-1,
                                        **params)
        modelo.fit(x[dentro], y[dentro])
        exac_dentro = modelo.score(x[dentro], y[dentro])
        exac_fuera = modelo.score(x[fuera], y[fuera])
        log.info("%s: exactitud dentro %.3f · fuera %.3f · brecha %.3f",
                 nombre, exac_dentro, exac_fuera,
                 exac_dentro - exac_fuera)

        # estadistico del ataque: -perdida (mas alto = mas «miembro»)
        puntuacion = np.concatenate([
            -perdida(modelo, x[dentro], y[dentro]),
            -perdida(modelo, x[fuera], y[fuera]),
        ])
        etiqueta = np.concatenate([np.ones(len(dentro)),
                                   np.zeros(len(fuera))])
        auc = roc_auc_score(etiqueta, puntuacion)
        fpr, tpr, _ = roc_curve(etiqueta, puntuacion)
        log.info("  MIA AUC = %.3f | TPR@FPR=1%% : %.3f | "
                 "TPR@FPR=0.1%% : %.3f",
                 auc, tpr_a_fpr(fpr, tpr, 0.01),
                 tpr_a_fpr(fpr, tpr, 0.001))
        # se submuestrea la curva para la figura del libro
        paso = max(1, len(fpr) // 200)
        curvas.append(pd.DataFrame({
            "modelo": nombre, "fpr": fpr[::paso], "tpr": tpr[::paso],
        }))

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(curvas).to_csv(SALIDA, index=False)
    log.info("guardado %s (%d puntos de curva)",
             SALIDA.relative_to(RAIZ), sum(len(c) for c in curvas))


if __name__ == "__main__":
    main()
