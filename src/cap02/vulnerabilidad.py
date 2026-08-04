"""quien es vulnerable: el riesgo no se reparte por igual.

Cruza el resultado del ataque de pertenencia con la rareza de cada
registro (el tamano de su clase de equivalencia, cap. 1) y mide como
crece la vulnerabilidad a medida que la persona es mas atipica.
Ademas mide la inferencia de atributo: adivinar el diagnostico de una
persona conocida a partir del modelo, comparado con acertar la clase
mayoritaria.

Escribe data/processed/vulnerabilidad.csv para la figura del libro.

Uso: python3 src/cap02/vulnerabilidad.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro
from comun.reident import clases_equivalencia

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SALIDA = RAIZ / "data" / "processed" / "vulnerabilidad.csv"

CUASI = ["edad", "sexo", "codigo_postal", "profesion"]
# rareza medida sobre un cuasi-identificador grueso, para que existan
# clases de varios tamanos (con el fino casi todas serian unicas)
CUASI_GRUESO = ["edad", "sexo", "profesion"]
OBJETIVO = "diagnostico"


def codificar(df: pd.DataFrame) -> np.ndarray:
    """Codifica los cuasi-identificadores como enteros."""
    return np.column_stack([
        df["edad"].to_numpy(),
        (df["sexo"] == "M").astype(int).to_numpy(),
        df["codigo_postal"].astype(int).to_numpy(),
        pd.factorize(df["profesion"])[0],
    ])


def main() -> None:
    """Mide vulnerabilidad por rareza e inferencia de atributo."""
    log = crear_registro("cap02.vulnerabilidad")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS)
    x, y = codificar(df), df[OBJETIVO].to_numpy()
    orden = rng.permutation(len(df))
    dentro, fuera = orden[: len(df) // 2], orden[len(df) // 2:]

    modelo = RandomForestClassifier(n_estimators=60, min_samples_leaf=1,
                                    random_state=42, n_jobs=-1)
    modelo.fit(x[dentro], y[dentro])

    # ── 1) vulnerabilidad al MIA segun la rareza del registro ────────
    p = modelo.predict_proba(x)
    clases = list(modelo.classes_)
    idx = np.array([clases.index(v) for v in y])
    confianza = p[np.arange(len(y)), idx]

    df = df.assign(
        tam_clase=clases_equivalencia(df, CUASI_GRUESO),
        confianza=confianza,
        miembro=np.isin(np.arange(len(df)), dentro),
    )
    cortes = [0, 1, 2, 4, 8, 16, np.inf]
    etiquetas = ["1", "2", "3-4", "5-8", "9-16", ">16"]
    df["rareza"] = pd.cut(df["tam_clase"], bins=cortes,
                          labels=etiquetas)

    filas = []
    log.info("vulnerabilidad al MIA segun el tamano de la clase de "
             "equivalencia (%s):", CUASI_GRUESO)
    for etiqueta, grupo in df.groupby("rareza", observed=True):
        m = grupo[grupo["miembro"]]["confianza"].mean()
        nm = grupo[~grupo["miembro"]]["confianza"].mean()
        log.info("  clase de %-5s (%5d filas): confianza miembro "
                 "%.3f vs no miembro %.3f -> separacion %.3f",
                 etiqueta, len(grupo), m, nm, m - nm)
        filas.append({"rareza": etiqueta, "filas": len(grupo),
                      "miembro": m, "no_miembro": nm,
                      "separacion": m - nm})

    # ── 2) inferencia de atributo ───────────────────────────────────
    mayoritaria = pd.Series(y[dentro]).value_counts().idxmax()
    base = float((y[fuera] == mayoritaria).mean())
    adivinado = modelo.predict(x[fuera])
    acierto = float((adivinado == y[fuera]).mean())
    log.info("inferencia de atributo sobre personas NO vistas: "
             "%.1f%% (linea base «siempre %s»: %.1f%%)",
             100 * acierto, mayoritaria, 100 * base)
    acierto_dentro = float((modelo.predict(x[dentro]) == y[dentro]).mean())
    log.info("la misma inferencia sobre personas del entrenamiento: "
             "%.1f%% -- la diferencia ES la fuga", 100 * acierto_dentro)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    log.info("guardado %s", SALIDA.relative_to(RAIZ))


if __name__ == "__main__":
    main()
