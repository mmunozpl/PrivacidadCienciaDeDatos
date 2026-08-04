"""estima la unicidad POBLACIONAL desde una muestra, y se comprueba.

El problema central del capitulo 3: se observa una muestra y hay que
decir cuantas de sus filas serian unicas en la poblacion entera. Aqui
se implementan dos estimadores y se validan contra la verdad conocida,
usando la poblacion sintetica del libro como «poblacion» y muestras
suyas como «muestras».

Estimadores:
- ingenuo: la unicidad de la muestra, tal cual (el que se usa por
  inercia y siempre sobreestima).
- Poisson: modela el tamano de clase poblacional como Poisson y
  corrige por la fraccion muestral; es el argumento del capitulo 1
  llevado a estimador.

Escribe data/processed/unicidad_estimadores.csv para la figura.

Uso: python3 src/cap03/unicidad_poblacional.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, progreso

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SALIDA = RAIZ / "data" / "processed" / "unicidad_estimadores.csv"

# cuasi-identificador grueso: con el fino todo seria unico y no habria
# nada que estimar (lo que ya ensena algo, y se comenta en el texto)
CUASI = ["edad5", "sexo", "provincia"]
FRACCIONES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
REPETICIONES = 5


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Genera el cuasi-identificador grueso del experimento."""
    cp = df["codigo_postal"].astype(str)
    return df.assign(provincia=cp.str[:2],
                     edad5=(df["edad"] // 5) * 5)


def unicidad(df: pd.DataFrame, cuasi: list[str]) -> float:
    """Fraccion de filas unicas en la tabla dada."""
    tam = df.groupby(cuasi, observed=True)[cuasi[0]].transform("size")
    return float((tam == 1).mean())


def estimar_poisson(muestra: pd.DataFrame, cuasi: list[str],
                    f: float) -> float:
    """Estima la unicidad poblacional con el modelo de Poisson.

    Si una clase tiene n_j filas en una muestra de fraccion f, su
    tamano poblacional esperado es n_j/f. Bajo un modelo de Poisson
    con esa media, la probabilidad de que la clase tenga exactamente
    un individuo en la poblacion es (n_j/f)·exp(-n_j/f). Se promedia
    ponderando por las filas de la muestra.

    Args:
        muestra: tabla observada.
        cuasi: columnas del cuasi-identificador.
        f: fraccion muestral (n muestra / N poblacion).

    Returns:
        Estimacion de la fraccion de filas unicas en la poblacion.
    """
    tam = muestra.groupby(cuasi, observed=True)[cuasi[0]].transform("size")
    lam = tam.to_numpy() / f
    return float(np.mean(lam * np.exp(-lam)))


def estimar_poisson_gamma(muestra: pd.DataFrame, cuasi: list[str],
                          f: float) -> float:
    """Estima la unicidad poblacional con un modelo Poisson-Gamma.

    El fallo del Poisson simple es suponer que todas las clases tienen
    el mismo tamano esperado. Aqui se admite heterogeneidad: el tamano
    poblacional de la clase j es Poisson(lambda_j) con
    lambda_j ~ Gamma(alfa, beta), cuyos parametros se ajustan por
    momentos sobre los recuentos observados. Una fila es unica en la
    poblacion si es unica en la muestra Y no hay nadie mas fuera:

        P(resto = 0 | n_j) = ((1/beta + f) / (1/beta + 1))^(alfa + n_j)

    Args:
        muestra: tabla observada.
        cuasi: columnas del cuasi-identificador.
        f: fraccion muestral.

    Returns:
        Estimacion de la fraccion de filas unicas en la poblacion.
    """
    n_j = muestra.groupby(cuasi, observed=True).size().to_numpy()
    m, v = n_j.mean(), n_j.var(ddof=1)
    # ajuste por momentos de la binomial negativa; si no hay
    # sobredispersion, se degrada al Poisson simple
    if v <= m or f <= 0:
        return estimar_poisson(muestra, cuasi, f)
    beta = (v / m - 1.0) / f
    alfa = m / (beta * f)
    tam = muestra.groupby(cuasi, observed=True)[cuasi[0]].transform("size")
    n = tam.to_numpy()
    base = (1.0 / beta + f) / (1.0 / beta + 1.0)
    p_resto_cero = base ** (alfa + n)
    return float(np.mean(np.where(n == 1, p_resto_cero, 0.0)))


def main() -> None:
    """Compara los dos estimadores contra la verdad, por fraccion."""
    log = crear_registro("cap03.unicidad")
    rng = fijar_semillas()

    poblacion = preparar(pd.read_parquet(DATOS))
    verdad = unicidad(poblacion, CUASI)
    log.info("«población» de %d filas; unicidad REAL con %s: %.2f%%",
             len(poblacion), CUASI, 100 * verdad)

    filas = []
    casos = [(f, r) for f in FRACCIONES for r in range(REPETICIONES)]
    for f, _ in progreso(casos, len(casos), log, cada=10,
                         tarea="muestreos"):
        idx = rng.choice(len(poblacion), size=int(f * len(poblacion)),
                         replace=False)
        muestra = poblacion.iloc[idx]
        filas.append({
            "fraccion": f,
            "ingenuo": unicidad(muestra, CUASI),
            "poisson": estimar_poisson(muestra, CUASI, f),
            "poisson_gamma": estimar_poisson_gamma(muestra, CUASI, f),
            "verdad": verdad,
        })

    res = pd.DataFrame(filas)
    resumen = res.groupby("fraccion").mean(numeric_only=True)
    log.info("estimaciones (media de %d repeticiones):", REPETICIONES)
    for f, fila in resumen.iterrows():
        log.info("  f=%4.0f%%  ingenuo=%6.2f%% (x%5.1f)  "
                 "poisson=%5.2f%%  poisson-gamma=%5.2f%% (x%.2f)  "
                 "verdad=%.2f%%",
                 100 * f, 100 * fila["ingenuo"],
                 fila["ingenuo"] / verdad, 100 * fila["poisson"],
                 100 * fila["poisson_gamma"],
                 fila["poisson_gamma"] / verdad, 100 * verdad)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    resumen.reset_index().to_csv(SALIDA, index=False)
    log.info("guardado %s", SALIDA.relative_to(RAIZ))
    peor = resumen["ingenuo"].max() / verdad
    log.info("el estimador ingenuo llega a sobreestimar el riesgo "
             "hasta %.0f veces", peor)


if __name__ == "__main__":
    main()
