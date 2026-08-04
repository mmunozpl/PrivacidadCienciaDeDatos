"""mide la informacion identificante de los cuasi-identificadores.

Sobre el dataset sintetico del libro calcula la entropia empirica de
cada cuasi-identificador, su suma (cota bajo independencia), la
entropia conjunta empirica y los bits necesarios para senalar a un
individuo en Espana; cierra con el k-anonimato y la unicidad muestral.

Uso: python3 codigo/cap01/entropia_cuasi.py
     (antes: python3 codigo/cap01/generar_dataset.py)
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "codigo"))

from comun.registro import crear_registro
from comun.reident import k_anonimato, unicidad_muestral

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
CUASI = ["edad", "sexo", "codigo_postal", "profesion"]
POBLACION_ES = 48_600_000


def entropia(serie: pd.Series) -> float:
    """Entropia de Shannon empirica de una columna, en bits.

    Args:
        serie: columna categorica o discreta.

    Returns:
        H = -sum(p * log2 p) sobre las frecuencias observadas.
    """
    p = serie.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def unicidad_esperada(df: pd.DataFrame, cuasi: list[str]) -> float:
    """Unicidad muestral esperada bajo independencia de columnas.

    Para cada fila estima la probabilidad de su combinacion como
    producto de frecuencias marginales y calcula la probabilidad de
    que ninguna de las otras n-1 filas la comparta.

    Args:
        df: tabla a evaluar.
        cuasi: columnas que actuan como cuasi-identificador.

    Returns:
        Media de (1 - p_fila)^(n-1) sobre las filas.
    """
    n = len(df)
    p = np.ones(n)
    for col in cuasi:
        frecuencias = df[col].map(
            df[col].value_counts(normalize=True)
        )
        p *= frecuencias.to_numpy()
    return float(np.mean((1.0 - p) ** (n - 1)))


def main() -> None:
    """Mide y registra la aritmetica de la identificacion."""
    log = crear_registro("cap01.entropia_cuasi")
    df = pd.read_parquet(DATOS)
    log.info("dataset: %d filas, cuasi-identificadores: %s",
             len(df), CUASI)

    suma = 0.0
    for col in CUASI:
        h = entropia(df[col])
        suma += h
        log.info("H(%s) = %.2f bits (%d valores distintos)",
                 col, h, df[col].nunique())

    conjunta = entropia(df[CUASI].astype(str).agg("|".join, axis=1))
    objetivo = math.log2(POBLACION_ES)
    log.info("suma de entropias (cota si independientes): %.2f bits",
             suma)
    log.info("entropia conjunta empirica: %.2f bits "
             "(techo muestral: log2(%d) = %.2f)",
             conjunta, len(df), math.log2(len(df)))
    log.info("bits para senalar a 1 entre %d espanoles: %.2f",
             POBLACION_ES, objetivo)

    k = k_anonimato(df, CUASI)
    unicos = unicidad_muestral(df, CUASI)
    log.info("k-anonimato: k=%d; unicidad muestral: %.1f%%",
             k, 100 * unicos)
    esperada = unicidad_esperada(df, CUASI)
    log.info("unicidad esperada (modelo de independencia): %.1f%%",
             100 * esperada)

    # gemelos esperados en la poblacion, fila a fila: lambda = N * p
    p = np.ones(len(df))
    for col in CUASI:
        p *= df[col].map(
            df[col].value_counts(normalize=True)
        ).to_numpy()
    lam = POBLACION_ES * p
    log.info("gemelos esperados en Espana por fila (lambda): "
             "min=%.1f, mediana=%.0f, max=%.0f",
             lam.min(), np.median(lam), lam.max())
    log.info("filas con lambda<1 (unicas esperables en poblacion): "
             "%.2f%%", 100 * float((lam < 1).mean()))


if __name__ == "__main__":
    main()
