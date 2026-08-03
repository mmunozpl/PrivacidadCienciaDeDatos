"""metricas de reidentificacion: k-anonimato y unicidad."""

import pandas as pd


def clases_equivalencia(df: pd.DataFrame,
                        cuasi: list[str]) -> pd.Series:
    """Calcula el tamano de la clase de equivalencia de cada fila.

    Una clase de equivalencia agrupa las filas que comparten los
    mismos valores en los cuasi-identificadores.

    Args:
        df: tabla a evaluar.
        cuasi: columnas que actuan como cuasi-identificador.

    Returns:
        Serie alineada con df con el tamano de la clase de cada fila.
    """
    return df.groupby(cuasi, observed=True, dropna=False)[
        cuasi[0]
    ].transform("size")


def k_anonimato(df: pd.DataFrame, cuasi: list[str]) -> int:
    """Devuelve el k del conjunto: la clase de equivalencia minima.

    Args:
        df: tabla a evaluar.
        cuasi: columnas que actuan como cuasi-identificador.

    Returns:
        El menor tamano de clase; k=1 significa filas unicas.
    """
    return int(clases_equivalencia(df, cuasi).min())


def unicidad_muestral(df: pd.DataFrame, cuasi: list[str]) -> float:
    """Fraccion de filas unicas en la muestra para esos cuasi.

    Es la unicidad MUESTRAL: la poblacional exige un modelo de la
    poblacion (cap. 3) y siempre es menor o igual que esta.

    Args:
        df: tabla a evaluar.
        cuasi: columnas que actuan como cuasi-identificador.

    Returns:
        Proporcion de filas con clase de equivalencia de tamano 1.
    """
    return float((clases_equivalencia(df, cuasi) == 1).mean())
