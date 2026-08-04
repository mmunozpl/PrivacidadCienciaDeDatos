"""esquema anotado: el papel de cada columna viaja con el dato."""

import pandas as pd

PAPELES = {"identificador", "cuasi", "sensible", "neutro"}


def validar(df: pd.DataFrame, esquema: dict[str, str]) -> None:
    """Comprueba que la tabla y su esquema anotado coinciden.

    Toda tabla del libro pasa por aqui al crearse y tras cada join:
    una columna sin papel declarado es un riesgo sin evaluar.

    Args:
        df: tabla a validar.
        esquema: papel declarado de cada columna.

    Raises:
        ValueError: si hay papeles desconocidos, columnas sin papel o
            columnas declaradas que no existen en la tabla.
    """
    desconocidos = set(esquema.values()) - PAPELES
    if desconocidos:
        raise ValueError(f"papeles desconocidos: {sorted(desconocidos)}")
    sin_papel = set(df.columns) - set(esquema)
    if sin_papel:
        raise ValueError(
            f"columnas sin papel declarado: {sorted(sin_papel)}"
        )
    fantasma = set(esquema) - set(df.columns)
    if fantasma:
        raise ValueError(
            f"columnas declaradas que no existen: {sorted(fantasma)}"
        )


def columnas(esquema: dict[str, str], papel: str) -> list[str]:
    """Devuelve las columnas que tienen el papel indicado.

    Args:
        esquema: papel declarado de cada columna.
        papel: uno de PAPELES.

    Returns:
        Lista de nombres de columna, en el orden del esquema.
    """
    if papel not in PAPELES:
        raise ValueError(f"papel desconocido: {papel}")
    return [c for c, p in esquema.items() if p == papel]
