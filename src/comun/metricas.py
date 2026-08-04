"""metricas sintacticas de riesgo: k-anonimato, l-diversidad y
t-cercania, y los tres modelos de riesgo de reidentificacion.

Implementaciones directas de las definiciones del capitulo 3, pensadas
para leerse junto al texto: cada funcion calcula una sola cosa.
"""

import numpy as np
import pandas as pd


# ── metricas sintacticas ─────────────────────────────────────────────

def k_anonimato(df: pd.DataFrame, cuasi: list[str]) -> int:
    """k del conjunto: la clase de equivalencia mas pequena.

    Args:
        df: tabla a evaluar.
        cuasi: columnas del cuasi-identificador.

    Returns:
        Tamano minimo de clase; k=1 significa que hay filas unicas.
    """
    return int(df.groupby(cuasi, observed=True).size().min())


def l_diversidad(df: pd.DataFrame, cuasi: list[str],
                 sensible: str, entropica: bool = False) -> float:
    """l del conjunto: diversidad minima del atributo sensible.

    Args:
        df: tabla a evaluar.
        cuasi: columnas del cuasi-identificador.
        sensible: columna del atributo sensible.
        entropica: si True, usa l-diversidad ENTROPICA
            (exp de la entropia de Shannon dentro de la clase); si
            False, la variante «distinct» (numero de valores).

    Returns:
        El menor valor de diversidad sobre todas las clases.
    """
    def diversidad(grupo: pd.Series) -> float:
        if not entropica:
            return float(grupo.nunique())
        p = grupo.value_counts(normalize=True).to_numpy()
        return float(np.exp(-(p * np.log(p)).sum()))

    return float(df.groupby(cuasi, observed=True)[sensible]
                 .apply(diversidad).min())


def t_cercania(df: pd.DataFrame, cuasi: list[str],
               sensible: str) -> float:
    """t del conjunto: distancia maxima entre clase y global.

    Usa la distancia de variacion total (mitad de la L1), que para
    atributos categoricos sin orden es el analogo natural de la
    Earth Mover's Distance de Li et al.

    Args:
        df: tabla a evaluar.
        cuasi: columnas del cuasi-identificador.
        sensible: columna del atributo sensible.

    Returns:
        El maximo sobre las clases de la distancia a la distribucion
        global; 0 = todas las clases son como el conjunto entero.
    """
    global_ = df[sensible].value_counts(normalize=True)

    def distancia(grupo: pd.Series) -> float:
        local = grupo.value_counts(normalize=True)
        alineadas = local.reindex(global_.index, fill_value=0.0)
        return float(0.5 * np.abs(alineadas - global_).sum())

    return float(df.groupby(cuasi, observed=True)[sensible]
                 .apply(distancia).max())


# ── modelos de riesgo de reidentificacion ────────────────────────────

def riesgos(df: pd.DataFrame, cuasi: list[str],
            n_poblacion: int | None = None) -> dict[str, float]:
    """Calcula los tres modelos de riesgo del capitulo 3.

    - fiscal (prosecutor): el adversario SABE que su objetivo esta en
      la tabla; su riesgo por fila es 1/tamano de clase. Se reporta el
      maximo (el peor caso) y la media.
    - periodista (journalist): el adversario no lo sabe, y la tabla es
      una muestra de una poblacion mayor; el riesgo por fila usa el
      tamano de la clase en la POBLACION, siempre >= el de la muestra,
      luego el riesgo es menor.
    - comercial (marketer): interesa reidentificar a muchos, no a uno;
      es el riesgo MEDIO sobre todas las filas.

    Args:
        df: tabla a evaluar.
        cuasi: columnas del cuasi-identificador.
        n_poblacion: tamano de la poblacion de referencia; si se
            indica, se estima el riesgo del periodista suponiendo
            muestreo aleatorio.

    Returns:
        Diccionario con los riesgos calculados.
    """
    tam = df.groupby(cuasi, observed=True)[cuasi[0]].transform("size")
    r_fila = 1.0 / tam
    salida = {
        "fiscal_max": float(r_fila.max()),
        "fiscal_medio": float(r_fila.mean()),
        "comercial": float(r_fila.mean()),
        "unicas": float((tam == 1).mean()),
    }
    if n_poblacion:
        # bajo muestreo aleatorio de fraccion f, el tamano esperado de
        # la clase en la poblacion es tam/f
        f = len(df) / n_poblacion
        salida["periodista_max"] = float((1.0 / (tam / f)).max())
        salida["periodista_medio"] = float((1.0 / (tam / f)).mean())
        salida["fraccion_muestral"] = f
    return salida
