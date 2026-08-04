"""mide k, l y t sobre el dataset del libro, y su coste en utilidad.

Recorre una escalera de generalizacion (de los datos en crudo a la
provincia y la franja etaria) y anota, en cada peldano, las tres
metricas sintacticas y la utilidad que queda. De ahi sale la curva
utilidad-riesgo del capitulo 3.

Uso: python3 src/cap03/metricas_medidas.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.metricas import k_anonimato, l_diversidad, riesgos, t_cercania
from comun.registro import crear_registro, progreso

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SALIDA = RAIZ / "data" / "processed" / "escalera_generalizacion.csv"

SENSIBLE = "diagnostico"
POBLACION_ES = 49_128_297

# la escalera: cada peldano generaliza un poco mas
PELDANOS = [
    ("crudo", ["edad", "sexo", "codigo_postal", "profesion"]),
    ("sin profesión", ["edad", "sexo", "codigo_postal"]),
    ("CP a 3 dígitos", ["edad", "sexo", "cp3"]),
    ("edad en quinquenios", ["edad5", "sexo", "cp3"]),
    ("provincia", ["edad5", "sexo", "provincia"]),
    ("franja etaria", ["franja", "sexo", "provincia"]),
]


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Anade las columnas generalizadas de la escalera.

    Args:
        df: tabla original del libro.

    Returns:
        La tabla con las columnas derivadas.
    """
    cp = df["codigo_postal"].astype(str)
    return df.assign(
        cp3=cp.str[:3],
        provincia=cp.str[:2],
        edad5=(df["edad"] // 5) * 5,
        franja=pd.cut(df["edad"], bins=[-1, 17, 34, 49, 64, 120],
                      labels=["0-17", "18-34", "35-49", "50-64",
                              "65+"]),
    )


def utilidad(df: pd.DataFrame, cuasi: list[str]) -> float:
    """Utilidad que queda tras generalizar, en bits.

    Se mide como la entropia conjunta empirica del cuasi-identificador:
    cuanta informacion sigue distinguiendo a unas filas de otras y, por
    tanto, cuanto analisis sigue siendo posible.

    Args:
        df: tabla generalizada.
        cuasi: columnas del cuasi-identificador.

    Returns:
        Entropia conjunta en bits.
    """
    combinada = df[cuasi].astype(str).agg("|".join, axis=1)
    p = combinada.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def suprimir_hasta(df: pd.DataFrame, cuasi: list[str],
                   k_objetivo: int) -> tuple[pd.DataFrame, float]:
    """Suprime las filas de clases pequenas hasta alcanzar k.

    Es lo que hace cualquier algoritmo de k-anonimizacion real: como k
    es un MINIMO, una sola fila rara lo mantiene en 1, y la unica forma
    de subirlo sin generalizar mas es retirar esas filas.

    Args:
        df: tabla generalizada.
        cuasi: columnas del cuasi-identificador.
        k_objetivo: k que se quiere alcanzar.

    Returns:
        La tabla superviviente y la fraccion de filas suprimidas.
    """
    tam = df.groupby(cuasi, observed=True)[cuasi[0]].transform("size")
    superviven = df[tam >= k_objetivo]
    return superviven, 1.0 - len(superviven) / len(df)


def error_consulta(original: pd.DataFrame, publicada: pd.DataFrame,
                   sensible: str) -> float:
    """Error que introduce la anonimizacion en una consulta real.

    La consulta: prevalencia de cada diagnostico por provincia, que es
    para lo que suele servir una publicacion sanitaria. Se compara la
    respuesta sobre la tabla publicada con la verdad, y se devuelve el
    error absoluto medio en puntos porcentuales.

    Args:
        original: tabla completa, sin anonimizar.
        publicada: tabla tras generalizar y suprimir.
        sensible: columna del atributo sensible.

    Returns:
        Error absoluto medio, en puntos porcentuales.
    """
    if not len(publicada):
        return float("nan")
    def prevalencia(df: pd.DataFrame) -> pd.Series:
        return (df.groupby(["provincia", sensible], observed=True)
                .size()
                / df.groupby("provincia", observed=True).size())
    v, p = prevalencia(original), prevalencia(publicada)
    alineada = p.reindex(v.index, fill_value=0.0)
    return float(100 * (alineada - v).abs().mean())


def main() -> None:
    """Recorre la escalera y guarda la tabla de resultados."""
    log = crear_registro("cap03.metricas")
    fijar_semillas()

    df = preparar(pd.read_parquet(DATOS))
    log.info("dataset: %d filas · %d diagnósticos distintos",
             len(df), df[SENSIBLE].nunique())

    filas = []
    for nombre, cuasi in progreso(PELDANOS, len(PELDANOS), log,
                                  cada=1, tarea="peldaños"):
        k = k_anonimato(df, cuasi)
        l_d = l_diversidad(df, cuasi, SENSIBLE)
        l_e = l_diversidad(df, cuasi, SENSIBLE, entropica=True)
        t = t_cercania(df, cuasi, SENSIBLE)
        r = riesgos(df, cuasi, n_poblacion=POBLACION_ES)
        u = utilidad(df, cuasi)
        clases = df.groupby(cuasi, observed=True).ngroups
        log.info("%-20s k=%-5d l=%-4.0f l_ent=%-4.1f t=%.2f  "
                 "riesgo fiscal max=%.3f medio=%.4f  únicas=%.1f%%  "
                 "utilidad=%.1f bits  clases=%d",
                 nombre, k, l_d, l_e, t, r["fiscal_max"],
                 r["fiscal_medio"], 100 * r["unicas"], u, clases)
        sup5, frac5 = suprimir_hasta(df, cuasi, 5)
        u5 = utilidad(sup5, cuasi) if len(sup5) else 0.0
        err = error_consulta(df, sup5, SENSIBLE)
        log.info("%-20s   para k>=5: suprimir %.1f%% (quedan %d, "
                 "utilidad %.1f bits) · error de la consulta de "
                 "prevalencia: %.2f puntos", "", 100 * frac5,
                 len(sup5), u5, err)
        filas.append({
            "supresion_k5": frac5, "utilidad_k5": u5,
            "error_consulta": err,
            "peldano": nombre, "k": k, "l_distinct": l_d,
            "l_entropica": l_e, "t": t,
            "riesgo_fiscal_max": r["fiscal_max"],
            "riesgo_fiscal_medio": r["fiscal_medio"],
            "riesgo_periodista_max": r["periodista_max"],
            "unicas": r["unicas"], "utilidad_bits": u,
            "clases": clases,
        })

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    log.info("guardado %s", SALIDA.relative_to(RAIZ))

    log.info("NINGÚN peldaño alcanza k>=5 por generalización sola: "
             "k es un mínimo y basta una fila rara para dejarlo en 1")
    barato = min(filas, key=lambda f: f["supresion_k5"])
    log.info("supresión más barata para k>=5: «%s», %.2f%% de las "
             "filas, conservando %.1f de los %.1f bits del crudo",
             barato["peldano"], 100 * barato["supresion_k5"],
             barato["utilidad_k5"], filas[0]["utilidad_bits"])


if __name__ == "__main__":
    main()
