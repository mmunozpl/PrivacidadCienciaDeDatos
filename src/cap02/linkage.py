"""reproduce el ataque de linkage: el mecanismo de Sweeney y Netflix.

Simula una fuente auxiliar publica (padron con nombre + edad + sexo +
CP) y la cruza con la tabla «anonima» del libro por los
cuasi-identificadores. Mide cuantas filas quedan reidentificadas de
forma univoca y cuantas quedan con ambiguedad acotada.

Uso: python3 src/cap02/linkage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SALIDA = RAIZ / "data" / "processed" / "linkage_resultado.parquet"

# el adversario solo ve estas columnas en las dos tablas
CUASI_PUBLICO = ["edad", "sexo", "codigo_postal"]
# fraccion de la tabla que el adversario tiene en su fuente auxiliar
COBERTURA = 0.60


def fuente_auxiliar(df: pd.DataFrame, rng: np.random.Generator,
                    cobertura: float = COBERTURA) -> pd.DataFrame:
    """Construye la fuente auxiliar del adversario.

    Un padron, censo o perfil publico: lleva NOMBRE y los mismos
    cuasi-identificadores demograficos, pero ningun dato sensible.

    Args:
        df: tabla completa de la que se toma la muestra.
        rng: generador sembrado.
        cobertura: fraccion de personas presentes en la auxiliar.

    Returns:
        Tabla con nombre y cuasi-identificadores.
    """
    idx = rng.choice(len(df), size=int(cobertura * len(df)),
                     replace=False)
    aux = df.iloc[np.sort(idx)][CUASI_PUBLICO].copy()
    aux.insert(0, "nombre",
               [f"persona_{i:05d}" for i in np.sort(idx)])
    return aux.reset_index(drop=True)


def atacar(anonima: pd.DataFrame,
           aux: pd.DataFrame) -> pd.DataFrame:
    """Cruza las dos tablas por los cuasi-identificadores.

    Args:
        anonima: tabla publicada sin identificadores directos.
        aux: fuente auxiliar con nombre.

    Returns:
        El join, con el numero de candidatos por fila del cruce.
    """
    cruce = anonima.merge(aux, on=CUASI_PUBLICO, how="inner")
    # candidatos: cuantos nombres compiten por cada combinacion
    cruce["candidatos"] = cruce.groupby(
        CUASI_PUBLICO, observed=True
    )["nombre"].transform("size")
    return cruce


def main() -> None:
    """Ejecuta el ataque y mide su exito."""
    log = crear_registro("cap02.linkage")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS)
    # la tabla «anonimizada»: se suprime el identificador directo
    anonima = df.drop(columns=["num_historia"])
    log.info("tabla publicada: %d filas, sin identificadores directos",
             len(anonima))

    aux = fuente_auxiliar(df, rng)
    log.info("fuente auxiliar del adversario: %d personas (%.0f%% de "
             "cobertura), columnas %s",
             len(aux), 100 * COBERTURA, list(aux.columns))

    cruce = atacar(anonima, aux)
    unicos = cruce[cruce["candidatos"] == 1]
    log.info("filas del cruce: %d", len(cruce))
    log.info("REIDENTIFICADAS de forma univoca: %d (%.1f%% de la tabla "
             "publicada, %.1f%% de las personas de la auxiliar)",
             len(unicos), 100 * len(unicos) / len(anonima),
             100 * len(unicos) / len(aux))

    for k in (2, 3, 5):
        n = int((cruce["candidatos"] <= k).sum())
        log.info("acotadas a %d candidatos o menos: %d (%.1f%%)",
                 k, n, 100 * n / len(anonima))

    # el dano: el diagnostico queda unido a un nombre
    fuga = unicos["diagnostico"].value_counts(normalize=True)
    log.info("diagnosticos expuestos con nombre y apellido: %s",
             ", ".join(f"{d} {100*p:.0f}%" for d, p in
                       fuga.head(3).items()))

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    unicos.to_parquet(SALIDA, index=False)
    log.info("guardado %s", SALIDA.relative_to(RAIZ))
    muestra_final(unicos[["nombre", "edad", "sexo", "codigo_postal",
                          "diagnostico"]], log)


if __name__ == "__main__":
    main()
