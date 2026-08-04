"""genera el dataset sintetico de cuasi-identificadores del libro.

Construye una poblacion sintetica con identificadores, cuasi-
identificadores y un atributo sensible, la guarda en parquet y mide
su riesgo de partida (k-anonimato y unicidad muestral). Es el dataset
transversal de la Parte I.

Uso: python3 codigo/cap01/generar_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "codigo"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final, progreso
from comun.reident import k_anonimato, unicidad_muestral

N = 20_000
SALIDA = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"

# el esquema anotado viaja con el dato: papel de cada columna frente
# a la reidentificacion (cap. 1, seccion 1.3)
ESQUEMA = {
    "num_historia": "identificador",  # se suprime al preparar
    "edad": "cuasi",
    "sexo": "cuasi",
    "codigo_postal": "cuasi",
    "profesion": "cuasi",
    "diagnostico": "sensible",        # art. 9: salud
}
CUASI = [c for c, papel in ESQUEMA.items() if papel == "cuasi"]

PROFESIONES = ["sanidad", "educacion", "comercio", "industria",
               "tecnologia", "administracion", "agricultura",
               "hosteleria", "transporte", "otras"]
DIAGNOSTICOS = ["ninguno", "hipertension", "diabetes", "asma",
                "dermatitis", "ansiedad"]


def generar(rng: np.random.Generator) -> pd.DataFrame:
    """Construye la poblacion sintetica fila a fila vectorizada.

    Args:
        rng: generador sembrado (ver comun/determinismo.py).

    Returns:
        Tabla con identificador directo, cuasi-identificadores y el
        atributo sensible.
    """
    # edades con piramide simplificada y codigos postales sesgados
    edad = np.clip(rng.normal(45, 19, N).round(), 0, 99).astype(int)
    sexo = rng.choice(["M", "F"], size=N)
    codigo_postal = rng.choice(
        [f"{p:05d}" for p in rng.integers(1000, 52999, 400)],
        size=N,
    )
    profesion = rng.choice(PROFESIONES, size=N,
                           p=np.linspace(2, 1, 10) / 15)
    diagnostico = rng.choice(DIAGNOSTICOS, size=N,
                             p=[0.55, 0.15, 0.10, 0.08, 0.07, 0.05])
    return pd.DataFrame({
        "num_historia": np.arange(1, N + 1),
        "edad": edad,
        "sexo": sexo,
        "codigo_postal": codigo_postal,
        "profesion": profesion,
        "diagnostico": diagnostico,
    })


def main() -> None:
    """Genera, guarda, mide el riesgo y verifica el artefacto."""
    log = crear_registro("cap01.generar_dataset")
    rng = fijar_semillas()

    log.info("generando %d registros sinteticos", N)
    partes = [generar(rng) for _ in progreso(range(1), 1, log,
                                            cada=1, tarea="sintesis")]
    df = pd.concat(partes, ignore_index=True)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SALIDA, index=False)
    log.info("guardado %s (%.1f kB)", SALIDA.relative_to(RAIZ),
             SALIDA.stat().st_size / 1024)
    for col, papel in ESQUEMA.items():
        log.info("esquema: %s -> %s", col, papel)

    # riesgo de partida sobre los cuasi-identificadores del libro
    k = k_anonimato(df, CUASI)
    unicos = unicidad_muestral(df, CUASI)
    log.info("k-anonimato con %s: k=%d", CUASI, k)
    log.info("unicidad muestral: %.1f%% de filas unicas",
             100 * unicos)

    # verificacion final del libro: quince observaciones aleatorias
    muestra_final(df, log)


if __name__ == "__main__":
    main()
