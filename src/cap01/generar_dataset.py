"""genera el dataset sintetico de cuasi-identificadores del libro.

Construye una poblacion sintetica CALIBRADA con las cifras reales del
INE (data/ine/, Cifras de Poblacion a 1 de enero de 2025, descargadas
con herramientas/descargar_ine.py):

- (edad, sexo) se muestrea de la piramide nacional real, conjunta.
- el codigo postal lleva el prefijo provincial real (dos digitos del
  codigo INE de provincia), con la provincia muestreada por su
  poblacion; el sufijo de tres digitos es estilizado (decae hacia los
  numeros altos, como decaen los CP perifericos).
- profesion y diagnostico son estilizados: categorias y pesos
  plausibles, sin pretension de exactitud estadistica.

Guarda el parquet, valida el esquema anotado y mide el riesgo de
partida (k-anonimato y unicidad muestral).

Uso: python3 src/cap01/generar_dataset.py
     (si data/ine/ no existe: python3 herramientas/descargar_ine.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.esquema import columnas, validar
from comun.registro import crear_registro, muestra_final, progreso
from comun.reident import k_anonimato, unicidad_muestral

N = 20_000
INE = RAIZ / "data" / "ine"
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
CUASI = columnas(ESQUEMA, "cuasi")

PROFESIONES = ["sanidad", "educacion", "comercio", "industria",
               "tecnologia", "administracion", "agricultura",
               "hosteleria", "transporte", "otras"]
DIAGNOSTICOS = ["ninguno", "hipertension", "diabetes", "asma",
                "dermatitis", "ansiedad"]


def generar(rng: np.random.Generator) -> pd.DataFrame:
    """Construye la poblacion sintetica calibrada con el INE.

    Args:
        rng: generador sembrado (ver comun/determinismo.py).

    Returns:
        Tabla con identificador directo, cuasi-identificadores y el
        atributo sensible.
    """
    # (edad, sexo) conjuntos desde la piramide real
    piramide = pd.read_csv(INE / "edad_sexo.csv")
    celdas = piramide.melt(id_vars=["edad"],
                           value_vars=["hombres", "mujeres"],
                           var_name="sexo", value_name="poblacion")
    celdas["sexo"] = celdas["sexo"].map(
        {"hombres": "M", "mujeres": "F"}
    )
    idx = rng.choice(len(celdas), size=N,
                     p=celdas["poblacion"] / celdas["poblacion"].sum())
    edad = celdas["edad"].to_numpy()[idx]
    sexo = celdas["sexo"].to_numpy()[idx]

    # provincia por poblacion real; el CP hereda su prefijo INE
    prov = pd.read_csv(INE / "provincias.csv",
                       dtype={"codigo": str})
    codigo = rng.choice(prov["codigo"], size=N,
                        p=prov["poblacion"] / prov["poblacion"].sum())
    # sufijo estilizado: decae hacia los numeros altos
    pesos = 1.0 / (np.arange(1000) + 5.0)
    sufijo = rng.choice(1000, size=N, p=pesos / pesos.sum())
    codigo_postal = np.array(
        [f"{c}{s:03d}" for c, s in zip(codigo, sufijo)]
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

    log.info("generando %d registros con marginales INE 2025", N)
    partes = [generar(rng) for _ in progreso(range(1), 1, log,
                                            cada=1, tarea="sintesis")]
    df = pd.concat(partes, ignore_index=True)
    validar(df, ESQUEMA)

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
