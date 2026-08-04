"""descarga del INE las marginales que calibran el dataset del libro.

Fuente: API Tempus3 del INE (servicios.ine.es/wstempus), operacion
ECP (Estadistica Continua de Poblacion / Cifras de Poblacion):

- tabla 56934: poblacion residente por sexo y edad simple (nacional)
- tabla 56945: poblacion residente por provincias (filtrada a todas
  las edades y ambos sexos)

Escribe data/ine/edad_sexo.csv y data/ine/provincias.csv con el anyo
de referencia como columna. Se ejecuta a mano cuando toque refrescar
las cifras; el resultado queda versionado para reproducibilidad.

Uso: python3 herramientas/descargar_ine.py
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from comun.registro import crear_registro

BASE = "https://servicios.ine.es/wstempus/js/ES"
DESTINO = RAIZ / "data" / "ine"

# filtros de la tabla 56945: edad «Todas las edades» y sexo «Total»
FILTRO_PROVINCIAS = "tv=356:15668&tv=18:451"


def descargar(url: str) -> list | dict:
    """Descarga y decodifica una respuesta JSON de la API del INE.

    Args:
        url: peticion completa a wstempus.

    Returns:
        El JSON decodificado.
    """
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def edad_sexo() -> pd.DataFrame:
    """Construye la piramide nacional edad simple x sexo.

    Returns:
        Tabla con columnas edad (0..100), hombres, mujeres y anyo.
    """
    series = descargar(f"{BASE}/DATOS_TABLA/56934?nult=1")
    filas = {}
    anyo = None
    for s in series:
        partes = [p.strip() for p in s["Nombre"].split(".")]
        edad_txt, sexo = partes[1], partes[2]
        # edades simples («0 años» ... «104 años», «1 año») y el
        # cierre «105 y más años»; se descartan los agregados
        # («Todas las edades», «85 y más años», «100 y más años»)
        m = re.fullmatch(r"(\d+) años?", edad_txt)
        if m:
            edad = int(m.group(1))
        elif edad_txt == "105 y más años":
            edad = 105
        else:
            continue
        dato = s["Data"][0]
        anyo = dato["Anyo"]
        filas.setdefault(edad, {})[sexo.lower()] = dato["Valor"]
    df = (pd.DataFrame.from_dict(filas, orient="index")
          .rename_axis("edad").sort_index().reset_index())
    df = df[["edad", "hombres", "mujeres"]].astype(
        {"hombres": int, "mujeres": int}
    )
    df["anyo"] = anyo
    return df


def provincias() -> pd.DataFrame:
    """Construye la poblacion por provincia con su codigo INE.

    El codigo de provincia coincide con los dos primeros digitos del
    codigo postal, que es lo que el generador necesita.

    Returns:
        Tabla con columnas codigo, provincia, poblacion y anyo.
    """
    valores = descargar(f"{BASE}/VALORES_GRUPOSTABLA/56945/113389")
    codigos = {v["Nombre"]: v["Codigo"] for v in valores if v["Codigo"]}
    series = descargar(
        f"{BASE}/DATOS_TABLA/56945?nult=1&{FILTRO_PROVINCIAS}"
    )
    filas = []
    for s in series:
        partes = [p.strip() for p in s["Nombre"].split(".")]
        nombre = partes[2]
        if nombre not in codigos:
            # se descarta el agregado nacional
            continue
        dato = s["Data"][0]
        filas.append({
            "codigo": codigos[nombre],
            "provincia": nombre,
            "poblacion": int(dato["Valor"]),
            "anyo": dato["Anyo"],
        })
    return pd.DataFrame(filas).sort_values("codigo",
                                           ignore_index=True)


def main() -> None:
    """Descarga las dos marginales y las deja en data/ine/."""
    log = crear_registro("herramientas.descargar_ine")
    DESTINO.mkdir(parents=True, exist_ok=True)

    piramide = edad_sexo()
    total = int(piramide[["hombres", "mujeres"]].to_numpy().sum())
    piramide.to_csv(DESTINO / "edad_sexo.csv", index=False)
    log.info("edad_sexo.csv: %d edades, poblacion %d (anyo %d)",
             len(piramide), total, piramide["anyo"].iloc[0])

    prov = provincias()
    prov.to_csv(DESTINO / "provincias.csv", index=False)
    log.info("provincias.csv: %d provincias, poblacion %d (anyo %d)",
             len(prov), prov["poblacion"].sum(),
             prov["anyo"].iloc[0])


if __name__ == "__main__":
    main()
