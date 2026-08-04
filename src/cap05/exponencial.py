"""cuando sumar ruido no sirve: el mecanismo exponencial.

Laplace y Gauss suman ruido a un numero. Hay dos situaciones en las
que eso no vale, y las dos aparecen en cualquier publicacion real:

1. La respuesta es una CATEGORIA. «El diagnostico mas frecuente de
   Soria» no admite sumarle 2,7: la salida tiene que ser uno de los
   seis diagnosticos, no un numero. Se comparan aqui las dos maneras
   correctas de resolverlo —el mecanismo exponencial y el argmax sobre
   recuentos ruidosos— y se mide cual acierta mas.
2. La respuesta es un ESTADISTICO DE ORDEN. La mediana de edad tiene
   sensibilidad global igual a todo el rango del dominio: en un grupo
   pequeno, cambiar a una persona puede moverla decenas de anos. Con
   Laplace calibrado a esa sensibilidad el resultado es inservible; el
   mecanismo exponencial con una utilidad basada en el RANGO tiene
   sensibilidad 1 y funciona. Aqui se mide la diferencia.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap05/exponencial.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final, progreso
from comun.ruido_dp import mecanismo_exponencial, mecanismo_laplace

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SAL_MODA = RAIZ / "data" / "processed" / "dp_moda.csv"
SAL_MEDIANA = RAIZ / "data" / "processed" / "dp_mediana.csv"

EPSILONES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
REPETICIONES = 2000
EDAD_MIN, EDAD_MAX = 0, 120     # dominio publico de la edad


def moda_exponencial(conteos: np.ndarray, candidatos: list,
                     eps: float, rng: np.random.Generator):
    """Diagnostico mas frecuente por el mecanismo exponencial.

    La utilidad de cada candidato es su recuento. Sustituir a una
    persona mueve como mucho un recuento en una unidad, luego la
    sensibilidad de la utilidad es 1.
    """
    return mecanismo_exponencial(candidatos, conteos.astype(float),
                                 1.0, eps, rng)


def moda_argmax_ruidoso(conteos: np.ndarray, candidatos: list,
                        eps: float, rng: np.random.Generator):
    """Diagnostico mas frecuente por argmax de recuentos ruidosos.

    Es el «report noisy max»: se suma ruido de Laplace de escala
    1/epsilon a cada recuento y se devuelve SOLO la posicion del
    maximo. Publicar los recuentos ruidosos costaria mas; devolver
    unicamente el argmax cuesta epsilon.
    """
    ruidosos = mecanismo_laplace(conteos.astype(float), 1.0, eps, rng)
    return candidatos[int(np.argmax(ruidosos))]


def comparar_moda(df: pd.DataFrame, rng: np.random.Generator,
                  log) -> pd.DataFrame:
    """Mide el acierto de los dos metodos por tamano de provincia."""
    candidatos = sorted(df["diagnostico"].unique())
    tabla = (df.groupby(["provincia", "diagnostico"], observed=True)
             .size().unstack(fill_value=0))
    tam = tabla.sum(axis=1).sort_values(ascending=False)
    # una provincia grande, una mediana y una pequena
    elegidas = [tam.index[0], tam.index[len(tam) // 2], tam.index[-1]]
    casos = [(p, e) for p in elegidas for e in EPSILONES]
    filas = []
    for prov, eps in progreso(casos, len(casos), log, cada=7,
                              tarea="provincia x epsilon"):
        conteos = tabla.loc[prov].to_numpy()
        verdadera = candidatos[int(np.argmax(conteos))]
        # margen: cuanto le saca la moda a la segunda. Es lo que
        # decide la dificultad del problema, mas que el tamano
        orden = np.sort(conteos)[::-1]
        acierto_exp = np.mean([
            moda_exponencial(conteos, candidatos, eps, rng)
            == verdadera for _ in range(REPETICIONES)])
        acierto_arg = np.mean([
            moda_argmax_ruidoso(conteos, candidatos, eps, rng)
            == verdadera for _ in range(REPETICIONES)])
        filas.append({"provincia": prov, "personas": int(tam[prov]),
                      "margen": int(orden[0] - orden[1]),
                      "epsilon": eps,
                      "acierto_exponencial": float(acierto_exp),
                      "acierto_argmax": float(acierto_arg)})
    return pd.DataFrame(filas)


def mediana_laplace(edades: np.ndarray, eps: float,
                    rng: np.random.Generator) -> float:
    """Mediana con ruido calibrado a su sensibilidad GLOBAL.

    La sensibilidad global de la mediana es el rango del dominio: en
    el peor caso, cambiar una persona la desplaza de un extremo al
    otro. Calibrar a esa cota es lo correcto y es lo que la hace
    inservible; calibrar a la sensibilidad OBSERVADA seria mas comodo
    y filtraria, porque esa cota depende de los propios datos.
    """
    return float(mecanismo_laplace(float(np.median(edades)),
                                   float(EDAD_MAX - EDAD_MIN), eps,
                                   rng))


def mediana_exponencial(edades: np.ndarray, eps: float,
                        rng: np.random.Generator) -> float:
    """Mediana por mecanismo exponencial con utilidad de rango.

    Cada edad posible del dominio es un candidato, y su utilidad es
    menos la distancia entre cuantas personas quedan por debajo y la
    mitad del grupo. Sustituir a una persona mueve esa cuenta en una
    unidad como mucho: sensibilidad 1, independiente del dominio.
    """
    candidatos = list(range(EDAD_MIN, EDAD_MAX + 1))
    n = len(edades)
    por_debajo = np.searchsorted(np.sort(edades), candidatos,
                                 side="left")
    utilidad = -np.abs(por_debajo - n / 2.0)
    return float(mecanismo_exponencial(candidatos, utilidad, 1.0, eps,
                                       rng))


def comparar_mediana(df: pd.DataFrame, rng: np.random.Generator,
                     log) -> pd.DataFrame:
    """Mide el error de las dos medianas por tamano de grupo."""
    tam = df.groupby("provincia", observed=True).size() \
            .sort_values(ascending=False)
    elegidas = [tam.index[0], tam.index[len(tam) // 2], tam.index[-1]]
    casos = [(p, e) for p in elegidas for e in EPSILONES]
    filas = []
    for prov, eps in progreso(casos, len(casos), log, cada=7,
                              tarea="mediana: provincia x epsilon"):
        edades = df.loc[df["provincia"] == prov, "edad"].to_numpy()
        real = float(np.median(edades))
        err_lap = np.mean([abs(mediana_laplace(edades, eps, rng)
                               - real) for _ in range(REPETICIONES)])
        err_exp = np.mean([abs(mediana_exponencial(edades, eps, rng)
                               - real) for _ in range(REPETICIONES)])
        filas.append({"provincia": prov, "personas": len(edades),
                     "epsilon": eps, "mediana_real": real,
                      "error_laplace": float(err_lap),
                      "error_exponencial": float(err_exp)})
    return pd.DataFrame(filas)


def main() -> None:
    """Ejecuta las dos comparaciones y guarda sus tablas."""
    log = crear_registro("cap05.exponencial")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS).assign(
        provincia=lambda t: t["codigo_postal"].astype(str).str[:2])

    # ── 1) una salida categorica ────────────────────────────────────
    log.info("diagnostico mas frecuente por provincia: mecanismo "
             "exponencial contra argmax de recuentos ruidosos")
    moda = comparar_moda(df, rng, log)
    for prov in moda["provincia"].unique():
        sub = moda[moda["provincia"] == prov]
        f0 = sub.iloc[0]
        log.info("  provincia %s (%d personas, la moda le saca %d a la "
                 "segunda):", prov, f0["personas"], f0["margen"])
        for _, f in sub.iterrows():
            log.info("      eps=%-5.2f exponencial %.3f · argmax "
                     "ruidoso %.3f", f["epsilon"],
                     f["acierto_exponencial"], f["acierto_argmax"])
    # solo se comparan los casos en que alguno de los dos falla: donde
    # ambos aciertan siempre, la comparacion no distingue nada
    disputa = moda[(moda[["acierto_exponencial",
                          "acierto_argmax"]] < 0.999).any(axis=1)]
    gana_arg = (disputa["acierto_argmax"]
                > disputa["acierto_exponencial"]).mean()
    log.info("  En las %d combinaciones donde alguno falla, el argmax "
             "ruidoso gana el %.0f%%. La razon es que el exponencial "
             "reparte probabilidad entre los SEIS candidatos mientras "
             "el argmax solo compite de hecho entre los dos primeros. "
             "Los dos son epsilon-DP: el exponencial no es mejor, es "
             "mas GENERAL, y es el unico de los dos que sirve cuando "
             "la utilidad no es un recuento.",
             len(disputa), 100 * gana_arg)
    SAL_MODA.parent.mkdir(parents=True, exist_ok=True)
    moda.to_csv(SAL_MODA, index=False)
    log.info("guardado %s", SAL_MODA.relative_to(RAIZ))

    # ── 2) un estadistico de orden ──────────────────────────────────
    log.info("mediana de edad: Laplace a la sensibilidad global "
             "(%d anos) contra exponencial con utilidad de rango "
             "(sensibilidad 1)", EDAD_MAX - EDAD_MIN)
    med = comparar_mediana(df, rng, log)
    for prov in med["provincia"].unique():
        sub = med[med["provincia"] == prov]
        f0 = sub.iloc[0]
        log.info("  provincia %s (%d personas, mediana real %.0f "
                 "anos):", prov, f0["personas"], f0["mediana_real"])
        for _, f in sub.iterrows():
            log.info("      eps=%-5.2f Laplace %8.2f anos de error · "
                     "exponencial %6.2f", f["epsilon"],
                     f["error_laplace"], f["error_exponencial"])
    factor = (med["error_laplace"] / med["error_exponencial"]).median()
    log.info("  El exponencial yerra %.0f veces menos, en mediana de "
             "todos los casos. No es que Laplace este mal usado: es "
             "que la mediana no tiene sensibilidad baja, y el "
             "mecanismo correcto no es el que suma ruido al resultado "
             "sino el que elige entre respuestas.", factor)
    med.to_csv(SAL_MEDIANA, index=False)
    log.info("guardado %s", SAL_MEDIANA.relative_to(RAIZ))

    muestra_final(med, log)


if __name__ == "__main__":
    main()
