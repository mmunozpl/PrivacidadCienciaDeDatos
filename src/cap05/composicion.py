"""como se suma el presupuesto cuando se pregunta mas de una vez.

Una consulta con epsilon=1 no es el problema. El problema es la
segunda, y la trigesima. Este script mide las tres formas de sumar y
la que no suma:

1. BASICA: k consultas de epsilon dan k*epsilon. Exacta, sin letra
   pequena, y ruinosa en cuanto k crece.
2. AVANZADA (Dwork, Rothblum y Vadhan, 2010): el termino dominante
   pasa a crecer con la raiz de k, a cambio de aceptar un delta
   adicional. Solo compensa a partir de cierto k, y aqui se mide
   exactamente cual.
3. zCDP (Bun y Steinke, 2016): se contabiliza en rho, los rho se suman
   sin correcciones y el delta aparece una sola vez, al traducir al
   final. Es la contabilidad que uso el censo de EE. UU.
4. PARALELA: si las consultas recaen sobre grupos DISJUNTOS de
   personas, el presupuesto no se suma en absoluto. Las 311 casillas
   del histograma del libro cuestan epsilon UNA vez, no 311. Es la
   diferencia entre un sistema utilizable y uno que no lo es.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap05/composicion.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final
from comun.ruido_dp import (composicion_avanzada, composicion_basica,
                            eps_de_rho, mecanismo_laplace)

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SAL_CONTA = RAIZ / "data" / "processed" / "dp_composicion.csv"
SAL_PARALELA = RAIZ / "data" / "processed" / "dp_paralela.csv"

DELTA = 1e-5
DELTA_PRIMA = 1e-6        # el delta extra que cobra la composicion avanzada
EPS_UNIDAD = 0.1          # presupuesto de cada consulta individual
KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def contabilidades(eps_unidad: float, delta_prima: float,
                   delta_traduccion: float) -> pd.DataFrame:
    """Calcula el epsilon total por las tres contabilidades.

    Args:
        eps_unidad: presupuesto de cada consulta.
        delta_prima: delta adicional de la composicion avanzada.
        delta_traduccion: delta al que se traduce el rho de zCDP.

    Returns:
        Tabla con una fila por k y el epsilon total de cada metodo.
    """
    # un mecanismo epsilon-DP puro satisface (eps^2/2)-zCDP
    # (Bun y Steinke, TCC 2016-B, prop. 1.4)
    rho_unidad = eps_unidad ** 2 / 2
    filas = []
    for k in KS:
        basica, _ = composicion_basica(eps_unidad, 0.0, k)
        avanzada, _ = composicion_avanzada(eps_unidad, 0.0, k,
                                           delta_prima)
        zcdp = eps_de_rho(k * rho_unidad, delta_traduccion)
        filas.append({"k": k, "basica": basica, "avanzada": avanzada,
                      "zcdp": zcdp,
                      "mejor": min((basica, "basica"),
                                   (avanzada, "avanzada"),
                                   (zcdp, "zcdp"))[1]})
    return pd.DataFrame(filas)


def presupuesto_inverso(eps_total: float, delta: float,
                        delta_prima: float) -> pd.DataFrame:
    """Reparte un presupuesto total entre k consultas, por cada metodo.

    Es la pregunta que se hace de verdad: «tengo epsilon=1 para todo el
    ano, cuanto ruido le toca a cada consulta si voy a hacer k».

    Args:
        eps_total: presupuesto global disponible.
        delta: delta global disponible.
        delta_prima: delta adicional de la composicion avanzada.

    Returns:
        Tabla con el epsilon por consulta que permite cada metodo y el
        ruido de Laplace que implica para un recuento.
    """
    def invertir(f, objetivo: float) -> float:
        """Mayor eps por consulta con f(eps) <= objetivo, por biseccion."""
        lo, hi = 1e-9, 10.0
        while f(hi) < objetivo:
            hi *= 2
        for _ in range(200):
            medio = 0.5 * (lo + hi)
            if f(medio) <= objetivo:
                lo = medio
            else:
                hi = medio
        return lo

    filas = []
    for k in KS:
        e_bas = eps_total / k
        e_ava = invertir(
            lambda e: composicion_avanzada(e, 0.0, k, delta_prima)[0],
            eps_total)
        e_zcdp = invertir(
            lambda e: eps_de_rho(k * e ** 2 / 2, delta), eps_total)
        filas.append({"k": k,
                      "eps_basica": e_bas, "ruido_basica": 1 / e_bas,
                      "eps_avanzada": e_ava, "ruido_avanzada": 1 / e_ava,
                      "eps_zcdp": e_zcdp, "ruido_zcdp": 1 / e_zcdp})
    return pd.DataFrame(filas)


def composicion_paralela(df: pd.DataFrame, eps: float,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Mide lo que ahorra publicar sobre grupos disjuntos.

    El histograma provincia x diagnostico particiona a la poblacion:
    cada persona cae en UNA casilla. Publicarlo entero cuesta epsilon
    una sola vez. Tratarlo como 311 consultas independientes obliga a
    repartir el presupuesto entre las 311, con 311 veces mas ruido en
    cada una para la misma garantia global.

    Args:
        df: poblacion con provincia y diagnostico.
        eps: presupuesto global de la publicacion.
        rng: generador sembrado.

    Returns:
        Tabla con el error medio por casilla bajo cada tratamiento.
    """
    conteos = (df.groupby(["provincia", "diagnostico"], observed=True)
               .size().to_numpy().astype(float))
    d = len(conteos)
    filas = []
    for nombre, eps_casilla in (("paralela (grupos disjuntos)", eps),
                                ("secuencial (mal contabilizada)",
                                 eps / d)):
        err = [float(np.abs(mecanismo_laplace(conteos, 1.0,
                                              eps_casilla, rng)
                            - conteos).mean()) for _ in range(100)]
        filas.append({"tratamiento": nombre, "casillas": d,
                      "eps_por_casilla": eps_casilla,
                      "error_medio": float(np.mean(err))})
    return pd.DataFrame(filas)


def main() -> None:
    """Ejecuta las mediciones de composicion y guarda sus tablas."""
    log = crear_registro("cap05.composicion")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS).assign(
        provincia=lambda t: t["codigo_postal"].astype(str).str[:2])

    # ── 1) las tres contabilidades, a igual consulta ────────────────
    con = contabilidades(EPS_UNIDAD, DELTA_PRIMA, DELTA)
    log.info("k consultas de epsilon=%.2f cada una (delta'=%.0e para "
             "la avanzada, delta=%.0e para traducir zCDP):",
             EPS_UNIDAD, DELTA_PRIMA, DELTA)
    for _, f in con.iterrows():
        log.info("  k=%-5d basica %8.2f · avanzada %7.2f · zCDP %7.2f "
                 "→ gana %s", int(f["k"]), f["basica"], f["avanzada"],
                 f["zcdp"], f["mejor"])
    corte = con[con["avanzada"] < con["basica"]]
    log.info("  La composicion avanzada NO siempre mejora: solo desde "
             "k=%s. Por debajo cobra el delta' y no da nada a cambio.",
             int(corte["k"].min()) if len(corte) else "nunca")
    corte_z = con[con["zcdp"] < con["avanzada"]]
    log.info("  zCDP mejora a la avanzada desde k=%s, y ademas evita "
             "elegir un delta' en cada paso.",
             int(corte_z["k"].min()) if len(corte_z) else "nunca")
    SAL_CONTA.parent.mkdir(parents=True, exist_ok=True)
    con.to_csv(SAL_CONTA, index=False)
    log.info("guardado %s", SAL_CONTA.relative_to(RAIZ))

    # ── 2) el reparto inverso: cuanto ruido toca a cada consulta ────
    inv = presupuesto_inverso(1.0, DELTA, DELTA_PRIMA)
    log.info("con un presupuesto TOTAL de epsilon=1 y delta=%.0e, "
             "escala del ruido de Laplace por consulta de recuento:",
             DELTA)
    for _, f in inv.iloc[::3].iterrows():
        log.info("  k=%-5d basica %8.1f · avanzada %7.1f · zCDP %7.1f "
                 "personas de ruido", int(f["k"]), f["ruido_basica"],
                 f["ruido_avanzada"], f["ruido_zcdp"])
    f1000 = inv[inv["k"] == 1000].iloc[0]
    log.info("  A k=1000: la contabilidad basica exige %.0f personas "
             "de ruido por consulta y zCDP se conforma con %.1f, un "
             "factor %.1f. Contabilizar mejor no cambia la garantia: "
             "cambia cuanto se paga por ella.", f1000["ruido_basica"],
             f1000["ruido_zcdp"],
             f1000["ruido_basica"] / f1000["ruido_zcdp"])
    inv.to_csv(SAL_CONTA.with_name("dp_composicion_inversa.csv"),
               index=False)

    # ── 3) la composicion que no cuesta: grupos disjuntos ───────────
    par = composicion_paralela(df, 1.0, rng)
    log.info("histograma de %d casillas con presupuesto global "
             "epsilon=1:", int(par["casillas"].iloc[0]))
    for _, f in par.iterrows():
        log.info("  %-32s epsilon por casilla %.5f · error medio %.2f "
                 "personas", f["tratamiento"], f["eps_por_casilla"],
                 f["error_medio"])
    log.info("  Factor entre ambos: %.0f. Cada persona esta en UNA "
             "casilla, luego la publicacion entera cuesta epsilon una "
             "vez. No reconocerlo es el error de contabilidad mas caro "
             "que se puede cometer.",
             par["error_medio"].max() / par["error_medio"].min())
    par.to_csv(SAL_PARALELA, index=False)
    log.info("guardado %s", SAL_PARALELA.relative_to(RAIZ))

    muestra_final(inv, log)


if __name__ == "__main__":
    main()
