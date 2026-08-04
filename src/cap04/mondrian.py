"""k-anonimizacion multidimensional con Mondrian, y su coste medido.

Mondrian (LeFevre, DeWitt y Ramakrishnan, ICDE 2006) particiona el
espacio de cuasi-identificadores en regiones y generaliza cada una por
separado: donde hay muchos datos conserva resolucion, y solo agranda
las zonas ralas. Aqui se implementa y se compara con la generalizacion
GLOBAL del capitulo 3 —misma k, mismo dataset— para medir cuanta
utilidad se gana.

Escribe data/processed/mondrian.csv para la figura del libro.

Uso: python3 src/cap04/mondrian.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final, progreso

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
SALIDA = RAIZ / "data" / "processed" / "mondrian.csv"
SALIDA_TABLA = RAIZ / "data" / "processed" / "mondrian_publicada.parquet"

CUASI = ["edad", "cp_num", "sexo_num"]
KS = [2, 5, 10, 25, 50]


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica los cuasi-identificadores como numeros ordenables."""
    return df.assign(cp_num=df["codigo_postal"].astype(int),
                     sexo_num=(df["sexo"] == "M").astype(int))


def anchura_normalizada(df: pd.DataFrame, col: str,
                        rangos: dict[str, float]) -> float:
    """Anchura del intervalo de una columna, en [0, 1]."""
    if rangos[col] == 0:
        return 0.0
    return float(df[col].max() - df[col].min()) / rangos[col]


def mondrian(df: pd.DataFrame, cuasi: list[str], k: int,
             rangos: dict[str, float]) -> list[pd.DataFrame]:
    """Particiona recursivamente por la dimension mas ancha.

    En cada paso se elige el atributo con mayor rango normalizado y se
    corta por su mediana; si alguno de los dos lados quedaria con menos
    de k filas, la region no se parte y se publica tal cual.

    Args:
        df: tabla (o region) a particionar.
        cuasi: columnas del cuasi-identificador.
        k: tamano minimo de region.
        rangos: rango global de cada columna, para normalizar.

    Returns:
        Lista de regiones, cada una con al menos k filas.
    """
    if len(df) < 2 * k:
        return [df]
    dim = max(cuasi, key=lambda c: anchura_normalizada(df, c, rangos))
    mediana = df[dim].median()
    izq, der = df[df[dim] <= mediana], df[df[dim] > mediana]
    if len(izq) < k or len(der) < k:
        return [df]
    return (mondrian(izq, cuasi, k, rangos)
            + mondrian(der, cuasi, k, rangos))


def publicar(regiones: list[pd.DataFrame],
             cuasi: list[str]) -> pd.DataFrame:
    """Sustituye cada valor por el intervalo de su region."""
    salida = []
    for r in regiones:
        g = r.copy()
        for c in cuasi:
            lo, hi = r[c].min(), r[c].max()
            g[c] = f"[{lo}-{hi}]" if lo != hi else str(lo)
        salida.append(g)
    return pd.concat(salida)


def utilidad(publicada: pd.DataFrame, cuasi: list[str]) -> float:
    """Entropia conjunta que sobrevive, en bits."""
    comb = publicada[cuasi].astype(str).agg("|".join, axis=1)
    p = comb.value_counts(normalize=True).to_numpy()
    return float(-(p * np.log2(p)).sum())


def error_consulta(original: pd.DataFrame,
                   publicada: pd.DataFrame) -> tuple[float, float]:
    """Error de la consulta del cap. 3 sobre la tabla de Mondrian.

    La consulta es la prevalencia de cada diagnostico por provincia.
    Mondrian no suprime filas, pero cuando una region abarca varias
    provincias esa fila deja de ser atribuible a una: se pierde para
    la consulta aunque siga publicada. Se mide el error sobre las
    atribuibles y cuantas se han perdido.

    Args:
        original: tabla sin anonimizar (con codigo_postal).
        publicada: salida de Mondrian, con cp_num como intervalo.

    Returns:
        Error absoluto medio en puntos y fraccion no atribuible.
    """
    def provincia_de(celda: str) -> str | None:
        if not celda.startswith("["):
            return f"{int(celda):05d}"[:2]
        lo, hi = celda[1:-1].split("-")
        p_lo, p_hi = f"{int(lo):05d}"[:2], f"{int(hi):05d}"[:2]
        return p_lo if p_lo == p_hi else None

    pub = publicada.copy()
    pub["provincia"] = pub["cp_num"].map(provincia_de)
    perdidas = float(pub["provincia"].isna().mean())
    pub = pub.dropna(subset=["provincia"])
    orig = original.assign(
        provincia=original["codigo_postal"].astype(str).str[:2])

    def prev(df):
        return (df.groupby(["provincia", "diagnostico"], observed=True)
                .size() / df.groupby("provincia", observed=True).size())

    if not len(pub):
        return float("nan"), perdidas
    v, q = prev(orig), prev(pub)
    return float(100 * (q.reindex(v.index, fill_value=0.0) - v)
                 .abs().mean()), perdidas


def main() -> None:
    """Ejecuta Mondrian para varias k y mide su coste."""
    log = crear_registro("cap04.mondrian")
    fijar_semillas()

    df = preparar(pd.read_parquet(DATOS))
    rangos = {c: float(df[c].max() - df[c].min()) for c in CUASI}
    u0 = utilidad(df.astype({c: str for c in CUASI}), CUASI)
    log.info("tabla original: %d filas · utilidad %.2f bits", len(df), u0)

    filas = []
    for k in progreso(KS, len(KS), log, cada=1, tarea="valores de k"):
        regiones = mondrian(df, CUASI, k, rangos)
        pub = publicar(regiones, CUASI)
        tam = pub.groupby(CUASI, observed=True)[CUASI[0]].transform("size")
        u = utilidad(pub, CUASI)
        log.info("k=%-3d → %4d regiones · k real=%d · utilidad %.2f "
                 "bits (%.0f%% de la original) · sin suprimir nada",
                 k, len(regiones), int(tam.min()), u, 100 * u / u0)
        err, perd = error_consulta(df, pub)
        log.info("      consulta de prevalencia por provincia: %.2f "
                 "puntos de error · %.1f%% de filas no atribuibles",
                 err, 100 * perd)
        filas.append({"k": k, "regiones": len(regiones),
                      "k_real": int(tam.min()), "utilidad": u,
                      "pct_utilidad": 100 * u / u0,
                      "error_consulta": err, "no_atribuibles": perd})
        if k == 5:
            SALIDA_TABLA.parent.mkdir(parents=True, exist_ok=True)
            pub.to_parquet(SALIDA_TABLA, index=False)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SALIDA, index=False)
    log.info("guardado %s", SALIDA.relative_to(RAIZ))

    # comparacion con la generalizacion GLOBAL del capitulo 3
    global_ = RAIZ / "data" / "processed" / "escalera_generalizacion.csv"
    if global_.exists():
        esc = pd.read_csv(global_)
        mejor = esc.loc[esc["utilidad_k5"].idxmax()]
        m5 = next(f for f in filas if f["k"] == 5)
        log.info("COMPARACIÓN a k=5, con la métrica que importa "
                 "(error de la consulta):")
        log.info("  global (cap. 3, «%s»): %.2f puntos, suprimiendo "
                 "el %.1f%% de las filas",
                 mejor["peldano"], mejor["error_consulta"],
                 100 * mejor["supresion_k5"])
        log.info("  Mondrian: %.2f puntos, sin suprimir ninguna "
                 "(%.1f%% no atribuibles a provincia)",
                 m5["error_consulta"], 100 * m5["no_atribuibles"])

    # ── variante que respeta lo que la consulta necesita ────────────
    # Dar la provincia como una dimension mas no basta: Mondrian la
    # corta en rangos ([1-28]) que siguen atravesando fronteras. Lo
    # que la consulta necesita es que la provincia NO se generalice
    # nunca, es decir, que sea CLAVE DE PARTICION y no dimension.
    df_p = df.assign(provincia=df["codigo_postal"].astype(str).str[:2])
    cuasi_p = ["edad", "sexo_num"]
    trozos, sueltas = [], 0
    for prov, grupo in df_p.groupby("provincia", observed=True):
        if len(grupo) < 5:
            sueltas += len(grupo)
            continue
        rg = {c: float(grupo[c].max() - grupo[c].min()) for c in cuasi_p}
        trozos.append(publicar(mondrian(grupo, cuasi_p, 5, rg),
                               cuasi_p))
    pub_p = pd.concat(trozos)
    u_p = utilidad(pub_p, cuasi_p + ["provincia"])
    prev = lambda d: (d.groupby(["provincia", "diagnostico"],
                                observed=True).size()
                      / d.groupby("provincia", observed=True).size())
    v, q = prev(df_p), prev(pub_p)
    err_p = float(100 * (q.reindex(v.index, fill_value=0.0) - v)
                  .abs().mean())
    log.info("VARIANTE con la provincia como CLAVE DE PARTICIÓN "
             "(no se generaliza nunca), k=5: utilidad %.2f bits · "
             "error de la consulta %.2f puntos · %d filas suprimidas "
             "(%.2f%%)", u_p, err_p, sueltas,
             100 * sueltas / len(df_p))

    pub5 = pd.read_parquet(SALIDA_TABLA)
    muestra_final(pub5[["edad", "cp_num", "sexo_num", "diagnostico"]],
                  log)


if __name__ == "__main__":
    main()
