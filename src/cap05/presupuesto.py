"""la misma consulta de los capitulos 3 y 4, ahora con garantia.

El hilo del libro es una consulta concreta: la prevalencia de cada
diagnostico en cada provincia. Ya se ha respondido dos veces.

  cap. 3, generalizacion global: 0,50 puntos de error, suprimiendo
          filas, y con k=1 en todos los peldanos (o sea, sin garantia).
  cap. 4, Mondrian con la provincia como clave de particion: 0,00
          puntos, sin suprimir nada, y tambien sin garantia frente a
          un adversario con informacion auxiliar.

Aqui se responde por tercera vez, con privacidad diferencial. El error
ya no es 0,00: hay que pagarlo. Lo que se compra a cambio es lo unico
que las dos respuestas anteriores no daban — una cota que vale sea
cual sea lo que el adversario sepa, incluida la tabla entera menos una
fila. Este script mide el precio.

Mide ademas dos cosas que deciden si el resultado es usable:
  - el POSPROCESADO (recortar negativos, cuadrar con el total publico)
    no gasta presupuesto y reduce el error;
  - el coste NO se reparte por igual: cae sobre las provincias
    pequenas, y con el mismo epsilon Melilla recibe un error relativo
    dos ordenes de magnitud mayor que Madrid.

Escribe tres CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap05/presupuesto.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.registro import crear_registro, muestra_final, progreso
from comun.ruido_dp import mecanismo_laplace

DATOS = RAIZ / "data" / "processed" / "poblacion_sintetica.parquet"
ESCALERA = RAIZ / "data" / "processed" / "escalera_generalizacion.csv"
MONDRIAN = RAIZ / "data" / "processed" / "mondrian.csv"
SAL_CURVA = RAIZ / "data" / "processed" / "dp_curva_epsilon.csv"
SAL_POSPRO = RAIZ / "data" / "processed" / "dp_posprocesado.csv"
SAL_PROV = RAIZ / "data" / "processed" / "dp_por_provincia.csv"

EPSILONES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
REPETICIONES = 300
SENS = 2.0                # vecindad acotada: sustituir cambia 2 casillas


def tabla_verdadera(df: pd.DataFrame) -> pd.DataFrame:
    """Recuentos exactos por provincia y diagnostico."""
    return (df.groupby(["provincia", "diagnostico"], observed=True)
            .size().unstack(fill_value=0).astype(float))


def prevalencias(conteos: pd.DataFrame) -> pd.DataFrame:
    """Convierte recuentos en porcentajes por fila (provincia)."""
    total = conteos.sum(axis=1).replace(0, np.nan)
    return 100 * conteos.div(total, axis=0)


def error_prevalencia(verdad: pd.DataFrame,
                      publicada: pd.DataFrame) -> tuple[float, float]:
    """Error medio en puntos y fraccion de provincias indefinidas.

    Con presupuestos muy bajos una provincia pequena puede salir con
    todas sus casillas a cero: entonces su prevalencia no es que sea
    mala, es que no existe. Se mide aparte en vez de promediarla.

    Args:
        verdad: recuentos exactos.
        publicada: recuentos publicados con ruido y posprocesado.

    Returns:
        Par (error absoluto medio en puntos sobre las provincias
        definidas, fraccion de provincias sin denominador).
    """
    dif = (prevalencias(publicada) - prevalencias(verdad)).abs()
    indefinidas = float(dif.isna().all(axis=1).mean())
    return float(np.nanmean(dif.to_numpy())), indefinidas


def publicar(conteos: pd.DataFrame, eps: float,
             rng: np.random.Generator, recortar: bool = False,
             cuadrar: bool = False) -> pd.DataFrame:
    """Publica el histograma con ruido y el posprocesado que se pida.

    Recortar negativos y cuadrar con un total publico son funciones de
    la salida ruidosa: por la inmunidad al posprocesado no consumen ni
    un epsilon adicional, y sin embargo mejoran el resultado.

    Args:
        conteos: recuentos exactos, provincias en filas.
        eps: presupuesto global de la publicacion.
        rng: generador sembrado.
        recortar: si es cierto, lleva a cero los recuentos negativos.
        cuadrar: si es cierto, reescala cada fila al total real de la
            provincia, que se supone publico (el padron lo es).

    Returns:
        Tabla publicada, del mismo tamano que la de entrada.
    """
    ruidosa = pd.DataFrame(
        mecanismo_laplace(conteos.to_numpy(), SENS, eps, rng),
        index=conteos.index, columns=conteos.columns)
    if recortar:
        ruidosa = ruidosa.clip(lower=0)
    if cuadrar:
        # una provincia que salga entera a cero no se puede reescalar:
        # se deja como esta y aguas abajo cuenta como indefinida
        suma = ruidosa.sum(axis=1).replace(0, np.nan)
        factor = (conteos.sum(axis=1) / suma).fillna(1.0)
        ruidosa = ruidosa.mul(factor, axis=0)
    return ruidosa


def curva(conteos: pd.DataFrame, rng: np.random.Generator,
          log) -> pd.DataFrame:
    """Error de la consulta en funcion del presupuesto."""
    filas = []
    for eps in progreso(EPSILONES, len(EPSILONES), log, cada=3,
                        tarea="epsilon"):
        medidas = [error_prevalencia(
            conteos, publicar(conteos, eps, rng, recortar=True,
                              cuadrar=True))
            for _ in range(REPETICIONES)]
        err = [m[0] for m in medidas]
        indef = [m[1] for m in medidas]
        filas.append({"epsilon": eps, "error": float(np.mean(err)),
                      "error_p95": float(np.percentile(err, 95)),
                      "pct_indefinidas": 100 * float(np.mean(indef))})
    return pd.DataFrame(filas)


def efecto_posprocesado(conteos: pd.DataFrame, eps: float,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Mide lo que aporta cada paso de posprocesado, a igual epsilon."""
    variantes = [("ninguno", False, False),
                 ("recorte de negativos", True, False),
                 ("recorte + cuadre con el total", True, True)]
    filas = []
    for nombre, rec, cua in variantes:
        err, negativos = [], []
        for _ in range(REPETICIONES):
            pub = publicar(conteos, eps, rng, recortar=rec,
                           cuadrar=cua)
            err.append(error_prevalencia(conteos, pub)[0])
            negativos.append(float((pub.to_numpy() < 0).mean()))
        filas.append({"posprocesado": nombre, "epsilon": eps,
                      "error": float(np.mean(err)),
                      "pct_negativos": 100 * float(np.mean(negativos))})
    return pd.DataFrame(filas)


def reparto_por_provincia(conteos: pd.DataFrame, eps: float,
                          rng: np.random.Generator) -> pd.DataFrame:
    """Mide el error relativo de cada provincia a igual epsilon."""
    acumulado = np.zeros(len(conteos))
    for _ in range(REPETICIONES):
        pub = publicar(conteos, eps, rng, recortar=True, cuadrar=True)
        acumulado += (prevalencias(pub)
                      - prevalencias(conteos)).abs().mean(axis=1)
    return pd.DataFrame({
        "provincia": conteos.index,
        "personas": conteos.sum(axis=1).to_numpy(),
        "error_puntos": acumulado / REPETICIONES}).sort_values(
            "personas", ascending=False).reset_index(drop=True)


def main() -> None:
    """Mide el precio de la garantia sobre la consulta del libro."""
    log = crear_registro("cap05.presupuesto")
    rng = fijar_semillas()

    df = pd.read_parquet(DATOS).assign(
        provincia=lambda t: t["codigo_postal"].astype(str).str[:2])
    conteos = tabla_verdadera(df)
    log.info("consulta del libro: prevalencia de %d diagnosticos en "
             "%d provincias = %d casillas sobre %d personas",
             conteos.shape[1], conteos.shape[0], conteos.size,
             len(df))
    log.info("vecindad acotada (sustituir a una persona): sensibilidad "
             "L1 = %.0f. Por composicion PARALELA la tabla entera "
             "cuesta epsilon una sola vez.", SENS)

    # ── 1) la curva del precio ──────────────────────────────────────
    cur = curva(conteos, rng, log)
    for _, f in cur.iterrows():
        log.info("  epsilon=%-5.2f error %6.3f puntos (p95: %6.3f) · "
                 "provincias sin prevalencia definida: %.1f%%",
                 f["epsilon"], f["error"], f["error_p95"],
                 f["pct_indefinidas"])
    rotas = cur[cur["pct_indefinidas"] > 0]
    if len(rotas):
        log.info("  Por debajo de epsilon=%.2f hay provincias que "
                 "salen enteras a cero: su prevalencia no es imprecisa, "
                 "es inexistente. El error medio de esa zona esta "
                 "calculado solo sobre las que sobreviven, asi que "
                 "MEJORA la cifra mientras empeora la publicacion.",
                 float(rotas["epsilon"].max()))
    SAL_CURVA.parent.mkdir(parents=True, exist_ok=True)
    cur.to_csv(SAL_CURVA, index=False)
    log.info("guardado %s", SAL_CURVA.relative_to(RAIZ))

    # ── 2) contra las dos respuestas anteriores ─────────────────────
    ref = {}
    if ESCALERA.exists():
        esc = pd.read_csv(ESCALERA)
        mejor = esc.loc[esc["error_consulta"].idxmin()]
        ref["cap. 3, generalizacion global"] = (
            float(mejor["error_consulta"]),
            f"peldano «{mejor['peldano']}», suprime el "
            f"{100 * mejor['supresion_k5']:.1f}% de las filas")
    if MONDRIAN.exists():
        ref["cap. 4, Mondrian por provincia"] = (
            0.00, "sin suprimir nada")
    log.info("la MISMA consulta, respondida tres veces:")
    for nombre, (err, nota) in ref.items():
        log.info("  %-32s %5.2f puntos · %s · SIN garantia formal",
                 nombre, err, nota)
    objetivo = max(e for e, _ in ref.values()) if ref else 0.5
    alcanza = cur[cur["error"] <= objetivo]
    if len(alcanza):
        eps_min = float(alcanza["epsilon"].min())
        log.info("  privacidad diferencial              iguala esos "
                 "%.2f puntos con epsilon=%.2f, y su garantia vale "
                 "aunque el adversario conozca las otras %d filas.",
                 objetivo, eps_min, len(df) - 1)
    else:
        log.info("  privacidad diferencial              no baja de "
                 "%.2f puntos en la rejilla probada (minimo %.3f a "
                 "epsilon=%.0f)", objetivo, cur["error"].min(),
                 cur.loc[cur["error"].idxmin(), "epsilon"])

    # ── 3) el posprocesado es gratis ────────────────────────────────
    pos = efecto_posprocesado(conteos, 1.0, rng)
    log.info("posprocesado a epsilon=1 (no consume presupuesto):")
    for _, f in pos.iterrows():
        log.info("  %-32s error %6.3f puntos · casillas negativas "
                 "%.1f%%", f["posprocesado"], f["error"],
                 f["pct_negativos"])
    log.info("  Mejora del %.0f%% sin gastar un epsilon mas: es la "
             "inmunidad al posprocesado, no un truco.",
             100 * (1 - pos["error"].min() / pos["error"].max()))
    pos.to_csv(SAL_POSPRO, index=False)
    log.info("guardado %s", SAL_POSPRO.relative_to(RAIZ))

    # ── 4) quien paga la garantia ───────────────────────────────────
    prov = reparto_por_provincia(conteos, 1.0, rng)
    SAL_PROV.parent.mkdir(parents=True, exist_ok=True)
    prov.to_csv(SAL_PROV, index=False)
    mayor, menor = prov.iloc[0], prov.iloc[-1]
    log.info("reparto del coste a epsilon=1, por tamano de provincia:")
    for _, f in pd.concat([prov.head(3), prov.tail(3)]).iterrows():
        log.info("  provincia %s · %5d personas · error %7.3f puntos",
                 f["provincia"], int(f["personas"]), f["error_puntos"])
    log.info("  De %.3f puntos en la mayor (%d personas) a %.3f en la "
             "menor (%d): un factor %.0f. El epsilon es el mismo para "
             "todos; el ERROR no. Publicar una tabla con DP obliga a "
             "decidir a que nivel de agregacion deja de ser util, y "
             "esa decision es de politica de difusion, no tecnica.",
             mayor["error_puntos"], int(mayor["personas"]),
             menor["error_puntos"], int(menor["personas"]),
             menor["error_puntos"] / mayor["error_puntos"])
    log.info("guardado %s", SAL_PROV.relative_to(RAIZ))

    muestra_final(prov, log)


if __name__ == "__main__":
    main()
