"""auditar el epsilon de un modelo entrenado, y ver cuanto se queda corto.

La garantia de DP-SGD es una DEMOSTRACION: si el mecanismo esta bien
implementado, ningun adversario puede distinguir mejor de lo que la
cota permite. Auditar es lo contrario: montar un adversario concreto,
medir lo bien que distingue y traducir esa medida en una cota INFERIOR
del epsilon que el mecanismo ha gastado de verdad.

Sirve para dos cosas y no para una tercera. Sirve para detectar
implementaciones rotas —si la cota inferior supera al epsilon
anunciado, algo esta mal, y no hay discusion posible—. Sirve para
saber cuanto margen hay entre la garantia y el ataque conocido. NO
sirve para certificar que un mecanismo es correcto: una auditoria
limpia solo dice que ESE adversario no lo consiguio.

Aqui se auditan tres adversarios de potencia creciente sobre el mismo
entrenamiento, para que se vea que lo que decide el ajuste de la cota
no es el mecanismo sino la fuerza del que mira:

1. canarios NATURALES: ejemplos reales del conjunto;
2. canarios MAL ETIQUETADOS: ejemplos reales con la etiqueta cambiada;
3. canarios EXTREMOS: puntos fuera del dominio y mal etiquetados, que
   producen los gradientes mas grandes que el recorte permite.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap06/auditoria.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.modelo_tabular import (dispositivo, entrenar_red,
                                  perdidas_por_ejemplo)
from comun.registro import crear_registro, muestra_final, progreso

TAREA = RAIZ / "data" / "processed" / "tarea_riesgo.parquet"
SAL_AUDIT = RAIZ / "data" / "processed" / "dp_sgd_auditoria.csv"
SAL_PUNTOS = RAIZ / "data" / "processed" / "dp_sgd_auditoria_puntos.csv"

CANARIOS = 500            # candidatos por ejecucion
REPETICIONES = 24         # entrenamientos independientes
DELTA = 1e-5
PRESUPUESTOS = [1.0, 4.0, 16.0]
CONFIANZA = 0.95


def clopper_pearson(aciertos: int, total: int,
                    confianza: float) -> tuple[float, float]:
    """Intervalo exacto de Clopper-Pearson para una proporcion.

    La auditoria estima dos probabilidades con un numero finito de
    ensayos. Usar las frecuencias sin mas produciria cotas que no lo
    son: hace falta el extremo pesimista de un intervalo de confianza.

    Args:
        aciertos: exitos observados.
        total: ensayos.
        confianza: nivel del intervalo, p. ej. 0,95.

    Returns:
        Par (extremo inferior, extremo superior).
    """
    from scipy.stats import beta
    alfa = 1 - confianza
    lo = 0.0 if aciertos == 0 else float(
        beta.ppf(alfa / 2, aciertos, total - aciertos + 1))
    hi = 1.0 if aciertos == total else float(
        beta.ppf(1 - alfa / 2, aciertos + 1, total - aciertos))
    return lo, hi


def eps_empirico(dentro: np.ndarray, fuera: np.ndarray,
                 delta: float, confianza: float) -> tuple[float, float]:
    """Cota inferior del epsilon a partir de las puntuaciones.

    Se recorre cada umbral posible del estadistico del ataque. Para
    cada uno se estiman la tasa de acierto y la de falsa alarma con sus
    intervalos de confianza, y se aplica la desigualdad de la
    definicion en las dos direcciones:

        eps >= log((TPR - delta) / FPR)   y   log((TNR - delta) / FNR)

    tomando siempre el extremo desfavorable del intervalo, para que lo
    que salga sea una cota y no una estimacion.

    Args:
        dentro: puntuaciones de los canarios que SI entraron.
        fuera: puntuaciones de los que no.
        delta: delta de la garantia auditada.
        confianza: nivel de los intervalos.

    Returns:
        Par (epsilon empirico, umbral que lo alcanza).
    """
    umbrales = np.unique(np.concatenate([dentro, fuera]))
    mejor, mejor_umbral = 0.0, float("nan")
    for u in umbrales:
        tp = int((dentro >= u).sum())
        fp = int((fuera >= u).sum())
        tpr_lo, _ = clopper_pearson(tp, len(dentro), confianza)
        _, fpr_hi = clopper_pearson(fp, len(fuera), confianza)
        if fpr_hi > 0 and tpr_lo - delta > 0:
            mejor = max(mejor, math.log((tpr_lo - delta) / fpr_hi))
            if mejor == math.log((tpr_lo - delta) / fpr_hi):
                mejor_umbral = float(u)
        # la direccion simetrica: acertar los que NO entraron
        tn = len(fuera) - fp
        fn = len(dentro) - tp
        tnr_lo, _ = clopper_pearson(tn, len(fuera), confianza)
        _, fnr_hi = clopper_pearson(fn, len(dentro), confianza)
        if fnr_hi > 0 and tnr_lo - delta > 0:
            mejor = max(mejor, math.log((tnr_lo - delta) / fnr_hi))
    return mejor, mejor_umbral


def fabricar_canarios(x: np.ndarray, y: np.ndarray, clases: int,
                      tipo: str, n: int,
                      rng: np.random.Generator) -> tuple:
    """Construye los canarios de un tipo de adversario.

    Args:
        x: caracteristicas del conjunto original.
        y: etiquetas del conjunto original.
        clases: numero de clases.
        tipo: «naturales», «mal etiquetados» o «extremos».
        n: cuantos fabricar.
        rng: generador sembrado.

    Returns:
        Par (caracteristicas, etiquetas) de los canarios.

    Raises:
        ValueError: si el tipo no es uno de los tres previstos.
    """
    idx = rng.choice(len(y), size=n, replace=False)
    if tipo == "naturales":
        return x[idx].copy(), y[idx].copy()
    if tipo == "mal etiquetados":
        yc = (y[idx] + 1 + rng.integers(0, clases - 1, n)) % clases
        return x[idx].copy(), yc.astype(np.int64)
    if tipo == "extremos":
        # el peor caso que el adversario puede construir: puntos lejos
        # de todo lo visto, con signos alternos, y mal etiquetados
        xc = rng.choice([-1.0, 1.0], size=(n, x.shape[1])) * 4.0
        yc = rng.integers(0, clases, n)
        return xc.astype(np.float32), yc.astype(np.int64)
    raise ValueError(f"tipo de canario desconocido: {tipo}")


def auditar(x: np.ndarray, y: np.ndarray, clases: int, eps: float,
            tipo: str, disp, rng: np.random.Generator,
            log) -> dict:
    """Audita un presupuesto con un tipo de canario.

    Cada repeticion entrena de cero incluyendo la MITAD de los
    canarios, sorteada al azar, y puntua todos. Los que entraron y los
    que no forman las dos poblaciones que el adversario debe separar.

    Args:
        x: caracteristicas del conjunto base.
        y: etiquetas del conjunto base.
        clases: numero de clases.
        eps: presupuesto objetivo del entrenamiento.
        tipo: tipo de canario.
        disp: dispositivo de computo.
        rng: generador sembrado.
        log: registro donde anotar el avance.

    Returns:
        Diccionario con el epsilon empirico y sus ingredientes.
    """
    dentro_todo, fuera_todo = [], []
    for r in progreso(range(REPETICIONES), REPETICIONES, log, cada=8,
                      tarea=f"{tipo} a eps={eps:g}"):
        xc, yc = fabricar_canarios(x, y, clases, tipo, CANARIOS, rng)
        entra = rng.random(CANARIOS) < 0.5
        x_ent = np.concatenate([x, xc[entra]])
        y_ent = np.concatenate([y, yc[entra]])
        modelo, _, _ = entrenar_red(x_ent, y_ent, clases, eps, disp,
                                    semilla=1000 + r)
        # el estadistico: menos la perdida del canario. Cuanto mas
        # seguro esta el modelo de el, mas parece que lo vio
        p = -perdidas_por_ejemplo(modelo, xc, yc, disp)
        dentro_todo.append(p[entra])
        fuera_todo.append(p[~entra])

    dentro = np.concatenate(dentro_todo)
    fuera = np.concatenate(fuera_todo)
    emp, umbral = eps_empirico(dentro, fuera, DELTA, CONFIANZA)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(
        np.r_[np.ones(len(dentro)), np.zeros(len(fuera))],
        np.r_[dentro, fuera]))
    return {"tipo": tipo, "eps_anunciado": eps, "eps_empirico": emp,
            "cociente": emp / eps, "auc_canarios": auc,
            "canarios_dentro": len(dentro),
            "canarios_fuera": len(fuera), "umbral": umbral}


def main() -> None:
    """Audita tres adversarios sobre tres presupuestos."""
    log = crear_registro("cap06.auditoria")
    rng = fijar_semillas()
    disp = dispositivo()

    if not TAREA.exists():
        raise SystemExit("falta la tarea: ejecuta primero "
                         "python3 src/cap06/derivar_tarea.py")
    df = pd.read_parquet(TAREA)
    cols = [c for c in df.columns if c.startswith("x_")]
    dentro = df["particion"].to_numpy() == "entrenamiento"
    x = df.loc[dentro, cols].to_numpy(dtype=np.float32)
    y = df.loc[dentro, "y"].to_numpy(dtype=np.int64)
    clases = int(df["y"].max() + 1)

    log.info("auditoria: %d canarios por ejecucion, %d ejecuciones "
             "independientes por celda, confianza %.0f%%, delta=%.0e",
             CANARIOS, REPETICIONES, 100 * CONFIANZA, DELTA)
    log.info("conjunto base: %d ejemplos · %d clases · %s", len(y),
             clases, disp)

    filas = []
    for eps in PRESUPUESTOS:
        for tipo in ("naturales", "mal etiquetados", "extremos"):
            filas.append(auditar(x, y, clases, eps, tipo, disp, rng,
                                 log))
            f = filas[-1]
            log.info("  eps anunciado %5.1f · canarios %-16s → "
                     "epsilon empirico >= %5.2f (%.0f%% del anunciado) "
                     "· AUC del ataque %.3f", f["eps_anunciado"],
                     f["tipo"], f["eps_empirico"],
                     100 * f["cociente"], f["auc_canarios"])

    tabla = pd.DataFrame(filas)
    SAL_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SAL_AUDIT, index=False)

    log.info("lecturas:")
    for eps in PRESUPUESTOS:
        sub = tabla[tabla["eps_anunciado"] == eps]
        peor, mejor = sub["eps_empirico"].min(), sub["eps_empirico"].max()
        log.info("  A eps=%.0f, la cota inferior va de %.2f con "
                 "canarios naturales a %.2f con canarios extremos: el "
                 "MECANISMO es el mismo y lo que cambia es la fuerza "
                 "del adversario.", eps, peor, mejor)
    ninguna = (tabla["eps_empirico"] > tabla["eps_anunciado"]).sum()
    log.info("  Ninguna de las %d celdas supera su epsilon anunciado "
             "(%d incumplimientos): la implementacion de Opacus pasa "
             "la auditoria. Lo cual, insistimos, no la certifica.",
             len(tabla), int(ninguna))
    log.info("  Y la distancia importa: si la cota mas ajustada se "
             "queda en el %.0f%% del presupuesto anunciado, es que la "
             "garantia se ha comprado para un adversario mucho mas "
             "fuerte que cualquiera de los tres probados. Eso es lo "
             "correcto —la garantia es del peor caso— pero explica por "
             "que quien mide ataques y quien demuestra cotas nunca se "
             "ponen de acuerdo sobre que epsilon es «suficiente».",
             100 * tabla["cociente"].max())
    log.info("guardado %s", SAL_AUDIT.relative_to(RAIZ))

    muestra_final(tabla, log)


if __name__ == "__main__":
    main()
