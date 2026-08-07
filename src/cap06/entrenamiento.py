"""el coste de entrenar con garantia, medido en las dos monedas.

Este es el guion central del capitulo. Entrena la MISMA red sobre la
MISMA tarea sin privacidad y con DP-SGD a varios presupuestos, y mide
en cada caso las dos cosas que hay que poner en la balanza:

- LO QUE SE PIERDE: exactitud sobre datos no vistos.
- LO QUE SE GANA: cuanto deja de funcionar el ataque de inferencia de
  pertenencia del capitulo 2, con el mismo criterio de perdida y la
  misma metrica —AUC y, sobre todo, TPR a tasas de falsa alarma muy
  bajas, que es donde un ataque se vuelve peligroso—.

Hay una trampa en esa comparacion que conviene desactivar antes de
medir nada. DP-SGD no anade solo ruido: tambien RECORTA los
gradientes, y el recorte por si solo cambia como aprende la red —a C
pequeno equivale a normalizar el gradiente—. Comparar «sin DP» contra
«DP» atribuye al presupuesto lo que hizo el recorte, y sobre esta
tarea el saldo llega a salir invertido. Por eso se miden TRES
regimenes: descenso corriente, recorte SIN ruido (que no da ninguna
garantia y sirve de referencia) y DP-SGD completo.

Y cada uno con su propia tasa de aprendizaje, elegida sobre un
conjunto de validacion apartado del de entrenamiento. Reutilizar la
tasa del regimen corriente en el privado es el error de ajuste mas
comun del capitulo, y produce modelos mucho peores de lo que el
presupuesto permitia.

Escribe dos CSV en data/processed/ para las figuras del libro.

Uso: python3 src/cap06/entrenamiento.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from comun.determinismo import fijar_semillas
from comun.modelo_tabular import (auc_pertenencia, dispositivo,
                                  entrenar_red, exactitud, tpr_a_fpr)
from comun.registro import crear_registro, muestra_final, progreso

TAREA = RAIZ / "data" / "processed" / "tarea_riesgo.parquet"
SAL_CURVA = RAIZ / "data" / "processed" / "dp_sgd_curva.csv"
SAL_ROC = RAIZ / "data" / "processed" / "dp_sgd_roc.csv"

LOTE = 512
EPOCAS = 30
C = 1.0
DELTA = 1e-5
TASAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0]
PRESUPUESTOS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
VALIDACION = 0.2
SEMILLAS = [42, 43, 44]   # el descenso sin recorte es inestable a
                          # tasas altas: una sola ejecucion puede
                          # converger o divergir segun la semilla


def cargar_tarea() -> tuple:
    """Carga la tarea derivada y su particion.

    Returns:
        Cuaterna (X dentro, y dentro, X fuera, y fuera).

    Raises:
        SystemExit: si la tarea no se ha generado todavia.
    """
    if not TAREA.exists():
        raise SystemExit("falta la tarea: ejecuta primero "
                         "python3 src/cap06/derivar_tarea.py")
    df = pd.read_parquet(TAREA)
    cols = [c for c in df.columns if c.startswith("x_")]
    x = df[cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int64)
    dentro = df["particion"].to_numpy() == "entrenamiento"
    return x[dentro], y[dentro], x[~dentro], y[~dentro]


def elegir_tasa(x: np.ndarray, y: np.ndarray, clases: int,
                eps: float | None, sigma: float | None, disp,
                rng: np.random.Generator) -> tuple[float, float]:
    """Elige la tasa de aprendizaje sobre un conjunto de validacion.

    Se aparta una parte del conjunto de ENTRENAMIENTO —nunca del de
    prueba, que se reserva para la cifra final y para el ataque— y se
    prueba cada tasa. El presupuesto de privacidad que consume esta
    busqueda se comenta en el texto: no es cero, y casi nadie lo
    contabiliza.

    Args:
        x: caracteristicas de entrenamiento.
        y: etiquetas de entrenamiento.
        clases: numero de clases.
        eps: presupuesto, si el regimen es privado.
        sigma: multiplicador de ruido explicito, si se da.
        disp: dispositivo de computo.
        rng: generador sembrado.

    Returns:
        Par (mejor tasa, exactitud que alcanzo en validacion).
    """
    corte = int(len(y) * (1 - VALIDACION))
    orden = rng.permutation(len(y))
    ent, val = orden[:corte], orden[corte:]
    mejor, mejor_exac = TASAS[0], -1.0
    for lr in TASAS:
        # se promedia sobre semillas: elegir por una sola ejecucion
        # premia a la configuracion con mas suerte, no a la mejor
        exac = np.mean([
            exactitud(entrenar_red(x[ent], y[ent], clases, eps, disp,
                                   lote=LOTE, epocas=EPOCAS,
                                   aprendizaje=lr, c=C, delta=DELTA,
                                   sigma=sigma, semilla=s)[0],
                       x[val], y[val], disp)
            for s in SEMILLAS])
        if exac > mejor_exac:
            mejor, mejor_exac = lr, float(exac)
    return mejor, mejor_exac


def main() -> None:
    """Barre los tres regimenes y mide exactitud y ataque."""
    log = crear_registro("cap06.entrenamiento")
    rng = fijar_semillas()
    disp = dispositivo()

    x_in, y_in, x_out, y_out = cargar_tarea()
    clases = int(max(y_in.max(), y_out.max()) + 1)
    mayoritaria = float(np.bincount(y_out).max() / len(y_out))
    log.info("tarea: %d dentro del entrenamiento · %d fuera · %d "
             "caracteristicas · %d clases · clase mayoritaria %.3f · "
             "dispositivo %s", len(y_in), len(y_out), x_in.shape[1],
             clases, mayoritaria, disp)
    log.info("configuracion comun: lote=%d · %d epocas · C=%.1f · "
             "delta=%.0e · la tasa se elige por regimen sobre un "
             "%.0f%% de validacion · todo promediado sobre %d "
             "semillas", LOTE, EPOCAS, C, DELTA, 100 * VALIDACION,
             len(SEMILLAS))

    # (nombre, epsilon, sigma explicito)
    escenarios = [("sin DP, sin recorte", None, None),
                  ("recorte sin ruido", None, 0.0)]
    escenarios += [(f"eps={e:g}", e, None) for e in PRESUPUESTOS]

    filas, curvas = [], []
    for nombre, eps, sigma in progreso(escenarios, len(escenarios),
                                       log, cada=1,
                                       tarea="regimenes"):
        lr, exac_val = elegir_tasa(x_in, y_in, clases, eps, sigma,
                                   disp, rng)
        medidas, fpr, tpr = [], None, None
        for s in SEMILLAS:
            modelo, gastado, segundos = entrenar_red(
                x_in, y_in, clases, eps, disp, lote=LOTE,
                epocas=EPOCAS, aprendizaje=lr, c=C, delta=DELTA,
                sigma=sigma, semilla=s)
            a, f, tp = auc_pertenencia(modelo, x_in, y_in, x_out,
                                       y_out, disp)
            medidas.append({
                "exactitud_dentro": exactitud(modelo, x_in, y_in,
                                              disp),
                "exactitud_fuera": exactitud(modelo, x_out, y_out,
                                             disp),
                "mia_auc": a,
                "mia_tpr_1pct": tpr_a_fpr(f, tp, 0.01),
                "mia_tpr_01pct": tpr_a_fpr(f, tp, 0.001),
                "segundos": segundos})
            if fpr is None:          # la curva de la figura, una sola
                fpr, tpr = f, tp
        m = pd.DataFrame(medidas)
        e_in = float(m["exactitud_dentro"].mean())
        e_out = float(m["exactitud_fuera"].mean())
        auc = float(m["mia_auc"].mean())
        filas.append({
            "escenario": nombre, "lr": lr,
            "exactitud_validacion": exac_val,
            "epsilon": np.inf if eps is None else eps,
            "epsilon_gastado": gastado,
            "exactitud_dentro": e_in, "exactitud_fuera": e_out,
            "exactitud_fuera_sd": float(m["exactitud_fuera"].std()),
            "brecha": e_in - e_out, "mia_auc": auc,
            "mia_tpr_1pct": float(m["mia_tpr_1pct"].mean()),
            "mia_tpr_01pct": float(m["mia_tpr_01pct"].mean()),
            "segundos": float(m["segundos"].mean())})
        paso = max(1, len(fpr) // 200)
        curvas.append(pd.DataFrame({"escenario": nombre,
                                    "fpr": fpr[::paso],
                                    "tpr": tpr[::paso]}))
        log.info("  %-20s lr=%-5g exactitud fuera %.3f ± %.3f "
                 "(dentro %.3f, brecha %+.3f) · MIA AUC %.3f",
                 nombre, lr, e_out, filas[-1]["exactitud_fuera_sd"],
                 e_in, e_in - e_out, auc)

    tabla = pd.DataFrame(filas)
    SAL_CURVA.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SAL_CURVA, index=False)
    pd.concat(curvas).to_csv(SAL_ROC, index=False)

    sin_dp = tabla.iloc[0]
    recorte = tabla.iloc[1]
    con_dp = tabla.iloc[2:]
    log.info("lecturas:")
    log.info("  El descenso corriente, con su mejor tasa (%g), acierta "
             "%.3f fuera. Anadir SOLO el recorte a C=%.1f —sin una "
             "gota de ruido y sin ninguna garantia— sube a %.3f: "
             "%+.1f puntos. El recorte no es una medida de privacidad, "
             "es una decision de optimizacion, y sobre esta tarea "
             "AYUDA.", sin_dp["lr"], sin_dp["exactitud_fuera"], C,
             recorte["exactitud_fuera"],
             100 * (recorte["exactitud_fuera"]
                    - sin_dp["exactitud_fuera"]))
    log.info("  El coste de la GARANTIA es por tanto la distancia "
             "entre el recorte sin ruido y DP-SGD, no entre DP-SGD y "
             "el descenso corriente: va de %+.1f puntos a epsilon=%g "
             "hasta %+.1f puntos a epsilon=%g.",
             100 * (con_dp.iloc[0]["exactitud_fuera"]
                    - recorte["exactitud_fuera"]),
             con_dp.iloc[0]["epsilon"],
             100 * (con_dp.iloc[-1]["exactitud_fuera"]
                    - recorte["exactitud_fuera"]),
             con_dp.iloc[-1]["epsilon"])
    log.info("  Compararlo contra el descenso corriente daria %+.1f "
             "puntos, es decir, la conclusion ABSURDA de que la "
             "privacidad diferencial mejora el modelo. Es el error de "
             "atribucion que este guion existe para evitar.",
             100 * (con_dp.iloc[-1]["exactitud_fuera"]
                    - sin_dp["exactitud_fuera"]))
    log.info("  La brecha entre dentro y fuera, que es lo que el "
             "ataque explota, crece con el presupuesto: %+.3f a "
             "epsilon=%g y %+.3f a epsilon=%g. Sin garantia ninguna "
             "es %+.3f.", con_dp.iloc[0]["brecha"],
             con_dp.iloc[0]["epsilon"], con_dp.iloc[-1]["brecha"],
             con_dp.iloc[-1]["epsilon"], recorte["brecha"])
    log.info("  El ataque de perdida, en cambio, no llega a funcionar "
             "en NINGUN regimen: su AUC va de %.3f a %.3f. Sobre esta "
             "tarea el modelo generaliza y no memoriza, y sin "
             "memorizacion no hay senal que atacar. No es un merito de "
             "DP-SGD: el modelo sin ninguna proteccion tambien "
             "resiste. Lo que esto demuestra es que un ataque que "
             "falla no mide privacidad, exactamente como la auditoria "
             "del vector disperso del capitulo 5.",
             tabla["mia_auc"].min(), tabla["mia_auc"].max())
    log.info("  Coste en tiempo: %.1f s el descenso corriente frente a "
             "%.1f s con gradiente por ejemplo, un factor %.1f.",
             sin_dp["segundos"], con_dp["segundos"].mean(),
             con_dp["segundos"].mean() / max(sin_dp["segundos"], 1e-9))
    log.info("guardado %s y %s", SAL_CURVA.relative_to(RAIZ),
             SAL_ROC.relative_to(RAIZ))

    muestra_final(tabla, log)


if __name__ == "__main__":
    main()
